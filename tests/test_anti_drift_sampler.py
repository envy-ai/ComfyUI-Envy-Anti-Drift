import importlib.util
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import comfy.cli_args

comfy.cli_args.args.cpu = True

import pytest
import torch
from PIL import Image


PACK_ROOT = Path(__file__).resolve().parents[1]


def _enable_corrections(correction, count=1):
    correction.selected_intervals = [(float("inf"), float("-inf"))] * count
    return correction


def _textured_image(height, width, seed=0):
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.rand((1, 3, 16, 16), generator=generator)
    smooth = torch.nn.functional.interpolate(
        coarse, size=(height, width), mode="bilinear", align_corners=False
    )
    detail = torch.rand((1, 3, height, width), generator=generator) * 0.04
    image = smooth * 0.8 + detail
    return image.permute(0, 2, 3, 1).contiguous()


def _translate(image, dy, dx):
    height, width = image.shape[1:3]
    translated = torch.zeros_like(image)
    source_y = slice(max(0, -dy), min(height, height - dy))
    source_x = slice(max(0, -dx), min(width, width - dx))
    target_y = slice(max(0, dy), min(height, height + dy))
    target_x = slice(max(0, dx), min(width, width + dx))
    translated[:, target_y, target_x] = image[:, source_y, source_x]
    return translated


def _fractionally_translate(image, dy, dx):
    batch, height, width, _ = image.shape
    y, x = torch.meshgrid(
        torch.arange(height, dtype=image.dtype),
        torch.arange(width, dtype=image.dtype),
        indexing="ij",
    )
    grids = []
    for _ in range(batch):
        grid_x = (x - dx + 0.5) * (2 / width) - 1
        grid_y = (y - dy + 0.5) * (2 / height) - 1
        grids.append(torch.stack((grid_x, grid_y), dim=-1))
    grid = torch.stack(grids)
    return torch.nn.functional.grid_sample(
        image.movedim(-1, 1), grid, mode="bilinear", padding_mode="zeros", align_corners=False
    ).movedim(1, -1)


@pytest.fixture(scope="module")
def anti_drift():
    import comfy.cli_args

    comfy.cli_args.args.cpu = True
    package_name = "envy_anti_drift_test_package"
    spec = importlib.util.spec_from_file_location(
        package_name,
        PACK_ROOT / "__init__.py",
        submodule_search_locations=[str(PACK_ROOT)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    return importlib.import_module(f"{package_name}.anti_drift")


def test_blend_uses_strict_mean_rgb_threshold_and_unblurred_reference(anti_drift):
    reference = torch.full((1, 1, 1, 3), 0.3)
    blurred_reference = torch.zeros_like(reference)
    below_threshold = torch.full_like(reference, 0.0999)
    at_threshold = torch.full_like(reference, 0.1)
    above_threshold = torch.full_like(reference, 0.1001)

    below_mask = anti_drift.detect_mask(below_threshold, blurred_reference, 10, 0)
    at_mask = anti_drift.detect_mask(at_threshold, blurred_reference, 10, 0)
    above_mask = anti_drift.detect_mask(above_threshold, blurred_reference, 10, 0)
    restored_below = anti_drift.compose_prediction(reference, below_threshold, below_mask.float())
    restored = anti_drift.compose_prediction(reference, at_threshold, at_mask.float())
    kept = anti_drift.compose_prediction(reference, above_threshold, above_mask.float())

    torch.testing.assert_close(restored_below, reference, rtol=0, atol=0)
    torch.testing.assert_close(restored, reference, rtol=0, atol=0)
    torch.testing.assert_close(kept, above_threshold, rtol=0, atol=0)


def test_mask_expansion_default_is_15_pixels_and_batch_local(anti_drift):
    height, width = 41, 43
    reference = torch.zeros((2, height, width, 3))
    prediction = torch.zeros_like(reference)
    prediction[0, 20, 21] = 1
    prediction[1, 5, 7] = 1

    binary = anti_drift.detect_mask(prediction, reference, 50, 0)
    expanded = anti_drift.feather_mask(binary, 0, 15, prediction.dtype)
    original = anti_drift.feather_mask(binary, 0, 0, prediction.dtype)

    assert expanded.shape == (2, height, width, 1)
    assert torch.count_nonzero(original[0]) == torch.count_nonzero(original[1]) == 1
    assert expanded[0, 5, 6, 0] == 1
    assert expanded[0, 35, 36, 0] == 1
    assert expanded[0, 4, 21, 0] == 0
    assert expanded[0, 20, 5, 0] == 0
    assert expanded[0, 36, 21, 0] == 0
    assert expanded[0, 20, 37, 0] == 0
    assert expanded[1, 0, 0, 0] == 1
    assert expanded[1, 20, 22, 0] == 1
    assert expanded[1, 21, 7, 0] == 0
    assert expanded[1, 5, 23, 0] == 0
    assert expanded[1, 30, 30, 0] == 0
    assert expanded[0, 30, 30, 0] == 1


def test_mask_expansion_precedes_gaussian_feathering(anti_drift):
    height, width = 51, 53
    reference = torch.zeros((1, height, width, 3))
    prediction = torch.zeros_like(reference)
    prediction[:, 24:27, 25:28] = 1
    blur_pixels = 1.5
    radius = 3

    blurred_reference = anti_drift.blur_image(reference, blur_pixels)
    binary = anti_drift.detect_mask(prediction, blurred_reference, 10, blur_pixels).to(prediction.dtype)
    expanded = torch.nn.functional.max_pool2d(
        binary.movedim(-1, 1), (1, radius * 2 + 1), stride=1,
        padding=(0, radius),
    )
    expanded = torch.nn.functional.max_pool2d(
        expanded, (radius * 2 + 1, 1), stride=1, padding=(radius, 0)
    ).movedim(1, -1)
    expected = anti_drift.blur_image(expanded, blur_pixels)
    actual = anti_drift.feather_mask(binary, blur_pixels, radius, prediction.dtype)

    torch.testing.assert_close(actual, expected)
    assert torch.any((actual > 0) & (actual < 1))


def test_blurred_comparison_ignores_opposite_high_frequency_texture(anti_drift):
    pattern = torch.tensor([[0.0, 1.0], [1.0, 0.0]]).repeat(64, 64)
    reference = pattern[None, :, :, None].expand(1, -1, -1, 3)
    prediction = (1 - pattern)[None, :, :, None].expand_as(reference)
    blurred_reference = anti_drift.blur_image(reference, 2)

    binary = anti_drift.detect_mask(prediction, blurred_reference, 10, 2)
    mask = anti_drift.feather_mask(binary, 2, 0, prediction.dtype)
    result = anti_drift.compose_prediction(reference, prediction, mask)

    torch.testing.assert_close(
        result[:, 60:68, 60:68], reference[:, 60:68, 60:68], rtol=0, atol=0
    )
    assert mask.shape == (1, 128, 128, 1)


def test_reference_uses_first_image_and_broadcasts_to_prediction_batch(anti_drift):
    image1 = torch.stack(
        [torch.full((2, 3, 3), 0.2), torch.full((2, 3, 3), 0.8)]
    )
    prediction = torch.zeros((3, 2, 3, 3), dtype=torch.float64)

    reference = anti_drift.prepare_reference(image1, prediction)
    binary = anti_drift.detect_mask(prediction, torch.zeros_like(reference), 100, 0)
    result = anti_drift.compose_prediction(reference, prediction, binary.to(prediction.dtype))

    assert reference.shape == (1, 2, 3, 3)
    assert reference.dtype == prediction.dtype
    assert reference.device == prediction.device
    torch.testing.assert_close(result, torch.full_like(prediction, 0.2))


def test_reference_resize_uses_lanczos_center_crop(anti_drift, monkeypatch):
    calls = []

    def upscale(image, width, height, method, crop):
        calls.append((width, height, method, crop))
        return torch.full((1, 3, height, width), 0.4, dtype=image.dtype)

    monkeypatch.setattr(anti_drift.comfy.utils, "common_upscale", upscale)
    source = torch.zeros((1, 4, 6, 3))
    prediction = torch.zeros((2, 3, 5, 3))

    reference = anti_drift.prepare_reference(source, prediction)

    assert calls == [(5, 3, "lanczos", "center")]
    assert reference.shape == (1, 3, 5, 3)
    torch.testing.assert_close(reference, torch.full_like(reference, 0.4))


def test_blur_zero_and_default_blur_are_finite_on_tiny_images(anti_drift):
    image = torch.rand((1, 2, 3, 3))

    assert anti_drift.blur_image(image, 0) is image
    blurred = anti_drift.blur_image(image, 10)

    assert blurred.shape == image.shape
    assert torch.isfinite(blurred).all()
    schema = anti_drift.EnvyKSamplerAdvancedAntiDrift.INPUT_TYPES()["required"]
    assert schema["blur_pixels"][1]["default"] == 6.0
    assert schema["mask_expand_pixels"][0] == "INT"
    assert schema["mask_expand_pixels"][1]["default"] == 15
    assert schema["correction_steps"][1]["default"] == "0.4,0.999"
    assert schema["threshold_percent"][1]["default"] == 5.0
    assert schema["dont_unmask_pixels"][1]["default"] is False
    assert schema["comparison_blur_pixels"][1]["default"] == 2.0
    all_inputs = {**schema, **anti_drift.EnvyKSamplerAdvancedAntiDrift.INPUT_TYPES().get("optional", {})}
    assert all_inputs["align_background"][1]["default"] is True
    assert "align_background" in anti_drift.EnvyKSamplerAdvancedAntiDrift.INPUT_TYPES()["optional"]
    assert "feather" in schema["blur_pixels"][1]["tooltip"].lower()
    assert "detection only" in schema["comparison_blur_pixels"][1]["tooltip"].lower()
    defaults = anti_drift.AntiDriftCorrection(torch.zeros((1, 1, 1, 3)), object(), 6)
    assert defaults.threshold_percent == 5.0
    assert defaults.correction_steps == {8, 19}
    assert defaults.mask_expand_pixels == 15
    assert defaults.dont_unmask_pixels is False
    assert defaults.comparison_blur_pixels == 2.0
    assert defaults.align_background is True


@pytest.mark.parametrize("dy,dx", [(5, -4), (-5, 4)])
def test_registration_recovers_signed_translation_from_unmasked_background(
    anti_drift, dy, dx
):
    height = width = 192
    reference = _textured_image(height, width, seed=9)
    edited = reference.clone()
    conflict = _textured_image(height, width, seed=22) * 2
    edited[:, 60:132, 60:132] = conflict[:, 60:132, 60:132]
    prediction = _translate(edited, dy, dx)
    edit_mask = torch.zeros((1, height, width, 1), dtype=torch.bool)
    edit_mask[:, 48:144, 48:144] = True

    offsets = anti_drift.estimate_translation(reference, prediction, edit_mask)

    assert offsets == [(dy, dx)]
    aligned, coverage = anti_drift.warp_image(prediction, offsets)
    background = (coverage > 0.999) & ~edit_mask
    assert background.sum() > 1000
    torch.testing.assert_close(
        aligned[background.expand_as(aligned)],
        reference[background.expand_as(reference)],
        rtol=0,
        atol=1e-6,
    )


def test_alignment_bootstraps_a_first_pass_without_an_existing_mask(anti_drift):
    height = width = 192
    reference = _textured_image(height, width, seed=15)
    edited = reference.clone()
    edit_mask = torch.zeros((1, height, width, 1), dtype=torch.bool)
    edit_mask[:, 48:144, 48:144] = True
    edited[:, 48:144, 48:144] = 0.95
    prediction = _translate(edited, 2, -2)

    offsets = anti_drift.alignment_offsets(
        reference, prediction, anti_drift.blur_image(reference, 2), 5, 2
    )
    aligned, coverage = anti_drift.warp_image(prediction, offsets)

    assert offsets[0] == pytest.approx((2.0, -2.0), abs=0.05)
    background = (coverage > 0.999) & ~edit_mask
    assert background.sum() > 1000
    torch.testing.assert_close(
        aligned[background.expand_as(aligned)],
        reference[background.expand_as(reference)],
        rtol=0,
        atol=0.02,
    )


def test_registration_leaves_no_shift_flat_and_ambiguous_images_unchanged(anti_drift):
    textured = _textured_image(192, 192, seed=31)
    flat = torch.full_like(textured, 0.4)
    stripes = (torch.arange(192) % 8).float().div(8).reshape(1, 1, 192, 1)
    ambiguous = stripes.expand(1, 192, 192, 3).clone()

    for image in (textured, flat, ambiguous):
        assert anti_drift.estimate_translation(image, image.clone()) == [(0, 0)]

    # Periodic stripes have equally good nonzero translations, so they cannot
    # identify a unique shift.
    periodic_reference = torch.roll(ambiguous, shifts=4, dims=2)
    assert anti_drift.estimate_translation(periodic_reference, ambiguous) == [(0, 0)]


def test_subpixel_estimates_recalculate_masks_only_above_one_third_pixel(anti_drift):
    reference = _textured_image(192, 192, seed=103)
    generator = torch.Generator().manual_seed(104)
    fine_texture = torch.rand(reference.shape, generator=generator) * 0.04 - 0.02
    reference = (reference + fine_texture).clamp(0, 1)
    prediction = torch.cat((
        _fractionally_translate(reference, 0.0, 0.2),
        _fractionally_translate(reference, 0.0, 0.4),
    ))
    offsets = anti_drift.estimate_translation(reference, prediction)
    assert offsets[0] == pytest.approx((0.0, 0.2), abs=0.12)
    assert offsets[1] == pytest.approx((0.0, 0.4), abs=0.12)

    masks, recalculate = anti_drift.output_binary_mask(
        reference, prediction, anti_drift.blur_image(reference, 0), 0.5, 0, offsets
    )
    raw_mask = anti_drift.detect_mask(
        prediction, anti_drift.blur_image(reference, 0), 0.5, 0
    )

    assert recalculate == [False, True]
    assert masks.shape == (2, 192, 192, 1)
    raw_false_positives = raw_mask[1, 32:160, 32:160].sum()
    aligned_false_positives = masks[1, 32:160, 32:160].sum()
    assert raw_false_positives > aligned_false_positives


def test_registration_does_not_use_content_inside_the_edit_mask(anti_drift):
    height = width = 192
    reference = _textured_image(height, width, seed=41)
    mask = torch.zeros((1, height, width, 1), dtype=torch.bool)
    mask[:, 40:152, 40:152] = True
    prediction = _textured_image(height, width, seed=74)
    shifted_reference = _translate(reference, 6, -5)
    prediction[:, 48:144, 48:144] = shifted_reference[:, 48:144, 48:144]

    assert anti_drift.estimate_translation(reference, prediction, mask) == [(0, 0)]


def test_registration_is_batch_local_and_edge_fill_uses_valid_histogram_support(
    anti_drift
):
    height = width = 192
    reference = _textured_image(height, width, seed=83)
    transform = torch.tensor([1.25, 1.1, 1.35]).reshape(1, 1, 1, 3)
    offset = torch.tensor([0.06, 0.08, 0.04]).reshape(1, 1, 1, 3)
    colored = reference * transform + offset
    prediction = torch.cat((_translate(colored, 5, -4), _translate(colored, -5, 4)))

    offsets = anti_drift.estimate_translation(reference, prediction)
    aligned, coverage = anti_drift.warp_image(prediction, offsets)
    assert offsets == [(5, -4), (-5, 4)]
    assert coverage.shape == (2, height, width, 1)
    assert not coverage.all()
    for batch_index in range(2):
        overlap = (coverage[batch_index] > 0.999).expand_as(reference[0])
        torch.testing.assert_close(
            aligned[batch_index][overlap], colored[0][overlap], rtol=0, atol=0
        )
    composite, _ = anti_drift.apply_translation(
        prediction, offsets, reference, torch.zeros((2, height, width, 1))
    )
    for batch_index, column in ((0, 0), (1, -1)):
        matched = anti_drift.match_background_color(
            reference, aligned[batch_index:batch_index + 1],
            (coverage[batch_index:batch_index + 1] > 0.999).logical_not(),
        )
        torch.testing.assert_close(
            composite[batch_index, 80, column], matched[0, 80, column], rtol=0, atol=1e-6
        )


def test_registration_feathers_exposed_edge_and_fills_holes_from_source(
    anti_drift
):
    height, width = 20, 24
    reference = torch.zeros((1, height, width, 3))
    reference[:] = torch.tensor([0.2, 0.3, 0.4])
    generated = torch.empty_like(reference)
    generated[:] = torch.tensor([0.7, 0.8, 0.9])
    prediction = _translate(generated, 0, 3)

    aligned, displayed_mask = anti_drift.apply_translation(
        prediction, [(0, 3)], reference, torch.ones((1, height, width, 1))
    )

    torch.testing.assert_close(aligned[0, 8, width - 4], torch.tensor([0.36666667, 0.46666667, 0.56666667]))
    torch.testing.assert_close(aligned[0, 8, width - 5], torch.tensor([0.53333336, 0.6333333, 0.73333335]))
    torch.testing.assert_close(aligned[0, 8, width - 2], reference[0, 8, width - 2], rtol=0, atol=0)
    torch.testing.assert_close(displayed_mask[0, 8, width - 4, 0], torch.tensor(1 / 3))
    torch.testing.assert_close(displayed_mask[0, 8, width - 5, 0], torch.tensor(2 / 3))
    assert not displayed_mask[0, 8, width - 3:, 0].any()


def test_final_alignment_moves_restored_background_and_edit_once(anti_drift):
    height = width = 192
    reference = _textured_image(height, width, seed=95)
    source_feature = torch.tensor([1.0, 0.0, 1.0])
    reference[:, 29:32, 29:32] = source_feature
    edited = reference.clone()
    mask = torch.zeros((1, height, width, 1), dtype=torch.bool)
    mask[:, 60:132, 60:132] = True
    edited[:, 90:95, 90:95] = 0.95
    prediction = _translate(edited, 4, 5)

    final = anti_drift.finalize_image(
        reference, prediction, mask, 0, 100, 0, comparison_blur_pixels=0,
        align_background=True, dont_unmask_pixels=True,
    )

    torch.testing.assert_close(final[0, 26, 25], source_feature, rtol=0, atol=1e-5)
    assert final[0, 30, 30].mean() < 0.7
    assert final[0, 92, 92].mean() > 0.8
    assert final[0, 98, 98].mean() < 0.8


def test_corrections_retain_display_mask_across_offsets_without_extra_vae_calls(
    anti_drift, monkeypatch
):
    import comfy.model_management
    import comfy.model_prefetch

    monkeypatch.setattr(comfy.model_prefetch, "pause_malloc_graph", nullcontext)
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda *args, **kwargs: None)
    reference = _textured_image(192, 192, seed=117)
    edited = reference.clone()
    edited[:, 84:96, 84:96] = 0.98
    predictions = [_translate(edited, 5, -4), _translate(edited, -3, 2)]
    decoded = []
    encoded = []

    class VAE:
        def decode(self, samples):
            image = predictions[len(decoded)]
            decoded.append(image)
            return image

        def encode(self, image):
            encoded.append(image.clone())
            return torch.zeros((1, 4, 2, 2))

    class Model:
        def process_latent_out(self, samples):
            return samples

        def process_latent_in(self, samples):
            return samples

    correction = anti_drift.AntiDriftCorrection(
        reference, VAE(), 0, 10, 0, dont_unmask_pixels=True,
        comparison_blur_pixels=4, align_background=True,
    )
    _enable_corrections(correction, 2)
    correction.models = [object()]
    correction.memory_required = 1
    correction.minimum_memory_required = 1
    estimated_offsets = []
    estimate = anti_drift.alignment_offsets

    def capture_offsets(*args, **kwargs):
        offsets = estimate(*args, **kwargs)
        estimated_offsets.append(offsets)
        return offsets

    monkeypatch.setattr(anti_drift, "alignment_offsets", capture_offsets)
    args = {"denoised": torch.zeros((1, 4, 2, 2)), "model": Model(), "sigma": torch.tensor([1.0])}
    correction(args)
    first_mask = correction.last_binary_mask.clone()
    correction(args)

    assert estimated_offsets == [[(5, -4)], [(-3, 2)]]
    assert len(decoded) == len(encoded) == 2
    assert first_mask[0, 90, 90, 0]
    assert correction.last_binary_mask[0, 90, 90, 0]
    assert torch.all(correction.last_binary_mask[first_mask])
    for image, displaced_center in ((encoded[0], (99, 81)), (encoded[1], (82, 97))):
        assert image[0, 90, 90].mean() > 0.8
        assert image[0, displaced_center[0], displaced_center[1]].mean() < 0.8


def test_disabled_alignment_keeps_the_legacy_final_composite(anti_drift, monkeypatch):
    prediction = torch.full((1, 3, 4, 3), 0.7)
    reference = torch.zeros_like(prediction)
    mask = torch.ones((1, 3, 4, 1), dtype=torch.bool)

    def unexpected_alignment(*args, **kwargs):
        raise AssertionError("disabled alignment must not register the prediction")

    monkeypatch.setattr(anti_drift, "alignment_offsets", unexpected_alignment)
    final = anti_drift.finalize_image(
        reference, prediction, mask, 0, 5, 0, align_background=False
    )

    torch.testing.assert_close(final, prediction)


def test_correction_normalizes_latents_caches_reference_and_restores_each_prediction(
    anti_drift, monkeypatch
):
    import comfy.model_management
    import comfy.model_prefetch

    monkeypatch.setattr(comfy.model_prefetch, "pause_malloc_graph", nullcontext)
    restored = []
    monkeypatch.setattr(
        comfy.model_management,
        "load_models_gpu",
        lambda models, **kwargs: restored.append((models, kwargs)),
    )
    decoded_inputs = []
    encoded_inputs = []

    class VAE:
        def decode(self, samples):
            decoded_inputs.append(samples.clone())
            return torch.full((1, 2, 2, 3), 0.7)

        def encode(self, image):
            encoded_inputs.append(image.clone())
            return image[..., :1].to(torch.float32).permute(0, 3, 1, 2).expand(-1, 4, -1, -1)

    class Model:
        def process_latent_out(self, samples):
            return samples * 2

        def process_latent_in(self, samples):
            return samples * 3

    vae = VAE()
    model = Model()
    correction = anti_drift.AntiDriftCorrection(
        torch.full((1, 2, 2, 3), 0.25), vae, 0, 100, 0
    )
    _enable_corrections(correction, 2)
    correction.models = [object()]
    correction.memory_required = 10
    correction.minimum_memory_required = 4
    original = torch.ones((1, 4, 2, 2), dtype=torch.float64)
    args = {"denoised": original, "model": model, "sigma": torch.tensor([1.0])}

    first = correction(args)
    cached_reference = correction.reference
    second = correction(args)

    assert len(decoded_inputs) == len(encoded_inputs) == 2
    torch.testing.assert_close(decoded_inputs[0], original * 2)
    torch.testing.assert_close(encoded_inputs[0], torch.full_like(encoded_inputs[0], 0.25))
    torch.testing.assert_close(first, torch.full_like(first, 0.75))
    torch.testing.assert_close(second, first)
    assert first.dtype == original.dtype and first.device == original.device
    assert correction.reference is cached_reference
    assert correction.image1 is None
    assert len(restored) == 2
    assert restored == [
        ([correction.models[0]], {"memory_required": 10, "minimum_memory_required": 4}),
        ([correction.models[0]], {"memory_required": 10, "minimum_memory_required": 4}),
    ]
    assert len(correction.intermediate_frames) == 2
    assert correction.intermediate_frames[0].shape == (1, 2, 4, 3)


def test_correction_reuses_corrected_pixels_for_full_resolution_preview(
    anti_drift, monkeypatch
):
    import comfy.model_management
    import comfy.model_prefetch

    monkeypatch.setattr(comfy.model_prefetch, "pause_malloc_graph", nullcontext)
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda *args, **kwargs: None)
    height, width = 513, 514
    decoded = []
    encoded = []

    class VAE:
        def decode(self, samples):
            decoded.append(samples)
            return torch.full((1, height, width, 3), 0.3)

        def encode(self, image):
            encoded.append(image.clone())
            return torch.zeros((1, 4, height, width))

    class Model:
        def process_latent_out(self, samples):
            return samples

        def process_latent_in(self, samples):
            return samples

    correction = anti_drift.AntiDriftCorrection(
        torch.full((1, height, width, 3), 0.25), VAE(), 0, 10, 0
    )
    _enable_corrections(correction)
    correction.models = [object()]
    correction.memory_required = 1
    correction.minimum_memory_required = 1
    correction({"denoised": torch.zeros((1, 4, height, width)), "model": Model(), "sigma": torch.tensor([1.0])})

    assert len(decoded) == len(encoded) == 1
    torch.testing.assert_close(encoded[0], torch.full_like(encoded[0], 0.25))
    assert correction.latest_preview.size == (width, height)
    assert correction.latest_preview.getpixel((0, 0)) == (63, 63, 63)
    assert correction.latest_preview.getpixel((0, 0)) != (76, 76, 76)


def test_capture_sampler_replaces_native_callback_with_latest_preview(
    anti_drift, monkeypatch
):
    import comfy.sampler_helpers
    import comfy.utils

    monkeypatch.setattr(comfy.sampler_helpers, "estimate_memory", lambda *args: (1, 1))
    monkeypatch.setattr(comfy.sampler_helpers, "get_additional_models", lambda *args: ([], 0))
    preview = Image.new("RGB", (640, 513), (1, 2, 3))
    correction = anti_drift.AntiDriftCorrection(torch.empty(0), object(), 0, 10, 0)
    correction.latest_preview = preview
    model = types.SimpleNamespace(model_dtype=lambda: torch.float32)
    model_wrap = types.SimpleNamespace(
        model_patcher=model, loaded_models=[], conds={}
    )
    progress = []
    monkeypatch.setattr(
        comfy.utils,
        "PROGRESS_BAR_HOOK",
        lambda step, total, item, **kwargs: progress.append((step, total, item, kwargs)),
    )
    native_callback_calls = []

    def native_callback(*args):
        native_callback_calls.append(args)

    def executor(wrap, sigmas, extra_args, callback, noise, *args):
        assert callback is not native_callback
        callback(0, None, None, 2)
        callback(1, None, None, 2)
        return "sampled"

    result = correction.capture_sampler(
        executor,
        model_wrap,
        torch.tensor([1.0, 0.5, 0.0]),
        {},
        native_callback,
        torch.zeros((1, 4, 2, 2)),
    )

    assert result == "sampled"
    assert native_callback_calls == []
    assert [(step, total) for step, total, _, _ in progress] == [(1, 2), (2, 2)]
    assert [item for _, _, item, _ in progress] == [
        ("JPEG", preview, None),
        ("JPEG", preview, None),
    ]
    assert all(kwargs == {"node_id": None} for _, _, _, kwargs in progress)


def test_preview_websocket_preserves_full_resolution_when_max_size_is_none():
    import asyncio
    import io

    from server import BinaryEventTypes, PromptServer

    sent = []

    async def send_bytes(event, data, sid=None):
        sent.append((event, data, sid))

    image = Image.new("RGB", (1600, 900), (10, 20, 30))
    prompt_server = types.SimpleNamespace(send_bytes=send_bytes)

    asyncio.run(PromptServer.send_image(prompt_server, ("JPEG", image, None)))

    assert len(sent) == 1
    event, data, sid = sent[0]
    assert event == BinaryEventTypes.PREVIEW_IMAGE
    assert sid is None
    assert data[:4] == (1).to_bytes(4, "big")
    with Image.open(io.BytesIO(data[4:])) as decoded:
        assert decoded.size == (1600, 900)


def test_clear_releases_reference_and_preview_buffers(anti_drift):
    correction = anti_drift.AntiDriftCorrection(torch.ones(1), object(), 0, 10, 0)
    correction.reference = torch.ones(1)
    correction.blurred_reference = torch.ones(1)
    correction.models = [object()]
    correction.latest_preview = Image.new("RGB", (513, 513))

    correction.clear()

    assert correction.image1 is None
    assert correction.vae is None
    assert correction.reference is None
    assert correction.blurred_reference is None
    assert correction.models is None
    assert correction.latest_preview is None


def test_sampler_wrapper_captures_memory_and_restores_models_before_post_cfg(
    anti_drift, monkeypatch
):
    import comfy.model_management
    import comfy.model_prefetch
    import comfy.patcher_extension
    import comfy.sampler_helpers

    monkeypatch.setattr(comfy.model_prefetch, "pause_malloc_graph", nullcontext)
    events = []
    restored = []

    class ToyModel:
        def model_dtype(self):
            return torch.float32

        def process_latent_out(self, samples):
            events.append("latent_out")
            return samples

        def process_latent_in(self, samples):
            events.append("latent_in")
            return samples

    toy_model = ToyModel()
    secondary_model = object()
    model_wrap = types.SimpleNamespace(
        model_patcher=toy_model,
        loaded_models=[secondary_model],
        conds={"positive": []},
    )
    estimate_args = []

    def estimate_memory(model, noise_shape, conds):
        estimate_args.append((model, noise_shape, conds))
        events.append("estimate")
        return 100, 40

    monkeypatch.setattr(comfy.sampler_helpers, "estimate_memory", estimate_memory)
    monkeypatch.setattr(
        comfy.sampler_helpers,
        "get_additional_models",
        lambda conds, dtype: ([], 7),
    )
    monkeypatch.setattr(
        comfy.model_management,
        "load_models_gpu",
        lambda models, **kwargs: (events.append("restore"), restored.append((models, kwargs))),
    )

    class VAE:
        def decode(self, samples):
            return torch.zeros((1, 2, 2, 3))

        def encode(self, image):
            return torch.zeros((1, 4, 2, 2))

    correction = anti_drift.AntiDriftCorrection(
        torch.zeros((1, 2, 2, 3)), VAE(), 0, 10, 0
    )

    def native_sampler(model_wrap, sigmas, extra_args, callback, noise,
                       latent_image=None, denoise_mask=None, disable_pbar=False):
        events.append("post_cfg")
        _enable_corrections(correction)
        return correction({"denoised": torch.ones((1, 4, 2, 2)), "model": toy_model, "sigma": torch.tensor([1.0])})

    executor = comfy.patcher_extension.WrapperExecutor.new_executor(
        native_sampler, [correction.capture_sampler]
    )
    noise = torch.zeros((1, 4, 2, 2))
    result = executor.execute(model_wrap, torch.tensor([1.0, 0.0]), {}, None, noise)

    assert estimate_args == [(toy_model, noise.shape, model_wrap.conds)]
    assert events.index("estimate") < events.index("post_cfg")
    assert events.index("post_cfg") < events.index("latent_out")
    assert events.index("latent_in") < events.index("restore")
    assert restored == [
        ([toy_model, secondary_model], {"memory_required": 107, "minimum_memory_required": 47})
    ]
    assert result.shape == (1, 4, 2, 2)


def test_correction_reports_vae_latent_shape_mismatch(anti_drift, monkeypatch):
    import comfy.model_management
    import comfy.model_prefetch

    monkeypatch.setattr(comfy.model_prefetch, "pause_malloc_graph", nullcontext)
    restorations = []
    monkeypatch.setattr(
        comfy.model_management,
        "load_models_gpu",
        lambda *args, **kwargs: restorations.append((args, kwargs)),
    )

    class VAE:
        def decode(self, samples):
            return torch.zeros((1, 2, 2, 3))

        def encode(self, image):
            return torch.zeros((1, 3, 2, 2))

    class Model:
        def process_latent_out(self, samples):
            return samples

        def process_latent_in(self, samples):
            return samples

    correction = anti_drift.AntiDriftCorrection(
        torch.zeros((1, 2, 2, 3)), VAE(), 0, 10, 0
    )
    _enable_corrections(correction)
    correction.models = [object()]
    correction.memory_required = 1
    correction.minimum_memory_required = 1

    with pytest.raises(
        ValueError,
        match=r"encoded latent shape \(1, 3, 2, 2\).*sampled latent shape \(1, 4, 2, 2\)",
    ):
        correction({"denoised": torch.zeros((1, 4, 2, 2)), "model": Model(), "sigma": torch.tensor([1.0])})

    assert restorations == []


def test_sampler_returns_eval_major_corrected_and_feathered_mask_panels(
    anti_drift, monkeypatch
):
    import comfy.model_management
    import comfy.model_prefetch
    from nodes import KSamplerAdvanced

    monkeypatch.setattr(comfy.model_prefetch, "pause_malloc_graph", nullcontext)
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda *args, **kwargs: None)
    batch, height, width = 2, 9, 9
    reference = torch.stack(
        [torch.full((height, width, 3), 0.25), torch.full((height, width, 3), 0.9)]
    )
    first_eval = torch.stack(
        [torch.full((height, width, 3), 0.25), torch.full((height, width, 3), 0.8)]
    )
    first_eval[0, 2:7, 2:7] = 0.9
    second_eval = torch.stack(
        [torch.full((height, width, 3), 0.4), torch.full((height, width, 3), 0.2)]
    )
    final_eval = torch.full((batch, height, width, 3), 0.5)
    predictions = [first_eval, second_eval, final_eval]
    decoded_inputs = []
    encoded_inputs = []
    captured = []

    class VAE:
        def decode(self, samples):
            image = predictions[len(decoded_inputs)]
            decoded_inputs.append(image)
            return image

        def encode(self, image):
            encoded_inputs.append(image.clone())
            return torch.zeros((batch, 4, height, width))

    class Patcher:
        def clone(self):
            return self

        def set_model_sampler_post_cfg_function(self, correction):
            self.correction = correction
            captured.append(correction)

        def add_wrapper_with_key(self, *args):
            pass

        def process_latent_out(self, samples):
            return samples

        def process_latent_in(self, samples):
            return samples

    patcher = Patcher()
    noise_mask = torch.full((batch, 1, height, width), 0.5)
    latent = {"samples": torch.ones((batch, 4, height, width)), "noise_mask": noise_mask, "batch_index": [4, 9]}
    native_output = {
        "samples": latent["samples"],
        "noise_mask": noise_mask,
        "batch_index": latent["batch_index"],
    }

    def sample(self, *args, **kwargs):
        patched = args[0]
        _enable_corrections(patched.correction, 2)
        for _ in predictions[:2]:
            patched.correction({
                "denoised": torch.zeros_like(latent["samples"]),
                "model": patched,
                "sigma": torch.tensor([1.0]),
            })
        return (native_output,)

    monkeypatch.setattr(KSamplerAdvanced, "sample", sample)
    result_latent, frames, final_image = anti_drift.EnvyKSamplerAdvancedAntiDrift().sample(
        model=patcher, add_noise="enable", noise_seed=4, steps=2, cfg=1,
        sampler_name="euler", scheduler="normal", positive=[], negative=[],
        latent_image=latent, start_at_step=0, end_at_step=2,
        return_with_leftover_noise="disable", image1=reference, vae=VAE(),
        blur_pixels=1, threshold_percent=10, mask_expand_pixels=0,
        comparison_blur_pixels=1,
    )

    assert result_latent is native_output
    assert result_latent["samples"] is latent["samples"]
    assert result_latent["noise_mask"] is noise_mask
    assert result_latent["batch_index"] is latent["batch_index"]
    assert len(decoded_inputs) == 3 and len(encoded_inputs) == 2
    assert final_image.shape == (batch, height, width, 3)
    assert final_image.dtype == torch.float32 and final_image.device.type == "cpu"
    assert frames.shape == (4, height, width * 2, 3)
    assert frames.device.type == "cpu" and frames.dtype == torch.float32
    assert frames.data_ptr() != decoded_inputs[0].data_ptr()

    expected_frames = []
    has_fractional_mask = False
    for prediction in predictions[:2]:
        binary = anti_drift.detect_mask(
            prediction, anti_drift.blur_image(reference[:1], 1), 10, 1
        )
        mask = anti_drift.feather_mask(binary, 1, 0, prediction.dtype)
        corrected = anti_drift.compose_prediction(reference[:1], prediction, mask)
        has_fractional_mask |= bool(torch.any((mask > 0) & (mask < 1)))
        for batch_index in range(batch):
            expected_frames.append((corrected[batch_index], mask[batch_index].expand(-1, -1, 3)))

    assert has_fractional_mask

    for index, (corrected, mask_rgb) in enumerate(expected_frames):
        torch.testing.assert_close(frames[index, :, :width], corrected)
        torch.testing.assert_close(frames[index, :, width:], mask_rgb)
        torch.testing.assert_close(
            frames[index, :, width:, 0], frames[index, :, width:, 1]
        )

    stored_frames = frames.clone()
    predictions[0].fill_(0.0)
    torch.testing.assert_close(frames, stored_frames)
    assert captured[0].intermediate_frames == []
    assert captured[0].reference is None
    assert captured[0].latest_preview is None


def test_sampler_delegates_once_with_advanced_controls_and_clears_correction(
    anti_drift, monkeypatch
):
    from nodes import KSamplerAdvanced

    calls = []
    controls = None

    class Model:
        def clone(self):
            return PatchedModel()

    class PatchedModel:
        def set_model_sampler_post_cfg_function(self, function):
            self.correction = function

        def add_wrapper_with_key(self, key, name, wrapper):
            self.wrapper = (key, name, wrapper)

    def sample(self, *args, **kwargs):
        calls.append((self, args, kwargs))
        patched = args[0]
        assert patched.correction is not None
        assert patched.wrapper[1] == "envy_anti_drift"
        assert patched.correction.mask_expand_pixels == 4
        return controls

    monkeypatch.setattr(KSamplerAdvanced, "sample", sample)
    samples = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2)
    native_output = {"samples": samples}
    controls = (native_output,)
    original_model = Model()
    latent = {
        "samples": samples,
        "noise_mask": torch.ones((1, 1, 2, 2)),
        "batch_index": [7],
    }
    class VAE:
        def __init__(self):
            self.decode_calls = 0
            self.encode_calls = 0

        def decode(self, samples):
            self.decode_calls += 1
            return torch.zeros((1, 2, 2, 3))

        def encode(self, image):
            self.encode_calls += 1
            raise AssertionError("final image must not be encoded")

    vae = VAE()
    sampler = anti_drift.EnvyKSamplerAdvancedAntiDrift()
    result = sampler.sample(
        model=original_model, add_noise="enable", noise_seed=42, steps=11, cfg=6.5,
        sampler_name="euler", scheduler="normal", positive=["p"], negative=["n"],
        latent_image=latent, start_at_step=2, end_at_step=8,
        return_with_leftover_noise="enable", image1=torch.zeros((1, 2, 2, 3)),
        vae=vae, blur_pixels=10, threshold_percent=10, denoise=0.7,
        mask_expand_pixels=4, correction_steps="",
    )

    assert result[0] is native_output
    assert result[1].shape == (0, 0, 0, 3)
    assert result[2].shape == (1, 2, 2, 3)
    assert result[2].dtype == torch.float32 and result[2].device.type == "cpu"
    torch.testing.assert_close(result[2], torch.zeros_like(result[2]))
    assert vae.decode_calls == 1 and vae.encode_calls == 0
    assert result[1].dtype == torch.float32 and result[1].device.type == "cpu"
    assert anti_drift.EnvyKSamplerAdvancedAntiDrift.RETURN_TYPES == ("LATENT", "IMAGE", "IMAGE")
    assert anti_drift.EnvyKSamplerAdvancedAntiDrift.RETURN_NAMES[-1] == "final_image"
    assert len(calls) == 1
    base_sampler, args, kwargs = calls[0]
    patched_model = args[0]
    assert base_sampler is sampler
    assert patched_model is not original_model
    assert args[1:] == (
        "enable", 42, 11, 6.5, "euler", "normal", ["p"], ["n"], latent,
        2, 8, "enable",
    )
    assert kwargs == {"denoise": 0.7}
    assert latent["samples"] is samples
    torch.testing.assert_close(latent["samples"], torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2))
    assert latent["batch_index"] == [7]
    schema = sampler.INPUT_TYPES()["required"]
    assert "frequency" not in str(schema).lower()
    assert "mask_start_step" not in schema and "mask_end_step" not in schema


def test_sampler_clears_correction_when_native_sampling_raises(anti_drift, monkeypatch):
    from nodes import KSamplerAdvanced

    captured = []

    class Model:
        def clone(self):
            return self

        def set_model_sampler_post_cfg_function(self, correction):
            captured.append(correction)
            self.correction = correction

        def add_wrapper_with_key(self, *args):
            pass

    def fail(self, *args, **kwargs):
        correction = args[0].correction
        correction.reference = torch.ones(1)
        correction.blurred_reference = torch.ones(1)
        correction.latest_preview = Image.new("RGB", (2, 2))
        raise KeyboardInterrupt

    monkeypatch.setattr(KSamplerAdvanced, "sample", fail)
    sampler = anti_drift.EnvyKSamplerAdvancedAntiDrift()
    with pytest.raises(KeyboardInterrupt):
        sampler.sample(
            model=Model(), add_noise="enable", noise_seed=1, steps=1, cfg=1,
            sampler_name="euler", scheduler="normal", positive=[], negative=[],
            latent_image={"samples": torch.zeros((1, 4, 1, 1))}, start_at_step=0,
            end_at_step=1, return_with_leftover_noise="disable",
            image1=torch.zeros((1, 1, 1, 3)), vae=object(),
        )

    assert len(captured) == 1
    assert captured[0].image1 is None
    assert captured[0].vae is None
    assert captured[0].reference is None
    assert captured[0].blurred_reference is None
    assert captured[0].models is None
    assert captured[0].latest_preview is None


def test_sampler_clears_cached_reference_after_vae_error(anti_drift, monkeypatch):
    import comfy.model_prefetch
    from nodes import KSamplerAdvanced

    monkeypatch.setattr(comfy.model_prefetch, "pause_malloc_graph", nullcontext)
    captured = []

    class Model:
        def clone(self):
            return self

        def set_model_sampler_post_cfg_function(self, correction):
            captured.append(correction)

        def add_wrapper_with_key(self, *args):
            pass

        def process_latent_out(self, samples):
            return samples

        def process_latent_in(self, samples):
            return samples

    class VAE:
        def decode(self, samples):
            return torch.zeros((1, 2, 2, 3))

        def encode(self, image):
            raise RuntimeError("VAE encode failed")

    def sample(self, *args, **kwargs):
        model = args[0]
        correction = captured[0]
        correction.models = [model]
        correction.memory_required = 1
        correction.minimum_memory_required = 1
        _enable_corrections(correction)
        correction({"denoised": torch.zeros((1, 4, 2, 2)), "model": model, "sigma": torch.tensor([1.0])})

    monkeypatch.setattr(KSamplerAdvanced, "sample", sample)
    sampler = anti_drift.EnvyKSamplerAdvancedAntiDrift()
    with pytest.raises(RuntimeError, match="VAE encode failed"):
        sampler.sample(
            model=Model(), add_noise="enable", noise_seed=1, steps=1, cfg=1,
            sampler_name="euler", scheduler="normal", positive=[], negative=[],
            latent_image={"samples": torch.zeros((1, 4, 2, 2))}, start_at_step=0,
            end_at_step=1, return_with_leftover_noise="disable",
            image1=torch.zeros((1, 2, 2, 3)), vae=VAE(), blur_pixels=0,
            mask_expand_pixels=0,
        )

    correction = captured[0]
    assert correction.image1 is None
    assert correction.vae is None
    assert correction.reference is None
    assert correction.blurred_reference is None
    assert correction.models is None
    assert correction.latest_preview is None


def test_correction_step_parser_accepts_mixed_separators_and_deduplicates(anti_drift):
    parse = anti_drift.parse_correction_steps

    assert parse("4,0.8", 20) == {4, 16}
    assert parse("16, 0.8", 20) == {16}
    assert parse("0.4, 0.999, 24", 25) == {10, 24}
    assert parse("0.79", 20) == {15}
    assert parse(" ,4,,\n\t0.8,,,16.0 ", 20) == {4, 16}
    assert parse(" ,\n,, ", 20) == set()
    assert parse("4.0", 20) == {4}
    for invalid in ("-1", "1.2", "nan", "inf", "oops"):
        with pytest.raises(ValueError, match="Invalid correction step"):
            parse(invalid, 20)


def test_sparse_steps_use_truncated_original_schedule_once_per_sigma_interval(
    anti_drift, monkeypatch
):
    import comfy.model_management
    import comfy.model_prefetch
    import comfy.sampler_helpers
    import comfy.utils

    monkeypatch.setattr(comfy.model_prefetch, "pause_malloc_graph", nullcontext)
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda *args, **kwargs: None)
    monkeypatch.setattr(comfy.sampler_helpers, "estimate_memory", lambda *args: (1, 1))
    monkeypatch.setattr(comfy.sampler_helpers, "get_additional_models", lambda *args: ([], 0))
    previews = []
    monkeypatch.setattr(
        comfy.utils, "PROGRESS_BAR_HOOK",
        lambda step, total, item, **kwargs: previews.append((step, item)),
    )

    class Model:
        def model_dtype(self):
            return torch.float32

        def process_latent_out(self, samples):
            return samples

        def process_latent_in(self, samples):
            return samples

    class VAE:
        def __init__(self):
            self.decode_calls = 0
            self.encode_calls = 0

        def decode(self, samples):
            self.decode_calls += 1
            return torch.ones((1, 2, 2, 3))

        def encode(self, image):
            self.encode_calls += 1
            return torch.zeros((1, 4, 2, 2))

    model = Model()
    vae = VAE()
    correction = anti_drift.AntiDriftCorrection(
        torch.zeros((1, 2, 2, 3)), vae, 0, 10, 0,
        correction_steps="16, 0.8, 18, 20", start_at_step=15, steps=20,
    )
    assert correction.correction_steps == {16, 18, 20}
    wrap = types.SimpleNamespace(model_patcher=model, loaded_models=[], conds={})
    native_calls = []
    original_callback = lambda *args: native_calls.append(args)
    image_previews = []
    denoised = torch.zeros((1, 4, 2, 2))

    def executor(model_wrap, sigmas, extra_args, callback, noise, *args):
        # Local schedule indices 1 and 3 map back to original steps 16 and 18.
        assert correction.selected_intervals == [(8.0, 6.0), (4.0, 2.0)]
        for local_step, sigma in enumerate((10, 8, 8, 6, 4, 2)):
            original = denoised.clone()
            result = correction({
                "denoised": original, "model": model, "sigma": torch.tensor([sigma])
            })
            active = sigma in (8, 4) and not (sigma == 8 and local_step == 2)
            if active:
                assert result.shape == original.shape
                image_previews.append(correction.latest_preview)
            else:
                assert result is original
                assert correction.latest_preview is None
            callback(local_step, None, None, 5)
        return "native-result"

    result = correction.capture_sampler(
        executor, wrap, torch.tensor([10., 8., 6., 4., 2., 0.]), {},
        original_callback, denoised,
    )

    assert result == "native-result"
    assert vae.decode_calls == vae.encode_calls == 2
    assert len(correction.intermediate_frames) == 2
    assert len(native_calls) == 4
    assert len(image_previews) == 2
    assert len(previews) == 2
    assert all(item[1] == ("JPEG", preview, None) for item, preview in zip(previews, image_previews))
    correction.clear()
    assert correction.selected_intervals == [] and correction.last_binary_mask is None


def test_empty_sparse_steps_skip_intermediate_correction_and_final_mask_falls_back(anti_drift):
    correction = anti_drift.AntiDriftCorrection(
        torch.zeros((1, 2, 2, 3)), object(), 0, 10, 0,
        correction_steps=" ,\n, ,", steps=20, comparison_blur_pixels=0,
    )
    denoised = torch.zeros((1, 4, 2, 2))
    assert correction.selected_intervals == []
    assert correction({"denoised": denoised}) is denoised
    assert correction.intermediate_frames == []

    prediction = torch.zeros((1, 2, 2, 3))
    prediction[:, 0, 0] = 1
    final = anti_drift.finalize_image(
        torch.zeros_like(prediction), prediction, None, 0, 10, 0,
        comparison_blur_pixels=0,
    )
    assert final.dtype == torch.float32 and final.device.type == "cpu"
    torch.testing.assert_close(final[0, 0, 0], torch.ones(3))
    torch.testing.assert_close(final[0, 1, 1], torch.zeros(3))


def test_final_image_reuses_binary_mask_color_matches_background_and_keeps_exact_source(
    anti_drift
):
    values = torch.arange(10, dtype=torch.float32) / 10
    prediction = values.reshape(1, 1, 10, 1).expand(-1, -1, -1, 3).clone()
    reference = torch.empty_like(prediction)
    reference[..., 0] = prediction[..., 0] * 0.5 + 0.2
    reference[..., 1] = prediction[..., 1] * 2 + 0.1
    reference[..., 2] = 0.7
    # A final prediction change in the unmasked area must not replace the binary mask
    # captured by the last intermediate correction.
    binary_mask = torch.zeros((1, 1, 10, 1), dtype=torch.bool)
    binary_mask[0, 0, 5, 0] = True
    source = reference.clone()
    source[0, 0, 0, 0] = 0.1234567

    matched = anti_drift.match_background_color(prediction, source, binary_mask.float())
    assert not torch.equal(matched[0, 0, 5, :2], prediction[0, 0, 5, :2])
    assert matched[0, 0, 5, 2] == prediction[0, 0, 5, 2]

    final = anti_drift.finalize_image(source, prediction, binary_mask, 0, 100, 0)
    assert final.dtype == torch.float32 and final.device.type == "cpu"
    outside = ~binary_mask[..., 0]
    assert torch.equal(final[outside], source[outside])
    torch.testing.assert_close(final[0, 0, 5], matched[0, 0, 5])

    no_background = anti_drift.match_background_color(
        prediction, reference, torch.ones_like(binary_mask, dtype=torch.float32)
    )
    torch.testing.assert_close(no_background, prediction)


def test_dont_unmask_retains_unexpanded_binary_mask_per_batch(anti_drift, monkeypatch):
    import comfy.model_management
    import comfy.model_prefetch

    monkeypatch.setattr(comfy.model_prefetch, "pause_malloc_graph", nullcontext)
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda *args, **kwargs: None)

    height, width = 9, 11
    reference = torch.zeros((1, height, width, 3))
    first = torch.zeros((2, height, width, 3))
    first[0, 2, 2] = 1
    first[1, 2, 8] = 1
    next_prediction = torch.zeros_like(first)
    next_prediction[0, 6, 7] = 1
    next_prediction[1, 6, 3] = 1

    class Model:
        def process_latent_out(self, samples):
            return samples

        def process_latent_in(self, samples):
            return samples

    class VAE:
        def __init__(self, predictions):
            self.predictions = iter(predictions)

        def decode(self, samples):
            return next(self.predictions)

        def encode(self, image):
            return torch.zeros((2, 4, height, width))

    denoised = torch.zeros((2, 4, height, width))
    retained = anti_drift.AntiDriftCorrection(
        reference, VAE([first, next_prediction, next_prediction]), 0, 10, 1,
        dont_unmask_pixels=True, comparison_blur_pixels=0,
    )
    _enable_corrections(retained, 3)
    for _ in range(3):
        retained({"denoised": denoised, "model": Model(), "sigma": torch.tensor([1.0])})
    frames = torch.cat(retained.intermediate_frames)
    masks = frames[:, :, width:, 0]

    assert retained.retained_mask.dtype == torch.bool
    assert retained.retained_mask.shape == (2, height, width, 1)
    assert retained.retained_mask[0, 2, 2, 0]
    assert retained.retained_mask[0, 6, 7, 0]
    assert not retained.retained_mask[1, 2, 2, 0]
    assert retained.retained_mask[1, 2, 8, 0]
    assert torch.count_nonzero(retained.retained_mask[0]) == 2
    assert torch.count_nonzero(masks[4] - masks[2]) == 0
    assert masks[2, 2, 2] == 1 and masks[2, 6, 7] == 1
    assert masks[2, 2, 3] == 1 and masks[2, 2, 0] == 0
    assert masks[2, 2, 2] == masks[2, 2, 2].round()

    shrinking = anti_drift.AntiDriftCorrection(
        reference, VAE([first, next_prediction]), 0, 10, 1,
        dont_unmask_pixels=False, comparison_blur_pixels=0,
    )
    _enable_corrections(shrinking, 2)
    for _ in range(2):
        shrinking({"denoised": denoised, "model": Model(), "sigma": torch.tensor([1.0])})
    shrinking_masks = torch.cat(shrinking.intermediate_frames)[:, :, width:, 0]
    assert shrinking_masks[0, 2, 2] == 1
    assert shrinking_masks[2, 2, 2] == 0
    assert shrinking_masks[2, 6, 7] == 1


def test_post_cfg_encodes_color_matched_prediction_over_exact_source_background(
    anti_drift, monkeypatch
):
    import comfy.model_management
    import comfy.model_prefetch

    monkeypatch.setattr(comfy.model_prefetch, "pause_malloc_graph", nullcontext)
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda *args, **kwargs: None)
    values = torch.tensor([0.1, 0.2, 1.0, 0.4, 0.5])
    prediction = values.reshape(1, 1, 5, 1).expand(-1, -1, -1, 3).clone()
    reference = (values * 0.9 + 0.02).reshape(1, 1, 5, 1).expand(-1, -1, -1, 3).clone()
    reference[0, 0, 2] = 0.29
    encoded_images = []

    class VAE:
        def __init__(self):
            self.decode_calls = 0
            self.encode_calls = 0

        def decode(self, samples):
            self.decode_calls += 1
            return prediction

        def encode(self, image):
            self.encode_calls += 1
            encoded_images.append(image.clone())
            return torch.zeros((1, 4, 1, 5))

    class Model:
        def process_latent_out(self, samples):
            return samples

        def process_latent_in(self, samples):
            return samples

    vae = VAE()
    correction = anti_drift.AntiDriftCorrection(
        reference, vae, 0, 10, 0, comparison_blur_pixels=0,
    )
    _enable_corrections(correction)
    correction.models = [object()]
    correction.memory_required = 0
    correction.minimum_memory_required = 0
    correction({
        "denoised": torch.zeros((1, 4, 1, 5)),
        "model": Model(),
        "sigma": torch.tensor([1.0]),
    })

    assert vae.decode_calls == vae.encode_calls == 1
    assert len(encoded_images) == 1
    binary_mask = torch.zeros((1, 1, 5, 1), dtype=torch.bool)
    binary_mask[0, 0, 2, 0] = True
    expected_matched = anti_drift.match_background_color(
        prediction, reference, binary_mask.float()
    )
    outside = ~binary_mask[..., 0]
    assert torch.equal(encoded_images[0][outside], reference[outside])
    torch.testing.assert_close(encoded_images[0][0, 0, 2], expected_matched[0, 0, 2])
    assert correction.last_binary_mask[0, 0, 2, 0]
    assert torch.count_nonzero(correction.last_binary_mask) == 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_background_percentiles_promote_low_precision_samples_and_restore_dtype(
    anti_drift, dtype
):
    prediction = torch.linspace(0, 1, 24, dtype=dtype).reshape(1, 2, 4, 3)
    reference = prediction * 0.8 + 0.1
    mask = torch.zeros((1, 2, 4, 1), dtype=dtype)

    matched = anti_drift.match_background_color(prediction, reference, mask)

    assert matched.dtype == dtype
    assert torch.isfinite(matched).all()



def test_comparison_blur_changes_detection_not_feathering_or_cached_reference(
    anti_drift, monkeypatch
):
    import comfy.model_management
    import comfy.model_prefetch

    monkeypatch.setattr(comfy.model_prefetch, "pause_malloc_graph", nullcontext)
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda *args, **kwargs: None)
    height = width = 65
    reference = torch.full((1, height, width, 3), 0.2)
    prediction = reference.clone()
    prediction[:, :, width // 2] = 0.8

    binary = anti_drift.detect_mask(prediction, anti_drift.blur_image(reference, 2), 5, 2)
    feathered = anti_drift.feather_mask(binary, 6, 15, prediction.dtype)
    detected = anti_drift.compose_prediction(reference, prediction, feathered)
    binary_suppressed = anti_drift.detect_mask(prediction, anti_drift.blur_image(reference, 6), 5, 6)
    no_feather = anti_drift.feather_mask(binary_suppressed, 6, 15, prediction.dtype)
    restored = anti_drift.compose_prediction(reference, prediction, no_feather)

    assert binary[0, height // 2, width // 2, 0]
    assert not binary_suppressed.any()
    torch.testing.assert_close(
        feathered,
        anti_drift.feather_mask(binary, 6, 15, prediction.dtype),
    )
    assert no_feather.count_nonzero() == 0
    assert detected[0, height // 2, width // 2, 0] > 0.7
    assert restored[0, height // 2, width // 2, 0] == reference[0, height // 2, width // 2, 0]

    decoded = []
    encoded = []

    class VAE:
        def decode(self, samples):
            decoded.append(samples)
            return prediction

        def encode(self, image):
            encoded.append(image.clone())
            return torch.zeros((1, 4, height, width))

    class Model:
        def process_latent_out(self, samples):
            return samples

        def process_latent_in(self, samples):
            return samples

    correction = anti_drift.AntiDriftCorrection(
        reference, VAE(), 6, 5, 15, comparison_blur_pixels=2,
    )
    _enable_corrections(correction)
    correction.models = [object()]
    correction.memory_required = 0
    correction.minimum_memory_required = 0
    correction({
        "denoised": torch.zeros((1, 4, height, width)),
        "model": Model(),
        "sigma": torch.tensor([1.0]),
    })
    assert len(decoded) == len(encoded) == 1
    torch.testing.assert_close(correction.blurred_reference, anti_drift.blur_image(reference, 2))
    torch.testing.assert_close(encoded[0], detected)

    fallback_detected = anti_drift.finalize_image(
        reference, prediction, None, 6, 5, 15, comparison_blur_pixels=2,
    )
    fallback_suppressed = anti_drift.finalize_image(
        reference, prediction, None, 6, 5, 15, comparison_blur_pixels=6,
    )
    assert fallback_detected[0, height // 2, width // 2, 0] > 0.7
    assert fallback_suppressed[0, height // 2, width // 2, 0] == reference[0, height // 2, width // 2, 0]
