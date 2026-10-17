# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Bart

import math
import re

import torch
import torch.nn.functional as F
from kornia.filters import bilateral_blur
from PIL import Image, ImageDraw
from torchvision.transforms.functional import gaussian_blur

import comfy.model_management
import comfy.model_prefetch
import comfy.patcher_extension
import comfy.sampler_helpers
import comfy.utils
import latent_preview
from nodes import KSamplerAdvanced

from .registration import alignment_displacement, apply_alignment, estimate_alignment, match_background_color, shift_mask, warp_image


def blur_image(image, blur_pixels):
    if blur_pixels == 0:
        return image
    radius = math.ceil(3 * blur_pixels)
    kernel_size = radius * 2 + 1
    image = image.movedim(-1, 1)
    # Separate passes keep wide Gaussian kernels inexpensive; replicated edges also work on tiny images.
    image = F.pad(image, (radius, radius, 0, 0), mode="replicate")
    image = gaussian_blur(image, [kernel_size, 1], [blur_pixels, blur_pixels])[:, :, :, radius:-radius]
    image = F.pad(image, (0, 0, radius, radius), mode="replicate")
    image = gaussian_blur(image, [1, kernel_size], [blur_pixels, blur_pixels])[:, :, radius:-radius, :]
    return image.movedim(1, -1).contiguous()


def filter_comparison_image(image, comparison_blur_pixels, comparison_filter, comparison_color_tolerance_percent):
    if comparison_blur_pixels == 0:
        return image
    if comparison_filter == "gaussian":
        return blur_image(image, comparison_blur_pixels)
    if comparison_filter != "bilateral":
        raise ValueError(f"Unknown comparison filter {comparison_filter!r}.")
    if comparison_color_tolerance_percent <= 0:
        raise ValueError("Bilateral comparison color tolerance must be greater than zero.")

    radius = math.ceil(3 * comparison_blur_pixels)
    kernel_size = radius * 2 + 1
    channels = image.shape[-1]
    height, width = image.shape[1:3]
    bytes_per_pixel = image.element_size() * kernel_size ** 2 * (3 * channels + 4)
    tile_budget = 256 * 1024 * 1024
    tile_size = max(1, math.isqrt(tile_budget // bytes_per_pixel) - 2 * radius)
    sigma_color = comparison_color_tolerance_percent / 100 * math.sqrt(3)
    source = image.movedim(-1, 1)
    result = torch.empty_like(source)
    for batch_index in range(image.shape[0]):
        for top in range(0, height, tile_size):
            bottom = min(top + tile_size, height)
            halo_top = max(0, top - radius)
            halo_bottom = min(height, bottom + radius)
            for left in range(0, width, tile_size):
                right = min(left + tile_size, width)
                halo_left = max(0, left - radius)
                halo_right = min(width, right + radius)
                tile = source[batch_index:batch_index + 1, :, halo_top:halo_bottom, halo_left:halo_right]
                filtered = bilateral_blur(tile, (kernel_size, kernel_size), sigma_color, (comparison_blur_pixels, comparison_blur_pixels), border_type="replicate", color_distance_type="l2")
                result[batch_index:batch_index + 1, :, top:bottom, left:right] = filtered[:, :, top - halo_top:bottom - halo_top, left - halo_left:right - halo_left]
    return result.movedim(1, -1).contiguous()


def prepare_reference(image1, prediction):
    reference = image1[:1, :, :, :3]
    height, width = prediction.shape[1:3]
    if reference.shape[1:3] != (height, width):
        reference = comfy.utils.common_upscale(reference.movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)
    return reference.to(device=prediction.device, dtype=prediction.dtype, copy=True)


def feather_mask(binary_mask, blur_pixels, mask_expand_pixels, dtype):
    mask = binary_mask.to(dtype)
    if mask_expand_pixels > 0:
        kernel_size = mask_expand_pixels * 2 + 1
        mask = F.max_pool2d(mask.movedim(-1, 1), (1, kernel_size), stride=1, padding=(0, mask_expand_pixels))
        mask = F.max_pool2d(mask, (kernel_size, 1), stride=1, padding=(mask_expand_pixels, 0)).movedim(1, -1)
    return blur_image(mask, blur_pixels)


def detect_mask(prediction, blurred_reference, threshold_percent, comparison_blur_pixels, comparison_filter="gaussian", comparison_color_tolerance_percent=10.0):
    difference = (filter_comparison_image(prediction, comparison_blur_pixels, comparison_filter, comparison_color_tolerance_percent) - blurred_reference).abs().mean(dim=-1, keepdim=True)
    return difference > threshold_percent / 100


def compose_prediction(reference, prediction, mask):
    matched = match_background_color(prediction, reference, mask)
    return torch.where(mask == 0, reference, torch.lerp(reference, matched, mask))


def parse_correction_steps(value, steps):
    result = set()
    for token in re.split(r"[\s,]+", value.strip()):
        if not token:
            continue
        message = f"Invalid correction step {token!r}. Use nonnegative integer steps or fractions between 0 and 1."
        try:
            step = int(token)
        except ValueError:
            try:
                fraction = float(token)
            except ValueError:
                raise ValueError(message) from None
            if 0 < fraction < 1:
                step = math.floor(fraction * steps)
            elif fraction >= 0 and fraction.is_integer():
                step = int(fraction)
            else:
                raise ValueError(message)
        if step < 0:
            raise ValueError(message)
        result.add(step)
    return result


def alignment_transforms(reference, prediction, blurred_reference, threshold_percent, comparison_blur_pixels, previous_mask=None, registration_masks=None, comparison_filter="gaussian", comparison_color_tolerance_percent=10.0):
    provisional = detect_mask(prediction, blurred_reference, threshold_percent, comparison_blur_pixels, comparison_filter, comparison_color_tolerance_percent)
    if previous_mask is not None:
        provisional |= previous_mask
    if registration_masks is not None:
        registration_masks.append(provisional)
    return estimate_alignment(reference, prediction, provisional)


def output_binary_mask(reference, prediction, blurred_reference, threshold_percent, comparison_blur_pixels, transforms, comparison_filter="gaussian", comparison_color_tolerance_percent=10.0):
    binary = shift_mask(detect_mask(prediction, blurred_reference, threshold_percent, comparison_blur_pixels, comparison_filter, comparison_color_tolerance_percent), transforms)
    recalculate = [displacement > 1 / 3 for displacement in alignment_displacement(transforms, *prediction.shape[1:3])]
    if any(recalculate):
        aligned, coverage = warp_image(prediction, transforms)
        measurement = torch.where(coverage > 0.999, aligned, reference)
        valid = blur_image(coverage, comparison_blur_pixels) > 0.999
        detected = detect_mask(measurement, blurred_reference, threshold_percent, comparison_blur_pixels, comparison_filter, comparison_color_tolerance_percent) & valid
        for index, needs_recalculation in enumerate(recalculate):
            if needs_recalculation:
                binary[index] = detected[index]
    return binary, recalculate


def diagnostic_frames(prediction, composite, display_mask, registration_mask, binary_mask, transforms, recalculate, mask_modes, stage, sigma=None):
    prediction = prediction.detach().to(device="cpu", dtype=torch.float32, copy=True)
    composite = composite.detach().to(device="cpu", dtype=torch.float32)
    display_mask = display_mask.detach().to(device="cpu", dtype=torch.float32)
    registration_mask = registration_mask.detach().to(device="cpu", dtype=torch.float32)
    binary_mask = binary_mask.detach().to(device="cpu", dtype=torch.float32)
    aligned, _ = warp_image(prediction, transforms)
    batch, height, width, _ = prediction.shape
    frames = torch.empty((batch * 3, height + 32, width * 2, 3), device="cpu", dtype=torch.float32)
    panels = (
        ("raw decode | aligned raw (black uncovered)", prediction, aligned),
        ("corrected composite | displayed feathered mask", composite, display_mask),
        ("registration exclusion | binary edit mask", registration_mask, binary_mask),
    )
    for index in range(batch):
        sy, sx, dy, dx = transforms[index]
        stage_label = stage
        if sigma is not None:
            stage_label += f" sigma={sigma:.6g}"
        label = f"{stage_label} batch={index} align=ECC sx={sx:.5f} sy={sy:.5f} dx={dx:+.3f} dy={dy:+.3f} recalc={'yes' if recalculate[index] else 'no'} mask={mask_modes[index]}"
        for panel_index, (title, left, right) in enumerate(panels):
            frame = frames[index * 3 + panel_index]
            header = Image.new("RGB", (width * 2, 32), (24, 24, 24))
            draw = ImageDraw.Draw(header)
            draw.text((2, 2), label, fill="white")
            draw.text((2, 17), title, fill="white")
            frame[:32].copy_(torch.frombuffer(bytearray(header.tobytes()), dtype=torch.uint8).reshape(32, width * 2, 3).float().div_(255))
            frame[32:, :width].copy_(left[index])
            frame[32:, width:].copy_(right[index].expand(height, width, 3))
    return frames


def finalize_image(image1, prediction, binary_mask, blur_pixels, threshold_percent, mask_expand_pixels, comparison_blur_pixels=2.0, dont_unmask_pixels=False, diagnostic_output=None, comparison_filter="gaussian", comparison_color_tolerance_percent=10.0):
    if prediction.ndim != 4 or prediction.shape[-1] < 3:
        raise ValueError("Anti-Drift requires an image VAE that decodes to [batch, height, width, RGB].")
    prediction = prediction[..., :3].detach().to(device="cpu", dtype=torch.float32, copy=True)
    reference = prepare_reference(image1, prediction)
    filtered_reference = filter_comparison_image(reference, comparison_blur_pixels, comparison_filter, comparison_color_tolerance_percent)
    if binary_mask is not None:
        binary_mask = binary_mask.detach().to(device="cpu")
    had_binary_mask = binary_mask is not None
    registration_masks = [] if diagnostic_output is not None else None
    previous_mask = binary_mask
    transforms = alignment_transforms(reference, prediction, filtered_reference, threshold_percent, comparison_blur_pixels, previous_mask, registration_masks, comparison_filter, comparison_color_tolerance_percent)
    recalculate = [displacement > 1 / 3 for displacement in alignment_displacement(transforms, *prediction.shape[1:3])]
    if binary_mask is None or any(recalculate):
        detected, recalculate = output_binary_mask(reference, prediction, filtered_reference, threshold_percent, comparison_blur_pixels, transforms, comparison_filter, comparison_color_tolerance_percent)
        if binary_mask is not None:
            for index, needs_recalculation in enumerate(recalculate):
                if not needs_recalculation:
                    detected[index] = binary_mask[index]
                elif dont_unmask_pixels:
                    detected[index] |= binary_mask[index]
        binary_mask = detected
    mask_modes = []
    for needs_recalculation in recalculate:
        if not had_binary_mask:
            mask_modes.append("detected")
        elif not needs_recalculation:
            mask_modes.append("reused")
        else:
            mask_modes.append("redetected+retained" if dont_unmask_pixels else "redetected")
    output_mask = feather_mask(binary_mask, blur_pixels, mask_expand_pixels, prediction.dtype)
    aligned, display_mask = apply_alignment(prediction, transforms, reference, output_mask)
    composite = compose_prediction(reference, aligned, display_mask)
    if diagnostic_output is not None:
        diagnostic_output.append(diagnostic_frames(prediction, composite, display_mask, registration_masks[0], binary_mask, transforms, recalculate, mask_modes, "Final"))
    return composite


class AntiDriftCorrection:
    def __init__(self, image1, vae, blur_pixels, threshold_percent=5.0, mask_expand_pixels=15, correction_steps="0.4,0.999", dont_unmask_pixels=False, start_at_step=0, steps=20, comparison_blur_pixels=2.0, comparison_filter="gaussian", comparison_color_tolerance_percent=10.0):
        self.image1 = image1
        self.vae = vae
        self.blur_pixels = blur_pixels
        self.comparison_blur_pixels = comparison_blur_pixels
        self.comparison_filter = comparison_filter
        self.comparison_color_tolerance_percent = comparison_color_tolerance_percent
        self.threshold_percent = threshold_percent
        self.mask_expand_pixels = mask_expand_pixels
        self.correction_steps = parse_correction_steps(correction_steps, steps)
        self.dont_unmask_pixels = dont_unmask_pixels
        self.start_at_step = start_at_step
        self.selected_intervals = []
        self.retained_mask = None
        self.last_binary_mask = None
        self.reference = None
        self.blurred_reference = None
        self.models = None
        self.memory_required = None
        self.minimum_memory_required = None
        self.latest_preview = None
        self.intermediate_frames = []

    def capture_sampler(self, executor, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        total_steps = len(sigmas) - 1
        selected = sorted(step - self.start_at_step for step in self.correction_steps if 0 <= step - self.start_at_step < total_steps)
        self.selected_intervals.clear()
        if selected:
            sigma_values = sigmas.detach().cpu().tolist()
            for step in selected:
                upper = sigma_values[step] if step > 0 else math.inf
                lower = sigma_values[step + 1] if step + 1 < total_steps else -math.inf
                self.selected_intervals.append((upper, lower))
        model = model_wrap.model_patcher
        self.models = [model] + list(model_wrap.loaded_models)
        memory_required, minimum_memory_required = comfy.sampler_helpers.estimate_memory(model, noise.shape, model_wrap.conds)
        _, inference_memory = comfy.sampler_helpers.get_additional_models(model_wrap.conds, model.model_dtype())
        self.memory_required = memory_required + inference_memory
        self.minimum_memory_required = minimum_memory_required + inference_memory
        progress = comfy.utils.ProgressBar(len(sigmas) - 1)

        def preview_callback(step, x0, x, total_steps):
            if self.latest_preview is None and callback is not None:
                return callback(step, x0, x, total_steps)
            preview = None if self.latest_preview is None else ("JPEG", self.latest_preview, None)
            progress.update_absolute(step + 1, total_steps, preview)

        return executor(model_wrap, sigmas, extra_args, preview_callback, noise, latent_image, denoise_mask, disable_pbar)

    def __call__(self, args):
        denoised = args["denoised"]
        sigma = args["sigma"].reshape(-1)[0].item() if self.selected_intervals else None
        for index, (upper, lower) in enumerate(self.selected_intervals):
            if lower < sigma <= upper:
                self.selected_intervals.pop(index)
                break
        else:
            self.latest_preview = None
            return denoised
        model = args["model"]
        with comfy.model_prefetch.pause_malloc_graph():
            prediction = self.vae.decode(model.process_latent_out(denoised))
            if prediction.ndim != 4 or prediction.shape[-1] < 3:
                raise ValueError("Anti-Drift requires an image VAE that decodes to [batch, height, width, RGB].")
            prediction = prediction[..., :3]
            if self.reference is None:
                self.reference = prepare_reference(self.image1, prediction)
                self.blurred_reference = filter_comparison_image(self.reference, self.comparison_blur_pixels, self.comparison_filter, self.comparison_color_tolerance_percent)
                self.image1 = None
            registration_masks = []
            previous_mask = self.last_binary_mask
            transforms = alignment_transforms(self.reference, prediction, self.blurred_reference, self.threshold_percent, self.comparison_blur_pixels, previous_mask, registration_masks, self.comparison_filter, self.comparison_color_tolerance_percent)
            binary_mask, recalculate = output_binary_mask(self.reference, prediction, self.blurred_reference, self.threshold_percent, self.comparison_blur_pixels, transforms, self.comparison_filter, self.comparison_color_tolerance_percent)
            retained = self.dont_unmask_pixels and self.retained_mask is not None
            if retained:
                binary_mask |= self.retained_mask
            output_mask = feather_mask(binary_mask, self.blur_pixels, self.mask_expand_pixels, prediction.dtype)
            aligned, mask = apply_alignment(prediction, transforms, self.reference, output_mask)
            corrected = compose_prediction(self.reference, aligned, mask)
            if self.dont_unmask_pixels:
                self.retained_mask = binary_mask
            self.last_binary_mask = binary_mask
            encoded = self.vae.encode(corrected)
            result = model.process_latent_in(encoded).to(device=denoised.device, dtype=denoised.dtype)
            if result.shape != denoised.shape:
                raise ValueError(f"Anti-Drift VAE encoded latent shape {tuple(result.shape)} does not match the sampled latent shape {tuple(denoised.shape)}. Use the VAE matching the model and latent.")
            self.latest_preview = latent_preview.preview_to_image(corrected[0], do_scale=False)
            modes = ["detected+retained" if retained else "detected"] * prediction.shape[0]
            self.intermediate_frames.append(diagnostic_frames(prediction, corrected, mask, registration_masks[0], binary_mask, transforms, recalculate, modes, f"Correction {len(self.intermediate_frames) + 1}", sigma))
            comfy.model_management.load_models_gpu(self.models, memory_required=self.memory_required, minimum_memory_required=self.minimum_memory_required)
        return result

    def clear(self):
        self.image1 = None
        self.vae = None
        self.reference = None
        self.blurred_reference = None
        self.models = None
        self.latest_preview = None
        self.retained_mask = None
        self.last_binary_mask = None
        self.selected_intervals.clear()
        self.intermediate_frames.clear()


class EnvyKSamplerAdvancedAntiDrift(KSamplerAdvanced):
    RETURN_TYPES = ("LATENT", "IMAGE", "IMAGE")
    RETURN_NAMES = ("LATENT", "intermediate_steps_and_masks", "final_image")

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "image1": ("IMAGE", {"tooltip": "Original image being edited. The first image is used for every latent in the batch; it is center-cropped and resized with Lanczos when necessary."}),
            "vae": ("VAE", {"tooltip": "The VAE matching the sampled model and latent."}),
            "blur_pixels": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 10000.0, "step": 1.0, "tooltip": "Gaussian standard deviation in pixels for mask feathering only. Zero disables feathering."}),
            "threshold_percent": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Keep predicted pixels where the mean absolute blurred RGB difference exceeds this percentage; restore image1 elsewhere."}),
            "mask_expand_pixels": ("INT", {"default": 15, "min": 0, "max": 10000, "step": 1, "tooltip": "Expand the binary edit mask by this many pixels on each side before Gaussian feathering. Zero disables expansion."}),
            "correction_steps": ("STRING", {"default": "0.4,0.999", "tooltip": "Steps separated by spaces or commas. Zero-based whole numbers select absolute steps; fractions between 0 and 1 multiply the full configured steps and round down (0.999 × 25 → 24). Mixed values work; duplicates run once. Uses the first evaluation in each selected noise interval. Empty disables intermediate correction."}),
            "dont_unmask_pixels": ("BOOLEAN", {"default": False, "display_name": "don't unmask pixels", "tooltip": "Keep every binary edit-mask location detected at selected correction steps. Retains mask locations only; predicted pixels continue changing."}),
            "comparison_blur_pixels": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10000.0, "step": 1.0, "tooltip": "Spatial standard deviation in pixels for RGB difference detection only. Zero compares unblurred images; mask feathering uses blur_pixels."}),
        })
        inputs.setdefault("optional", {}).update({
            "comparison_filter": (["gaussian", "bilateral"], {"default": "gaussian", "tooltip": "Choose Gaussian smoothing or edge-preserving bilateral smoothing for RGB difference detection. Mask feathering remains Gaussian."}),
            "comparison_color_tolerance_percent": ("FLOAT", {"default": 10.0, "min": 0.01, "max": 100.0, "step": 0.1, "tooltip": "Bilateral color range standard deviation as a percentage of root-mean-square RGB difference, not a strict cutoff. Lower values preserve weaker edges; higher values smooth across more color differences. threshold_percent still decides which pixels count as edits."}),
        })
        return inputs

    DESCRIPTION = "Corrects selected clean denoised predictions against image1 during one uninterrupted native KSampler Advanced run."

    def sample(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, image1, vae, blur_pixels=6.0, threshold_percent=5.0, denoise=1.0, mask_expand_pixels=15, correction_steps="0.4,0.999", dont_unmask_pixels=False, comparison_blur_pixels=2.0, comparison_filter="gaussian", comparison_color_tolerance_percent=10.0):
        correction = AntiDriftCorrection(image1, vae, blur_pixels, threshold_percent, mask_expand_pixels, correction_steps, dont_unmask_pixels, start_at_step, steps, comparison_blur_pixels, comparison_filter, comparison_color_tolerance_percent)
        try:
            patched_model = model.clone()
            patched_model.set_model_sampler_post_cfg_function(correction)
            patched_model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE, "envy_anti_drift", correction.capture_sampler)
            result = super().sample(patched_model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise=denoise)
            with comfy.model_prefetch.pause_malloc_graph():
                prediction = vae.decode(result[0]["samples"])
                final_frames = []
                final_image = finalize_image(image1, prediction, correction.last_binary_mask, blur_pixels, threshold_percent, mask_expand_pixels, comparison_blur_pixels, dont_unmask_pixels, diagnostic_output=final_frames, comparison_filter=comparison_filter, comparison_color_tolerance_percent=comparison_color_tolerance_percent)
            preview = latent_preview.preview_to_image(final_image[0], do_scale=False)
            comfy.utils.ProgressBar(steps).update_absolute(steps, steps, ("JPEG", preview, None))
            intermediate_steps_and_masks = torch.cat([*correction.intermediate_frames, *final_frames], dim=0)
            return result[0], intermediate_steps_and_masks, final_image
        finally:
            correction.clear()
