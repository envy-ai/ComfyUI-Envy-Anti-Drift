import asyncio
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

import comfy.cli_args

comfy.cli_args.args.cpu = True

import nodes as comfy_nodes

from comfy.text_encoders.qwen_image21 import QwenImage21Tokenizer
from comfy.text_encoders.qwen_vl import process_qwen2vl_images

NODE_PATH = Path(__file__).resolve().parents[1] / "qwen_text_encode.py"
SPEC = importlib.util.spec_from_file_location("envy_anti_drift_qwen_text_encode", NODE_PATH)
nodes = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = nodes
SPEC.loader.exec_module(nodes)


@pytest.mark.parametrize(
    ("width", "height", "image_width", "image_height", "expected"),
    [
        (1080, 1080, None, None, (1088, 1088)),
        (1001, 777, None, None, (1008, 784)),
        (0, 0, 640, 480, (640, 480)),
        (512, 0, 640, 480, (512, 384)),
        (0, 512, 640, 480, (688, 512)),
        (512, 0, None, None, (512, 1024)),
        (0, 0, None, None, (1024, 1024)),
    ],
)
def test_resolve_size(width, height, image_width, image_height, expected):
    assert nodes._resolve_size(width, height, image_width, image_height) == expected


def test_resolve_size_uses_each_reference_aspect_ratio():
    assert nodes._resolve_size(0, 512, 640, 480) == (688, 512)
    assert nodes._resolve_size(0, 512, 1024, 512) == (1024, 512)


def test_qwen_vision_preprocessor_accepts_sixteen_pixel_reference_dimensions():
    # 48x80 is aligned to 16 but not 32; this is a valid input image size for
    # the encoder even though its vision patch merger may resize it internally.
    pixels, grid = process_qwen2vl_images(
        torch.zeros((1, 48, 80, 3)),
        patch_size=16,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    )
    assert pixels.shape[1] == 3 * 2 * 16 * 16
    assert grid.tolist() == [[1, 4, 4]]


class FakeClip:
    def __init__(self):
        self.calls = []
        self.llama_template = "clip-owned-template"

    def tokenize(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return text

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.tensor([len(tokens)]), {"clip_metadata": "kept"}]]


class FakeVAE:
    def __init__(self):
        self.images = []

    def encode(self, image):
        self.images.append(image.clone())
        return torch.full((1, 64, image.shape[1] // 16, image.shape[2] // 16), len(self.images), dtype=image.dtype)


def _rgba(width, height, red, alpha=1.0):
    image = torch.zeros((1, height, width, 4))
    image[..., 0] = red
    image[..., 1] = 0.2
    image[..., 2] = 0.3
    image[..., 3] = alpha
    return image


def test_execute_formats_both_prompts_orders_images_and_keeps_rgba_for_vae(monkeypatch):
    monkeypatch.setattr(nodes.comfy.model_management, "intermediate_device", lambda: torch.device("cpu"))
    clip = FakeClip()
    vae = FakeVAE()
    images = {
        "image_10": _rgba(64, 48, 0.1),
        "image_2": _rgba(48, 64, 0.2),
        "image_1": _rgba(80, 48, 0.4, alpha=0.25),
    }

    result = nodes.TextEncodeQwenImage21Experimental.execute(
        clip,
        "<|im_start|>already has a chat marker",
        "negative {text}",
        system_prompt="Describe {objects} [and] {shape].",
        width=0,
        height=0,
        vae=vae,
        images=images,
    )

    assert len(clip.calls) == 2
    assert clip.llama_template == "clip-owned-template"
    positive_text, positive_kwargs = clip.calls[0]
    negative_text, negative_kwargs = clip.calls[1]
    assert positive_text.startswith(
        "<|im_start|>system\nDescribe {objects} [and] {shape].<|im_end|>\n"
        "<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|> "
        "<image2><|vision_start|><|image_pad|><|vision_end|> "
        "<image3><|vision_start|><|image_pad|><|vision_end|>"
    )
    assert "<|im_start|>already has a chat marker" in positive_text
    assert "negative {text}" in negative_text
    for _, kwargs in clip.calls:
        assert kwargs["skip_template"] is True
        assert kwargs["keep_vision"] is False
        assert kwargs["prevent_empty_text"] is True
        assert [image.shape for image in kwargs["images"]] == [
            torch.Size((1, 48, 80, 3)),
            torch.Size((1, 64, 48, 3)),
            torch.Size((1, 48, 64, 3)),
        ]
        assert kwargs["llama_template"] == positive_kwargs["llama_template"]
    assert positive_kwargs["images"][0][0, 0, 0].tolist() == pytest.approx([0.85, 0.8, 0.825])
    assert len(vae.images) == 3
    assert [image.shape for image in vae.images] == [
        torch.Size((1, 48, 80, 4)),
        torch.Size((1, 64, 48, 4)),
        torch.Size((1, 48, 64, 4)),
    ]
    assert vae.images[0][0, 0, 0].tolist() == pytest.approx([0.4, 0.2, 0.3, 0.25])
    positive, negative, latent = result.args
    reference_latents = positive[0][1]["reference_latents"]
    assert len(reference_latents) == 3
    assert [value.shape for value in reference_latents] == [
        (1, 64, 3, 5),
        (1, 64, 4, 3),
        (1, 64, 3, 4),
    ]
    assert negative[0][1]["reference_latents"] is reference_latents
    assert negative[0][1]["clip_metadata"] == "kept"
    assert latent["samples"].shape == (1, 64, 3, 5)


def test_execute_without_images_uses_default_system_prompt_and_keeps_vision(monkeypatch):
    monkeypatch.setattr(nodes.comfy.model_management, "intermediate_device", lambda: torch.device("cpu"))
    clip = FakeClip()
    result = nodes.TextEncodeQwenImage21Experimental.execute(
        clip, "", "negative", system_prompt=" \n ", width=512, height=0
    )

    assert len(clip.calls) == 2
    for text, kwargs in clip.calls:
        assert text.startswith(
            "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
            "<|im_start|>user\n"
        )
        assert kwargs["images"] == []
        assert kwargs["keep_vision"] is True
    positive, negative, latent = result.args
    assert positive[0][1] == {"clip_metadata": "kept"}
    assert negative[0][1] == {"clip_metadata": "kept"}
    assert latent["samples"].shape == (1, 64, 64, 32)


def test_execute_keeps_reference_vision_when_vae_is_omitted(monkeypatch):
    monkeypatch.setattr(nodes.comfy.model_management, "intermediate_device", lambda: torch.device("cpu"))
    clip = FakeClip()
    image = torch.zeros((1, 48, 80, 3))

    result = nodes.TextEncodeQwenImage21Experimental.execute(
        clip, "describe", "", images={"image_1": image}
    )

    assert all(call[1]["keep_vision"] is True for call in clip.calls)
    assert len(clip.calls[0][1]["images"]) == 1
    positive, negative, _ = result.args
    assert "reference_latents" not in positive[0][1]
    assert "reference_latents" not in negative[0][1]


def test_qwen_image_tokenizer_parses_local_template_and_image_tokens():
    tokenizer = QwenImage21Tokenizer()
    refs = " ".join(
        f"<image{i + 1}>{nodes.VISION_BLOCK}" for i in range(2)
    )
    template = (
        "<|im_start|>system\nExplain {{objects}}.<|im_end|>\n"
        "<|im_start|>user\n" + refs + "{}<|im_end|>\n<|im_start|>assistant\n"
    )
    tokenized = tokenizer.tokenize_with_weights(
        template.format("describe them"),
        images=[torch.zeros((1, 16, 16, 3)), torch.ones((1, 16, 16, 3))],
        llama_template=template,
        skip_template=True,
        keep_vision=False,
        prevent_empty_text=True,
    )
    key = next(iter(tokenized))
    row = tokenized[key][0]
    values = [piece[0] for piece in row]
    assert values.count(151644) == 3
    image_embeds = [value for value in values if isinstance(value, dict) and value.get("type") == "image"]
    assert len(image_embeds) == 2
    assert image_embeds[0]["data"][0, 0, 0, 0].item() == 0
    assert image_embeds[1]["data"][0, 0, 0, 0].item() == 1


def test_custom_node_loader_registers_moved_sampler_and_encoder_contracts():
    names = ("EnvyKSamplerAdvancedAntiDrift", "TextEncodeQwenImage21Experimental")
    old_classes = {name: comfy_nodes.NODE_CLASS_MAPPINGS.pop(name, None) for name in names}
    old_displays = {name: comfy_nodes.NODE_DISPLAY_NAME_MAPPINGS.pop(name, None) for name in names}
    try:
        assert asyncio.run(comfy_nodes.load_custom_node(
            str(Path(__file__).resolve().parents[1]), module_parent="custom_nodes"
        ))
        sampler = comfy_nodes.NODE_CLASS_MAPPINGS[names[0]]
        encoder = comfy_nodes.NODE_CLASS_MAPPINGS[names[1]]
        assert sampler.RETURN_TYPES == ("LATENT", "IMAGE", "IMAGE", "IMAGE", "MASK")
        assert sampler.RETURN_NAMES == (
            "LATENT", "intermediate_steps_and_masks", "final_image", "raw_final_image", "final_mask"
        )
        assert comfy_nodes.NODE_DISPLAY_NAME_MAPPINGS[names[1]] == "Text Encode Qwen Image 2.1 (Experimental)"

        schema = encoder.GET_SCHEMA()
        assert schema.node_id == names[1]
        assert [item.id for item in schema.inputs] == [
            "clip", "prompt", "negative_prompt", "system_prompt", "width", "height", "vae", "images"
        ]
        images = schema.inputs[-1]
        assert images.template.min == 0
        assert images.template.names == [f"image_{index}" for index in range(1, 101)]
        assert [item.io_type for item in schema.outputs] == ["CONDITIONING", "CONDITIONING", "LATENT"]
    finally:
        for name in names:
            comfy_nodes.NODE_CLASS_MAPPINGS.pop(name, None)
            comfy_nodes.NODE_DISPLAY_NAME_MAPPINGS.pop(name, None)
            if old_classes[name] is not None:
                comfy_nodes.NODE_CLASS_MAPPINGS[name] = old_classes[name]
            if old_displays[name] is not None:
                comfy_nodes.NODE_DISPLAY_NAME_MAPPINGS[name] = old_displays[name]
