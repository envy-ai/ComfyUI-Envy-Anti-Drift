# Derived from ComfyUI's comfy_extras/nodes_qwen.py (GPL-3.0).
import torch

import comfy.model_management
import comfy.utils
import node_helpers
from comfy.text_encoders.qwen_image21 import VISION_BLOCK
from comfy_api.latest import io


def _resolve_size(width, height, image_width=None, image_height=None):
    if width < 0 or height < 0:
        raise ValueError("Width and height must be nonnegative.")
    if image_width is None:
        width, height = width or 1024, height or 1024
    elif width == 0 and height == 0:
        width, height = image_width, image_height
    elif width == 0:
        width = height * image_width / image_height
    elif height == 0:
        height = width * image_height / image_width
    return max(16, int((width + 8) // 16) * 16), max(16, int((height + 8) // 16) * 16)


class TextEncodeQwenImage21Experimental(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeQwenImage21Experimental",
            display_name="Text Encode Qwen Image 2.1 (Experimental)",
            category="model/conditioning/qwen image",
            is_experimental=True,
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("negative_prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, default="",
                                tooltip="Applied to both prompts. Empty or whitespace-only uses: Comprehend and analyze the provided prompt."),
                io.Int.Input("width", default=0, min=0, step=16,
                             tooltip="Reference width, rounded to the nearest multiple of 16. 0 uses each image's width, or infers it from height. Without images, 0 uses 1024."),
                io.Int.Input("height", default=0, min=0, step=16,
                             tooltip="Reference height, rounded to the nearest multiple of 16. 0 uses each image's height, or infers it from width. Without images, 0 uses 1024."),
                io.Vae.Input("vae", optional=True),
                io.Autogrow.Input(
                    "images",
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input("image"),
                        names=[f"image_{i}" for i in range(1, 101)],
                        min=0,
                    ),
                    tooltip="Reference images in socket order. Only the first image in each batch is used.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent",
                                 tooltip="Empty latent matching the first reference image's resized dimensions."),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, negative_prompt, system_prompt="", width=0, height=0, vae=None, images: io.Autogrow.Type = None) -> io.NodeOutput:
        ref_latents = []
        images_vl = []
        latent_w, latent_h = _resolve_size(width, height)
        images = images or {}
        for name in sorted(images, key=lambda n: int(n.rsplit("_", 1)[-1])):
            image = images[name]
            if image is None:
                continue
            samples = image[:1].movedim(-1, 1)
            image_w, image_h = _resolve_size(width, height, samples.shape[3], samples.shape[2])
            if (image_w, image_h) == (samples.shape[3], samples.shape[2]):
                s = image[:1]
            else:
                s = comfy.utils.common_upscale(samples, image_w, image_h, "lanczos", "disabled").movedim(1, -1)
            if not images_vl:
                latent_w, latent_h = image_w, image_h
            rgb = s[:, :, :, :3]
            if s.shape[-1] > 3:
                rgb = rgb * s[:, :, :, 3:] + (1.0 - s[:, :, :, 3:])
            images_vl.append(rgb)
            if vae is not None:
                ref_latents.append(vae.encode(s))

        system = system_prompt if system_prompt.strip() else "Comprehend and analyze the provided prompt."
        system = system.replace("{", "{{").replace("}", "}}")
        refs = " ".join(f"<image{i + 1}>{VISION_BLOCK}" for i in range(len(images_vl)))
        llama_template = "<|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n" + refs + "{}<|im_end|>\n<|im_start|>assistant\n"
        keep_vision = len(ref_latents) == 0
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(llama_template.format(prompt or " "), images=images_vl, llama_template=llama_template, skip_template=True, keep_vision=keep_vision, prevent_empty_text=True))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(llama_template.format(negative_prompt or " "), images=images_vl, llama_template=llama_template, skip_template=True, keep_vision=keep_vision, prevent_empty_text=True))
        if len(ref_latents) > 0:
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": ref_latents}, append=True)
        latent = torch.zeros([1, 64, latent_h // 16, latent_w // 16], device=comfy.model_management.intermediate_device())
        return io.NodeOutput(positive, negative, {"samples": latent})
