# ComfyUI Qwen 2.1 Experimental

**Text Encode Qwen Image 2.1 (Experimental)** (`TextEncodeQwenImage21Experimental`) ships with [ComfyUI Envy Anti-Drift](../README.md) alongside the built-in node. It requires a ComfyUI version with native Qwen Image 2.1 support and no additional dependencies.

Connect a Qwen Image 2.1 CLIP, positive and negative prompts, and optional reference images and VAE. Outputs are positive conditioning, negative conditioning, and an empty 64-channel latent matching the first resized reference. Without references, the latent uses the requested dimensions.

The multiline `system_prompt` applies to both positive and negative encoding for this call. Its empty default, including whitespace-only input, uses exactly `Comprehend and analyze the provided prompt.` Literal braces are supported. It does not alter shared tokenizer settings.

| Width / height | With reference images | Without reference images |
| --- | --- | --- |
| Both 0 | Each reference uses its original dimensions. | 1024 × 1024. |
| One 0 | Infer that dimension from each reference's original aspect ratio. | The missing dimension is 1024. |
| Both positive | Resize every reference to those dimensions, stretching without cropping. | Use those dimensions. |

Dimensions are rounded to the nearest multiple of 16, with halfway values rounded up and a minimum actual size of 16. Aspect ratio inference happens before rounding. There is no added upper resolution limit. The text encoder's vision preprocessing still performs its own resizing to its patch grid; VAE references and the output latent use the node's resolved dimensions.

References are processed in numeric socket order, using only the first image of each batch. The current ComfyUI Autogrow API supports up to 100 sockets; this node exposes that framework limit. RGBA images are composited over white for the vision tower, while the VAE receives the resized original channels. With a VAE, references are attached to both conditionings and native image-slot handling is preserved. Without a VAE, vision tokens remain in conditioning.

Derived from ComfyUI's `TextEncodeQwenImage21` in `comfy_extras/nodes_qwen.py`, copyright its ComfyUI contributors. Distributed under GPL-3.0; see [LICENSE](../LICENSE).
