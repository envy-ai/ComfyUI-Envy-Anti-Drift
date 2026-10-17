# ComfyUI Envy Anti-Drift

`Envy KSampler Advanced Anti-Drift` restores small blurred RGB differences to the original image while retaining larger changes. It adds sparse correction steps, mask expansion, optional mask retention, and a final image composite to the native KSampler Advanced controls.

Clone [ComfyUI-Envy-Anti-Drift](https://github.com/envy-ai/ComfyUI-Envy-Anti-Drift) into `ComfyUI/custom_nodes` and restart ComfyUI. No additional dependencies are required.

Connect the original image being edited to `image1` and the matching VAE to `vae`. If `image1` contains a batch, its first image is the reference for every sampled latent. When pixel dimensions differ, the reference is aspect-preserving center-cropped and resized with Lanczos to the decoded prediction; matching dimensions avoid resampling. Alpha is ignored.

At each selected correction, the node decodes the clean prediction after CFG. When `align_background` is enabled, it first registers a confident whole-image pixel shift against the reference. It compares the mean absolute RGB difference between copies of the aligned prediction and reference blurred by `comparison_blur_pixels`. Pixels above `threshold_percent` become a binary edit mask. The binary mask expands by `mask_expand_pixels` before `blur_pixels` feathers it. The same background-based percentile gain/offset used by the final pass color-corrects the prediction before blending the **unblurred prediction inside** with the **unblurred reference outside**. The composite is VAE-encoded and returned to the sampler as the corrected clean prediction. Previews and intermediate frames include this color correction; it adds no VAE calls.

`comparison_blur_pixels` controls RGB difference detection only (default **2** pixels); **0** compares unblurred images. `blur_pixels` controls mask feathering only (default **6** pixels); **0** disables feathering. Both are Gaussian standard deviations, truncated at three standard deviations with replicated edges. `threshold_percent` defaults to **5%** of the normalized RGB range. The aligned source and its comparison-blurred copy are cached between intermediate corrections.

`mask_expand_pixels` defaults to **15** pixels in every direction using a square footprint, limited to the existing image bounds. **0** disables expansion. Expansion occurs before feathering.

`align_background` is optional and defaults to **true**. It compares unmasked background patches in the original and decoded prediction, searches whole-pixel shifts smaller than 32 pixels per axis (less for small images), and applies a shift only when several textured patches agree clearly. Pixels above the current difference threshold and the expanded, feathered footprint of any saved edit mask are excluded from both sides of the comparison. If too little reliable background remains, the prediction is left in place. Exposed edge pixels are filled from the original image after matching its background colors to the prediction, with a two-pixel feather into valid prediction pixels. Synthetic pixels are excluded from subsequent difference detection and color statistics. This registration also runs independently on the final decode; edit masks remain anchored to the original image. Disable the option to keep the previous behavior.

`correction_steps` defaults to **`0.4,0.999`**. Separate values with spaces or commas; mixed inputs such as `4, 0.8` work. Whole numbers are zero-based original KSampler Advanced step positions. Fractions strictly between 0 and 1 multiply the **full configured `steps`** and round down: with 25 steps, `0.999` selects step 24, and `0.999, 24` corrects only once. Duplicates are removed after conversion. Positions outside the actual sampled range are ignored, including positions skipped by `start_at_step`. An empty string disables intermediate corrections; invalid values raise an error.

Each selected position runs once, at the first evaluation in its sigma interval. Predictor/corrector samplers such as Heun can enter that interval during the preceding outer step; repeated evaluations in the same interval do not trigger another correction. Other evaluations pass through without VAE work, history frames, or mask updates. Sampling remains one native call, preserving solver history and all native noise, start/end, and leftover-noise controls.

**don't unmask pixels** defaults to **false**. When enabled, each batch item's binary edit mask accumulates all locations detected at selected corrections. The retained mask is unexpanded and unfeathered; expansion and blur are applied afresh, avoiding repeated outward growth. Only mask locations persist, while predicted pixels keep evolving. Early false positives remain selected. With retention disabled, the final pass uses the last detected binary mask.

During correction, live previews reuse the corrected composite at full decoded resolution, using the first image of the sampled batch. Other evaluations use normal configured latent previews. The final composite also receives a full-resolution preview. Previews have no magenta overlay; masks remain grayscale in the second output.

The first output is the native sampled `LATENT`. `intermediate_steps_and_masks` is an `IMAGE` batch recording each executed intermediate correction. Each frame places the corrected RGB image on the left and its actual feathered mask on the right (white keeps prediction, black restores source). Frames are ordered by evaluation, then by sampled batch position. This output reuses existing decoded pixels and stores CPU float32 copies; it uses about 24 bytes per decoded pixel per correction and batch item, with another copy needed when assembling the final batch. A run with no intermediate corrections returns an empty image batch shaped `(0, 0, 0, 3)`.

The third output, `final_image`, decodes the returned latent once and composites it with the original image in CPU float32. It reuses the last binary mask; if no intermediate correction ran, it detects a mask from this final decode. Expansion and feathering are applied afresh. Before restoring the source background, it compares the generated and source pixels where the feathered mask is exactly zero and the prediction contains reliable pixels. Per-image, per-channel 10th/50th/90th percentiles define a color gain and offset, which are applied to the generated image. Empty background samples or zero percentile spread in either channel leave that channel unchanged.

The final composite copies the aligned original pixels exactly wherever the feathered mask is zero. Feathered edges mix source and generated pixels. This guarantee applies to `final_image`; VAE reconstruction can still alter the `LATENT` outside the mask. The final image is never re-encoded, and the first output remains the native sampler result. When the source needs resizing, exact preservation refers to its Lanczos-aligned pixels. Color matching may produce values outside 0–1.

Each intermediate correction adds one VAE decode and encode, and finalization adds one decode. Model transfers between RAM and VRAM can dominate the cost. The final image is composited even when intermediate corrections are disabled or no selected position was sampled.

The threshold measures change size, not edit intent: it can preserve large unwanted changes or erase subtle intended edits. Early predictions can differ across most of the image and therefore receive little correction.

Requires ComfyUI 0.40.0 or newer and a compatible image VAE with RGB image decoding. Tested with ComfyUI 0.40.0 using its native model-patcher, sampler-wrapper, and memory-management APIs; it makes no changes to ComfyUI core.

Licensed under GPL-3.0-or-later. This node builds on ComfyUI's native KSampler Advanced; see [LICENSE](LICENSE).

## Tests

From the pack directory, run the CPU tests with ComfyUI available on `PYTHONPATH`:

```bash
PYTHONPATH=/path/to/ComfyUI python -m pytest --import-mode=importlib -q tests/test_anti_drift_sampler.py
```
