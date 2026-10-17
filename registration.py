# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Bart

import math

import numpy as np
from scipy import ndimage, optimize

import torch
import torch.nn.functional as F


def match_background_color(prediction, reference, mask):
    matched = prediction.clone()
    for index in range(prediction.shape[0]):
        unmasked = mask[index, ..., 0] == 0
        if not unmasked.any():
            continue
        generated_pixels = prediction[index][unmasked]
        source_pixels = reference[0][unmasked]
        if generated_pixels.dtype in (torch.float16, torch.bfloat16):
            generated_pixels = generated_pixels.float()
            source_pixels = source_pixels.float()
        quantiles = generated_pixels.new_tensor([0.1, 0.5, 0.9])
        generated = torch.quantile(generated_pixels, quantiles, dim=0)
        source = torch.quantile(source_pixels, quantiles, dim=0)
        generated_spread = generated[2] - generated[0]
        source_spread = source[2] - source[0]
        valid = (generated_spread != 0) & (source_spread != 0)
        gain = torch.ones_like(generated_spread)
        offset = torch.zeros_like(generated_spread)
        gain[valid] = source_spread[valid] / generated_spread[valid]
        offset[valid] = source[1, valid] - gain[valid] * generated[1, valid]
        matched[index].mul_(gain).add_(offset)
    return matched


def _correlation(reference, prediction, unmasked, radius):
    height, width = reference.shape
    size = (height * 2, width * 2)
    source = unmasked * reference
    target = unmasked * prediction
    source_fft = torch.fft.rfft2(source, s=size)
    mask_fft = torch.fft.rfft2(unmasked, s=size)
    target_fft = torch.fft.rfft2(target, s=size)
    valid_fft = mask_fft

    def correlate(left, right):
        return torch.fft.irfft2(left.conj() * right, s=size)

    offsets = torch.arange(-radius, radius + 1, device=reference.device)
    rows = offsets.remainder(size[0])[:, None]
    columns = offsets.remainder(size[1])[None, :]

    def window(values):
        return values[rows, columns]

    count = window(correlate(mask_fft, valid_fft)).clamp_min(0)
    source_sum = window(correlate(source_fft, valid_fft))
    source_square = window(correlate(torch.fft.rfft2(source * reference, s=size), valid_fft))
    target_sum = window(correlate(mask_fft, target_fft))
    target_square = window(correlate(mask_fft, torch.fft.rfft2(target * prediction, s=size)))
    cross = window(correlate(source_fft, target_fft))
    count_safe = count.clamp_min(1)
    numerator = cross - source_sum * target_sum / count_safe
    source_variance = (source_square - source_sum.square() / count_safe).clamp_min(0)
    target_variance = (target_square - target_sum.square() / count_safe).clamp_min(0)
    score = numerator / (source_variance * target_variance).sqrt().clamp_min(1e-8)
    eligible = unmasked.sum()
    score = score.masked_fill((count < eligible * 0.6) | (count < 64) | (source_variance < count * 1e-5) | (target_variance < count * 1e-5), -2)
    return score


def _refine_patch(reference, prediction, unmasked, dy, dx):
    source = reference.cpu().numpy()
    target = prediction.cpu().numpy()
    available = unmasked.cpu().numpy().astype(bool)
    available_float = available.astype(np.float32)
    yy, xx = np.indices(source.shape, dtype=np.float64)

    def valid_pixels(offset):
        sample_y = yy + offset[0]
        sample_x = xx + offset[1]
        valid = available & (sample_y >= 0) & (sample_y <= source.shape[0] - 1) & (sample_x >= 0) & (sample_x <= source.shape[1] - 1)
        valid &= ndimage.map_coordinates(available_float, (sample_y, sample_x), order=1, mode="constant", cval=0) > 0.999
        return valid, sample_y, sample_x

    def score(offset):
        valid, sample_y, sample_x = valid_pixels(offset)
        if valid.sum() < 64:
            return 1.0
        sampled = ndimage.map_coordinates(target, (sample_y, sample_x), order=1, mode="constant", cval=0)
        a = source[valid] - source[valid].mean()
        b = sampled[valid] - sampled[valid].mean()
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        return -float(np.dot(a, b) / denominator) if denominator > 1e-8 else 1.0

    result = optimize.minimize(score, (dy, dx), method="Powell",
                               bounds=((dy - 0.75, dy + 0.75), (dx - 0.75, dx + 0.75)),
                               options={"maxfev": 80, "xtol": 1e-3, "ftol": 1e-4})
    valid, _, _ = valid_pixels(result.x)
    if valid.sum() < 64:
        return None
    center_y, center_x = np.where(valid)
    return result.x[0], result.x[1], center_y.mean(), center_x.mean()


def _patch_alignment(reference, prediction, unmasked):
    height, width = reference.shape
    patch_height = min(192, height // 4)
    patch_width = min(192, width // 4)
    radius = min(32, patch_height // 4, patch_width // 4)
    if radius < 1:
        return (1.0, 1.0, 0.0, 0.0)
    patches = []
    for row in range(5):
        top = (height - patch_height) * row // 4
        for column in range(5):
            left = (width - patch_width) * column // 4
            region = (slice(top, top + patch_height), slice(left, left + patch_width))
            tile_mask = unmasked[region].float()
            if tile_mask.sum() < max(64, patch_height * patch_width * 0.3):
                continue
            scores = _correlation(reference[region], prediction[region], tile_mask, radius)
            best = torch.argmax(scores).item()
            y, x = divmod(best, scores.shape[1])
            peak = scores[y, x].item()
            if peak < 0.4 or y in (0, 2 * radius) or x in (0, 2 * radius):
                continue
            other = scores.clone()
            other[max(0, y - 2):y + 3, max(0, x - 2):x + 3] = -2
            if peak - other.max().item() < 0.002:
                continue
            refined = _refine_patch(reference[region], prediction[region], tile_mask, y - radius, x - radius)
            if refined is None:
                continue
            dy, dx, center_y, center_x = refined
            patches.append((top + center_y, left + center_x, dy, dx))
    if len(patches) < 4:
        return (1.0, 1.0, 0.0, 0.0)
    patches = np.asarray(patches)
    keep = np.ones(len(patches), dtype=bool)
    for _ in range(3):
        if keep.sum() < 4 or np.ptp(patches[keep, 0]) < height * 0.35 or np.ptp(patches[keep, 1]) < width * 0.35:
            return (1.0, 1.0, 0.0, 0.0)
        y_fit = optimize.least_squares(lambda line: np.polyval(line, patches[keep, 0]) - patches[keep, 2],
                                       np.polyfit(patches[keep, 0], patches[keep, 2], 1), loss="soft_l1", f_scale=1).x
        x_fit = optimize.least_squares(lambda line: np.polyval(line, patches[keep, 1]) - patches[keep, 3],
                                       np.polyfit(patches[keep, 1], patches[keep, 3], 1), loss="soft_l1", f_scale=1).x
        residual = np.hypot(patches[:, 2] - np.polyval(y_fit, patches[:, 0]), patches[:, 3] - np.polyval(x_fit, patches[:, 1]))
        next_keep = residual < max(1.0, np.median(residual[keep]) * 2.5)
        if np.array_equal(keep, next_keep):
            break
        keep = next_keep
    if keep.sum() < 4 or np.ptp(patches[keep, 0]) < height * 0.35 or np.ptp(patches[keep, 1]) < width * 0.35:
        return (1.0, 1.0, 0.0, 0.0)
    y_fit = np.polyfit(patches[keep, 0], patches[keep, 2], 1)
    x_fit = np.polyfit(patches[keep, 1], patches[keep, 3], 1)
    residual = np.hypot(patches[:, 2] - np.polyval(y_fit, patches[:, 0]), patches[:, 3] - np.polyval(x_fit, patches[:, 1]))
    sy, dy = 1 + y_fit[0], y_fit[1]
    sx, dx = 1 + x_fit[0], x_fit[1]
    if not 0.9 < sy < 1.1 or not 0.9 < sx < 1.1 or np.median(residual[keep]) > 1.5:
        return (1.0, 1.0, 0.0, 0.0)
    return (float(sy), float(sx), float(dy), float(dx))


def _ecc_alignment(reference, prediction, unmasked):
    height, width = reference.shape
    identity = (1.0, 1.0, 0.0, 0.0)
    if unmasked.sum() < max(256, height * width * 0.1):
        return identity
    patch_height, patch_width = min(192, height // 4), min(192, width // 4)
    supported = []
    for row in range(5):
        top = (height - patch_height) * row // 4
        for column in range(5):
            left = (width - patch_width) * column // 4
            if unmasked[top:top + patch_height, left:left + patch_width].float().mean() >= 0.3:
                supported.append((top + patch_height / 2, left + patch_width / 2))
    if len(supported) < 4 or np.ptp([point[0] for point in supported]) < height * 0.35 or np.ptp([point[1] for point in supported]) < width * 0.35:
        return identity
    scales = [min(1.0, 192 / max(height, width)), min(1.0, 384 / max(height, width))]
    transform = np.array(identity, dtype=np.float64)
    for scale in dict.fromkeys(scales):
        size = (round(height * scale), round(width * scale))
        resize_y, resize_x = size[0] / height, size[1] / width
        source = F.interpolate(reference[None, None], size=size, mode="area")[0, 0].cpu().numpy()
        target = F.interpolate(prediction[None, None], size=size, mode="area")[0, 0].cpu().numpy()
        available = F.interpolate(unmasked[None, None].float(), size=size, mode="area")[0, 0].cpu().numpy() > 0.999
        available_float = available.astype(np.float32)
        yy, xx = np.indices(size, dtype=np.float64)
        if available.sum() < max(256, size[0] * size[1] * 0.1):
            return identity
        if np.var(source[available]) < 1e-5 or np.var(target[available]) < 1e-5:
            return identity

        def sample_support(parameters):
            sy, sx, dy, dx = parameters
            sample_y = sy * yy + dy
            sample_x = sx * xx + dx
            valid = available & (sample_y >= 0) & (sample_y <= size[0] - 1) & (sample_x >= 0) & (sample_x <= size[1] - 1)
            valid &= ndimage.map_coordinates(available_float, (sample_y, sample_x), order=1, mode="constant", cval=0) > 0.999
            return valid, sample_y, sample_x

        def correlation(parameters, shared_support=None):
            valid, sample_y, sample_x = sample_support(parameters)
            if shared_support is not None:
                valid &= shared_support
            if valid.sum() < max(256, available.sum() * 0.4):
                return -1.0
            sampled = ndimage.map_coordinates(target, (sample_y, sample_x), order=1, mode="constant", cval=0)
            a = source[valid]
            b = sampled[valid]
            a = a - a.mean()
            b = b - b.mean()
            denominator = np.linalg.norm(a) * np.linalg.norm(b)
            return float(np.dot(a, b) / denominator) if denominator > 1e-8 else -1.0

        start = transform.copy()
        start[2] = resize_y * transform[2] + (transform[0] - 1) * (1 - resize_y) / 2
        start[3] = resize_x * transform[3] + (transform[1] - 1) * (1 - resize_x) / 2
        bounds = ((0.9, 1.1), (0.9, 1.1), (-min(32 * resize_y, size[0] / 4), min(32 * resize_y, size[0] / 4)),
                  (-min(32 * resize_x, size[1] / 4), min(32 * resize_x, size[1] / 4)))
        result = optimize.minimize(lambda parameters: -correlation(parameters), start, method="Powell", bounds=bounds,
                                   options={"maxfev": 350, "xtol": 1e-4, "ftol": 1e-5})
        transform = result.x
        common = sample_support(identity)[0] & sample_support(transform)[0]
        base_score = correlation(identity, common)
        score = correlation(transform, common)
        transform[2] = (transform[2] - (transform[0] - 1) * (1 - resize_y) / 2) / resize_y
        transform[3] = (transform[3] - (transform[1] - 1) * (1 - resize_x) / 2) / resize_x
    if score < 0.45 or score - base_score < max(1e-7, 0.03 * (1 - base_score)):
        return identity
    return tuple(float(value) for value in transform)


def estimate_alignment(reference, prediction, exclusion_mask=None, method="patch"):
    if method not in ("patch", "ecc"):
        raise ValueError(f"Unknown background alignment method: {method}")
    batch, height, width, _ = prediction.shape
    source_luma = reference[0].float().mean(dim=-1)
    transforms = []
    for index in range(batch):
        available = torch.ones((height, width), dtype=torch.bool, device=prediction.device) if exclusion_mask is None else ~exclusion_mask[index, ..., 0]
        target_luma = prediction[index].float().mean(dim=-1)
        estimator = _patch_alignment if method == "patch" else _ecc_alignment
        transforms.append(estimator(source_luma, target_luma, available))
    return transforms


def invert_alignment(transforms):
    return [(1 / sy, 1 / sx, -dy / sy, -dx / sx) for sy, sx, dy, dx in transforms]


def alignment_displacement(transforms, height, width):
    return [max(math.hypot((sy - 1) * y + dy, (sx - 1) * x + dx)
                for y in (0, height - 1) for x in (0, width - 1))
            for sy, sx, dy, dx in transforms]


def warp_image(image, transforms, mode="bilinear"):
    batch, height, width, _ = image.shape
    warped = torch.zeros_like(image)
    grid_dtype = torch.float64 if image.dtype == torch.float64 else torch.float32
    coverage = torch.zeros((batch, height, width, 1), dtype=grid_dtype, device=image.device)
    for index, (sy, sx, dy, dx) in enumerate(transforms):
        if sy == 1 and sx == 1 and float(dy).is_integer() and float(dx).is_integer():
            dy, dx = int(dy), int(dx)
            source_y = slice(max(0, -dy), min(height, height - dy))
            source_x = slice(max(0, -dx), min(width, width - dx))
            target_y = slice(max(0, dy), min(height, height + dy))
            target_x = slice(max(0, dx), min(width, width + dx))
            warped[index, source_y, source_x] = image[index, target_y, target_x]
            coverage[index, source_y, source_x] = 1
            continue
        y = torch.arange(height, device=image.device, dtype=grid_dtype) * sy + dy
        x = torch.arange(width, device=image.device, dtype=grid_dtype) * sx + dx
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack(((grid_x + 0.5) * (2 / width) - 1, (grid_y + 0.5) * (2 / height) - 1), dim=-1)[None]
        pixels = image[index:index + 1].movedim(-1, 1).to(grid_dtype)
        warped[index] = F.grid_sample(pixels, grid, mode=mode, align_corners=False).movedim(1, -1)[0].to(image.dtype)
        coverage[index] = F.grid_sample(torch.ones((1, 1, height, width), dtype=grid_dtype, device=image.device), grid, mode=mode, align_corners=False).movedim(1, -1)[0]
    return warped, coverage


def shift_mask(mask, transforms, mode="nearest"):
    warped, _ = warp_image(mask.float() if mask.dtype == torch.bool else mask, transforms, mode)
    return warped > 0.5 if mask.dtype == torch.bool else warped


def apply_alignment(image, transforms, reference, edit_mask):
    shifted, coverage = warp_image(image, transforms)
    translated = shifted.clone()
    displayed_mask = edit_mask.clone()
    for index, transform in enumerate(transforms):
        if transform == (1.0, 1.0, 0.0, 0.0):
            continue
        background = (coverage[index, ..., 0] > 0.999) & (edit_mask[index, ..., 0] == 0)
        fill = match_background_color(reference, shifted[index:index + 1], (~background)[None, ..., None])[0]
        invalid = (coverage[index, ..., 0] <= 0.999).float()[None, None]
        one_pixel = F.max_pool2d(invalid, 3, stride=1, padding=1)
        two_pixels = F.max_pool2d(one_pixel, 3, stride=1, padding=1)
        ramp = 1 - (one_pixel + two_pixels)[0, 0] / 3
        translated[index] = fill * (1 - coverage[index] * ramp[..., None]) + shifted[index] * ramp[..., None]
        displayed_mask[index] *= ramp[..., None]
    return translated, displayed_mask
