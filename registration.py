# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Bart

import math
import statistics

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


def _translation(reference, prediction, unmasked):
    height, width = reference.shape
    patch_height = min(256, height // 3)
    patch_width = min(256, width // 3)
    radius = min(32, patch_height // 4, patch_width // 4)
    if radius < 1:
        return 0, 0
    votes = {}
    eligible_tiles = 0
    for row in range(3):
        top = (height - patch_height) * row // 2
        for column in range(3):
            left = (width - patch_width) * column // 2
            region = (slice(top, top + patch_height), slice(left, left + patch_width))
            tile_mask = unmasked[region].float()
            if tile_mask.sum() < max(64, patch_height * patch_width * 0.35):
                continue
            scores = _correlation(reference[region], prediction[region], tile_mask, radius)
            center = radius
            zero_score = scores[center, center]
            best = torch.argmax(scores).item()
            y, x = divmod(best, scores.shape[1])
            peak = scores[y, x]
            if peak < 0.5 or y in (0, 2 * radius) or x in (0, 2 * radius):
                continue
            eligible_tiles += 1
            other = scores.clone()
            other[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = -2
            next_peak = other.max()
            if peak - next_peak <= max(1e-4, 0.2 * (1 - next_peak)):
                continue
            fractions = []
            peak_value = peak.item()
            refined_peak = peak_value
            for before, after in ((scores[y - 1, x], scores[y + 1, x]), (scores[y, x - 1], scores[y, x + 1])):
                before, after = before.item(), after.item()
                curvature = before - 2 * peak_value + after
                fraction = (before - after) / (2 * curvature) if before > -1 and after > -1 and curvature < -1e-6 else 0
                if abs(fraction) > 0.5:
                    fraction = 0
                gain = (after - before) * fraction / 4
                if gain <= max(1e-6, 0.05 * (1 - peak_value)):
                    fraction = 0
                    gain = 0
                refined_peak += gain
                fractions.append(fraction)
            if refined_peak - zero_score <= max(1e-6, 0.2 * (1 - zero_score)):
                continue
            offset = (y - center, x - center)
            votes.setdefault(offset, []).append((offset[0] + fractions[0], offset[1] + fractions[1]))
    if not votes:
        return 0, 0
    _, refinements = max(votes.items(), key=lambda item: len(item[1]))
    if len(refinements) < 3 or len(refinements) * 2 <= eligible_tiles:
        return 0, 0
    dy = statistics.median(item[0] for item in refinements)
    dx = statistics.median(item[1] for item in refinements)
    return (dy, dx) if math.hypot(dy, dx) > 0.05 else (0, 0)


def estimate_translation(reference, prediction, exclusion_mask=None):
    batch, height, width, _ = prediction.shape
    source_luma = reference[0].float().mean(dim=-1)
    offsets = []
    for index in range(batch):
        available = torch.ones((height, width), dtype=torch.bool, device=prediction.device) if exclusion_mask is None else ~exclusion_mask[index, ..., 0]
        offsets.append(_translation(source_luma, prediction[index].float().mean(dim=-1), available))
    return offsets


def warp_image(image, offsets, mode="bilinear"):
    batch, height, width, _ = image.shape
    warped = torch.zeros_like(image)
    grid_dtype = torch.float64 if image.dtype == torch.float64 else torch.float32
    coverage = torch.zeros((batch, height, width, 1), dtype=grid_dtype, device=image.device)
    for index, (dy, dx) in enumerate(offsets):
        if float(dy).is_integer() and float(dx).is_integer():
            dy, dx = int(dy), int(dx)
            source_y = slice(max(0, -dy), min(height, height - dy))
            source_x = slice(max(0, -dx), min(width, width - dx))
            target_y = slice(max(0, dy), min(height, height + dy))
            target_x = slice(max(0, dx), min(width, width + dx))
            warped[index, source_y, source_x] = image[index, target_y, target_x]
            coverage[index, source_y, source_x] = 1
            continue
        y = torch.arange(height, device=image.device, dtype=grid_dtype) + dy
        x = torch.arange(width, device=image.device, dtype=grid_dtype) + dx
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack(((grid_x + 0.5) * (2 / width) - 1, (grid_y + 0.5) * (2 / height) - 1), dim=-1)[None]
        pixels = image[index:index + 1].movedim(-1, 1).to(grid_dtype)
        warped[index] = F.grid_sample(pixels, grid, mode=mode, align_corners=False).movedim(1, -1)[0].to(image.dtype)
        coverage[index] = F.grid_sample(torch.ones((1, 1, height, width), dtype=grid_dtype, device=image.device), grid, mode=mode, align_corners=False).movedim(1, -1)[0]
    return warped, coverage


def shift_mask(mask, offsets, mode="nearest"):
    warped, _ = warp_image(mask.float() if mask.dtype == torch.bool else mask, offsets, mode)
    return warped > 0.5 if mask.dtype == torch.bool else warped


def apply_translation(image, offsets, reference, edit_mask):
    batch, height, width, _ = image.shape
    shifted, coverage = warp_image(image, offsets)
    translated = shifted.clone()
    displayed_mask = edit_mask.clone()
    for index, (dy, dx) in enumerate(offsets):
        if dy == 0 and dx == 0:
            continue
        background = (coverage[index, ..., 0] > 0.999) & (edit_mask[index, ..., 0] == 0)
        fill = match_background_color(reference, shifted[index:index + 1], (~background)[None, ..., None])[0]
        weight = torch.ones((height, width), dtype=torch.float32, device=image.device)
        if dy > 0:
            y = torch.arange(height, device=image.device)
            weight *= ((height - dy - y) / 3).clamp(0, 1)[:, None]
        elif dy < 0:
            y = torch.arange(height, device=image.device)
            weight *= ((y + dy + 1) / 3).clamp(0, 1)[:, None]
        if dx > 0:
            x = torch.arange(width, device=image.device)
            weight *= ((width - dx - x) / 3).clamp(0, 1)[None, :]
        elif dx < 0:
            x = torch.arange(width, device=image.device)
            weight *= ((x + dx + 1) / 3).clamp(0, 1)[None, :]
        translated[index] = fill * (1 - coverage[index] * weight[..., None]) + shifted[index] * weight[..., None]
        displayed_mask[index] *= weight[..., None]
    return translated, displayed_mask
