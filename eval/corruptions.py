from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np


CORRUPTION_TYPES = ("clean", "stripe", "ghost", "dropout", "combined")
CORRUPTION_SEVERITIES = ("mild", "medium", "severe")


@dataclass(frozen=True, slots=True)
class CorruptionConfig:
    stripe_count: int
    stripe_thickness: int
    stripe_opacity: float
    stripe_target_intensity: float
    stripe_top_band_px: int
    ghost_shift_px: int
    ghost_opacity: float
    dropout_regions: int
    dropout_width_px: int
    dropout_height_px: int
    dropout_contrast_scale: float
    dropout_intensity_scale: float


_SEVERITY_CONFIGS: dict[str, CorruptionConfig] = {
    "mild": CorruptionConfig(
        stripe_count=1,
        stripe_thickness=3,
        stripe_opacity=0.45,
        stripe_target_intensity=220.0,
        stripe_top_band_px=180,
        ghost_shift_px=12,
        ghost_opacity=0.10,
        dropout_regions=1,
        dropout_width_px=72,
        dropout_height_px=112,
        dropout_contrast_scale=0.62,
        dropout_intensity_scale=0.74,
    ),
    "medium": CorruptionConfig(
        stripe_count=2,
        stripe_thickness=6,
        stripe_opacity=0.78,
        stripe_target_intensity=236.0,
        stripe_top_band_px=200,
        ghost_shift_px=24,
        ghost_opacity=0.25,
        dropout_regions=2,
        dropout_width_px=112,
        dropout_height_px=160,
        dropout_contrast_scale=0.46,
        dropout_intensity_scale=0.58,
    ),
    "severe": CorruptionConfig(
        stripe_count=3,
        stripe_thickness=9,
        stripe_opacity=0.92,
        stripe_target_intensity=248.0,
        stripe_top_band_px=200,
        ghost_shift_px=40,
        ghost_opacity=0.45,
        dropout_regions=3,
        dropout_width_px=160,
        dropout_height_px=220,
        dropout_contrast_scale=0.30,
        dropout_intensity_scale=0.42,
    ),
}


def corruption_output_suffix(corruption: str, severity: str) -> str:
    corruption_key = str(corruption).strip().lower()
    if corruption_key == "clean":
        return ""
    return f"_{corruption_key}_{str(severity).strip().lower()}"


def _rng_for_sample(*, sample_key: str, corruption: str, severity: str, seed: int) -> np.random.Generator:
    token = f"{seed}:{corruption}:{severity}:{sample_key}".encode("utf-8", errors="ignore")
    seed_u32 = zlib.crc32(token) & 0xFFFFFFFF
    return np.random.default_rng(seed_u32)


def _stripe_anchor_rows(height: int, cfg: CorruptionConfig) -> np.ndarray:
    top_band = min(max(int(cfg.stripe_top_band_px), 16), max(height - 1, 1))
    anchors = np.arange(3, top_band, 13, dtype=np.int32)
    if anchors.size == 0:
        anchors = np.array([max(top_band // 2, 1)], dtype=np.int32)
    return anchors


def _sample_stripe_centers(height: int, cfg: CorruptionConfig, rng: np.random.Generator) -> np.ndarray:
    anchors = _stripe_anchor_rows(height, cfg)
    count = min(int(cfg.stripe_count), int(anchors.size))
    chosen = rng.choice(anchors, size=count, replace=False)
    jitter_max = max(int(round(float(cfg.stripe_thickness) * 0.15)), 0)
    if jitter_max > 0:
        jitter = rng.integers(-jitter_max, jitter_max + 1, size=count)
        chosen = chosen + jitter
    return np.sort(np.clip(chosen.astype(np.int32), 1, max(height - 2, 1)))


def _stripe_vertical_profile(height: int, center: int, thickness: int) -> np.ndarray:
    yy = np.arange(height, dtype=np.float32)
    sigma = max(float(thickness) / 2.2, 1.0)
    profile = np.exp(-0.5 * ((yy - float(center)) / sigma) ** 2)
    profile /= max(float(profile.max()), 1e-6)
    return profile.astype(np.float32)


def _apply_stripes(image: np.ndarray, cfg: CorruptionConfig, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(image, dtype=np.float32).copy()
    height, _width = out.shape
    target_floor = float(cfg.stripe_target_intensity)
    for center in _sample_stripe_centers(height, cfg, rng):
        profile = _stripe_vertical_profile(height, int(center), int(cfg.stripe_thickness))
        blend = float(cfg.stripe_opacity) * profile[:, None]
        target = np.maximum(out, target_floor)
        out = out * (1.0 - blend) + target * blend
    return np.clip(out, 0.0, 255.0)


def _apply_ghost(image: np.ndarray, cfg: CorruptionConfig) -> np.ndarray:
    out = np.asarray(image, dtype=np.float32)
    shift = max(int(cfg.ghost_shift_px), 1)
    ghost = np.zeros_like(out, dtype=np.float32)
    if shift < out.shape[0]:
        ghost[shift:, :] = out[:-shift, :]
    else:
        ghost[...] = out.mean()
    mixed = out + cfg.ghost_opacity * ghost
    return np.clip(mixed, 0.0, 255.0)


def _raised_cosine_window(length: int, center: float, half_width: float) -> np.ndarray:
    coords = np.arange(length, dtype=np.float32)
    denom = max(float(half_width), 1.0)
    dist = np.abs(coords - float(center)) / denom
    window = np.zeros((length,), dtype=np.float32)
    inside = dist < 1.0
    window[inside] = 0.5 * (1.0 + np.cos(np.pi * dist[inside]))
    return window


def _apply_dropout(image: np.ndarray, cfg: CorruptionConfig, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(image, dtype=np.float32).copy()
    height, width = out.shape
    base_mean = float(out.mean())
    low_contrast = base_mean + float(cfg.dropout_contrast_scale) * (out - base_mean)
    low_contrast *= float(cfg.dropout_intensity_scale)

    for _ in range(int(cfg.dropout_regions)):
        half_w = max(int(round(float(cfg.dropout_width_px) * float(rng.uniform(0.85, 1.15)) / 2.0)), 12)
        half_h = max(int(round(float(cfg.dropout_height_px) * float(rng.uniform(0.85, 1.15)) / 2.0)), 20)

        cx_low = half_w
        cx_high = max(width - half_w - 1, cx_low)
        cy_low = max(half_h, 48)
        cy_high = min(max(height - half_h - 1, cy_low), 320)
        if cy_high < cy_low:
            cy_low = half_h
            cy_high = max(height - half_h - 1, cy_low)

        center_x = int(rng.integers(cx_low, cx_high + 1))
        center_y = int(rng.integers(cy_low, cy_high + 1))
        xw = _raised_cosine_window(width, center_x, half_w)
        yw = _raised_cosine_window(height, center_y, half_h)
        mask = np.outer(yw, xw)
        out = out * (1.0 - mask) + low_contrast * mask

    return np.clip(out, 0.0, 255.0)


def apply_oct_corruption(
    image: np.ndarray,
    *,
    corruption: str,
    severity: str,
    sample_key: str,
    seed: int = 0,
) -> np.ndarray:
    corruption_key = str(corruption).strip().lower()
    severity_key = str(severity).strip().lower()
    if corruption_key not in CORRUPTION_TYPES:
        raise ValueError(f"Unknown corruption={corruption!r}; expected one of {CORRUPTION_TYPES}")
    if corruption_key == "clean":
        return np.asarray(image, dtype=np.uint8).copy()
    if severity_key not in CORRUPTION_SEVERITIES:
        raise ValueError(f"Unknown severity={severity!r}; expected one of {CORRUPTION_SEVERITIES}")

    cfg = _SEVERITY_CONFIGS[severity_key]
    rng = _rng_for_sample(sample_key=sample_key, corruption=corruption_key, severity=severity_key, seed=int(seed))
    out = np.asarray(image, dtype=np.uint8)

    if corruption_key == "stripe":
        out_f = _apply_stripes(out, cfg, rng)
    elif corruption_key == "ghost":
        out_f = _apply_ghost(out, cfg)
    elif corruption_key == "dropout":
        out_f = _apply_dropout(out, cfg, rng)
    elif corruption_key == "combined":
        out_f = _apply_ghost(_apply_stripes(out, cfg, rng), cfg)
    else:  # pragma: no cover
        raise AssertionError(f"Unhandled corruption {corruption_key!r}")

    return np.clip(out_f, 0.0, 255.0).astype(np.uint8)


__all__ = [
    "CORRUPTION_TYPES",
    "CORRUPTION_SEVERITIES",
    "CorruptionConfig",
    "apply_oct_corruption",
    "corruption_output_suffix",
]
