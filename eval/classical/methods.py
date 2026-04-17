from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import convolve, gaussian_filter, median_filter
from scipy.signal import savgol_filter


@dataclass(frozen=True, slots=True)
class ClassicalMethodSpec:
    key: str
    display_name: str
    description: str
    requires_background: bool = False


CLASSICAL_METHOD_SPECS: dict[str, ClassicalMethodSpec] = {
    "gf": ClassicalMethodSpec(
        key="gf",
        display_name="GF",
        description="Gradient-filter baseline with row-wise DC suppression.",
    ),
    "gf_b": ClassicalMethodSpec(
        key="gf_b",
        display_name="GF-B",
        description="Gradient-filter baseline augmented with background subtraction before DC suppression.",
        requires_background=True,
    ),
    "grad_sg": ClassicalMethodSpec(
        key="grad_sg",
        display_name="GRAD-SG",
        description="Gradient method with row-wise DC suppression, smoothing, and Savitzky-Golay post-processing.",
    ),
    "grad_eng": ClassicalMethodSpec(
        key="grad_eng",
        display_name="GRAD-ENG",
        description="Gradient method with row-wise DC suppression, weighted gradients, and stronger 1D post-processing.",
    ),
    "legacy_sobel_dc": ClassicalMethodSpec(
        key="legacy_sobel_dc",
        display_name="TUNED-SOBEL-DC",
        description="Legacy Sobel-plus-manual-DC-suppression detector ported from bench/process_image.py.",
    ),
}

# Backward-compatible alias for older code paths.
CLASSICAL_METHODS = CLASSICAL_METHOD_SPECS

METHOD_ALIASES: dict[str, str] = {
    "engineered": "grad_eng",
    "detect_lines_savgol": "grad_sg",
    "detect_lines_old": "legacy_sobel_dc",
}

AVAILABLE_METHODS: tuple[str, ...] = tuple([*CLASSICAL_METHOD_SPECS.keys(), *METHOD_ALIASES.keys()])


def resolve_method_key(method: str) -> str:
    method_key = str(method).strip().lower()
    return METHOD_ALIASES.get(method_key, method_key)


def _ensure_gray_u8(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    if img.ndim != 2:
        raise ValueError(f"Expected grayscale image with shape (H, W), got {tuple(img.shape)!r}")
    if img.dtype == np.uint8:
        return img
    img_f = img.astype(np.float32)
    img_f = np.clip(img_f, 0.0, 255.0)
    return img_f.astype(np.uint8)


def _normalize01(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.float32)
    img -= float(img.min())
    mx = float(img.max())
    if mx > 0.0:
        img /= mx
    return img


def _suppress_row_dc(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.float32)
    return img - img.mean(axis=1, keepdims=True)


def _subtract_background(raw: np.ndarray, background: np.ndarray) -> np.ndarray:
    raw_f = np.asarray(raw, dtype=np.float32)
    bg_f = np.asarray(background, dtype=np.float32)
    return raw_f - bg_f


def _build_gaussian_filter(nx: int, ny: int) -> np.ndarray:
    x = np.arange(nx, dtype=np.float32) - (nx - 1) / 2
    y = np.arange(ny, dtype=np.float32) - (ny - 1) / 2
    xx, yy = np.meshgrid(x, y)
    kernel = np.exp(-(xx**2) / max((nx / 4) ** 2, 1e-6)) * np.exp(-(yy**2) / max((ny / 4) ** 2, 1e-6))
    total = float(kernel.sum())
    if total > 0.0:
        kernel /= total
    return kernel


def _lowpass(image: np.ndarray, nx: int, ny: int) -> np.ndarray:
    kernel = _build_gaussian_filter(nx, ny)
    return convolve(np.asarray(image, dtype=np.float32), kernel, mode="nearest")


def _weighted_gradient(image: np.ndarray) -> np.ndarray:
    img_f = np.asarray(image, dtype=np.float32)
    kx = np.array([[0.0, 0.0, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]], dtype=np.float32)
    ky = kx.T
    gy = convolve(img_f, ky, mode="nearest")
    gx = convolve(img_f, kx, mode="nearest")
    return np.sqrt(np.maximum(0.65 * gy**2 + gx**2, 0.0), dtype=np.float32)


def _positive_vertical_gradient(image: np.ndarray) -> np.ndarray:
    grad_y = np.gradient(np.asarray(image, dtype=np.float32), axis=0)
    return np.maximum(grad_y, 0.0)


def _pick_column_peaks(score: np.ndarray) -> np.ndarray:
    score_f = np.asarray(score, dtype=np.float32)
    return np.argmax(score_f, axis=0).astype(np.float32)


def _median_filter_1d(signal: np.ndarray, window_size: int) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float32).copy()
    window = max(int(window_size), 1)
    if window % 2 == 0:
        window += 1
    half_win = window // 2
    padded = np.pad(values, (half_win, half_win), mode="edge")
    for i in range(values.shape[0]):
        values[i] = float(np.median(padded[i : i + window]))
    return values


def _zero_dc_legacy(image: np.ndarray, zidx: tuple[int, ...], window: int) -> np.ndarray:
    img = np.asarray(image, dtype=np.float32).copy()
    height = img.shape[0]
    for i in zidx:
        start_idx = max(int(i) - window // 2, 0)
        end_idx = min(int(i) + window // 2, height)
        filter_window = img[start_idx:end_idx, :]
        mean_col = np.mean(filter_window, axis=1, keepdims=True)
        img[start_idx:end_idx, :] -= mean_col
    return np.clip(img, 0.0, 255.0).astype(np.uint8)


def _ol_removal(signal: np.ndarray) -> np.ndarray:
    observations = np.asarray(signal, dtype=np.float32).copy()
    obs_length = observations.shape[0]
    window = max(1, obs_length // 3)
    z_max = 40.0
    slope = 0.0
    sigma_best = np.inf
    for seg_idx in range(max(obs_length // 3, 1)):
        start = max(0, seg_idx * window)
        end = min((seg_idx + 1) * window, obs_length - 1)
        segment = observations[start:end].copy()
        if segment.size == 0:
            continue
        sigma = float(np.std(segment))
        segment_f = _median_filter_1d(segment, window)
        if sigma < sigma_best:
            sigma_best = sigma
            start_seg = float(segment_f[0])
            end_seg = float(segment_f[-1])
            slope_new = (end_seg - start_seg) / max(float(segment_f.shape[0]), 1.0)
            if abs(slope_new) < z_max:
                slope = slope_new

    for i, value in enumerate(observations):
        if i == 0:
            observations[0] = float(np.median(observations[:30]))
            continue
        prev_value = float(observations[i - 1])
        mse = abs(float(value) - prev_value)
        if mse > z_max:
            observations[i] = prev_value + slope
    return observations


def _kalman_filter(signal: np.ndarray) -> np.ndarray:
    observations = np.asarray(signal, dtype=np.float32)
    x_k = float(np.median(observations[:10]))
    p_k = 1.0
    q = 0.01
    r = 0.5
    estimates = np.empty_like(observations, dtype=np.float32)
    for idx, z_k in enumerate(observations):
        x_pred = x_k
        p_pred = p_k + q
        k_gain = p_pred / (p_pred + r)
        x_k = x_pred + k_gain * (float(z_k) - x_pred)
        p_k = (1.0 - k_gain) * p_pred
        estimates[idx] = x_k
    return estimates


def _prepare_base_image(raw: np.ndarray, background: np.ndarray | None) -> np.ndarray:
    if background is not None:
        base = _subtract_background(raw, background)
    else:
        base = np.asarray(raw, dtype=np.float32)
    return _suppress_row_dc(base)


def _detect_grad_sg(raw: np.ndarray) -> np.ndarray:
    img_raw = np.asarray(raw, dtype=np.uint8)
    img = median_filter(img_raw, size=(5, 5))
    img = img.astype(np.float32) - np.mean(img, axis=1, keepdims=True)
    img = median_filter(img, size=(3, 11))
    img = gaussian_filter(img, sigma=3)

    gy = np.gradient(img, axis=0)
    surface = np.argmax(gy, axis=0).astype(np.float32)
    observations = surface.copy()

    x_k = float(np.median(observations[:50]))
    p_k = 1.0
    q = 0.01
    r = 0.5
    for idx, z_k in enumerate(observations):
        x_pred = x_k
        p_pred = p_k + q
        k_gain = p_pred / (p_pred + r)
        x_k = x_pred + k_gain * (float(z_k) - x_pred)
        p_k = (1.0 - k_gain) * p_pred
        surface[idx] = x_k

    observations = surface.copy()
    obs_length = len(observations)
    window = max(1, obs_length // 10)
    for i, pt in enumerate(observations):
        start = max(0, i - window)
        end = min(obs_length, i + window + 1)
        local = observations[start:end]
        mu = float(np.mean(local))
        sigma = float(np.std(local))
        med = float(np.median(local))
        z = (float(pt) - mu) / sigma if sigma != 0.0 else float(pt) - mu
        if z > 0.5:
            observations[i] = mu if sigma != 0.0 and ((med - mu) / sigma) > 3.0 else med

    window_length = min(15, observations.shape[0] if observations.shape[0] % 2 == 1 else observations.shape[0] - 1)
    if window_length < 5:
        return observations.astype(np.float32)
    return savgol_filter(observations + 1.0, window_length=window_length, polyorder=3).astype(np.float32)


def _detect_legacy_sobel_dc(raw: np.ndarray) -> np.ndarray:
    img_raw = np.asarray(raw, dtype=np.uint8)
    kx = np.array([[1, 2, 0, -2, -1]], dtype=np.float32)
    ky = kx.T
    img = convolve(img_raw.astype(np.float32), ky, mode="nearest")
    img = np.clip(img, 0.0, 255.0).astype(np.uint8)
    img = median_filter(img, size=(11, 11))
    img = gaussian_filter(img, sigma=(11 / 6, 11 / 6))
    img = _zero_dc_legacy(img, (0, 5, 12, 24, 37, 51, 63, 75, 87, 99, 112, 126), 14)

    surface = _pick_column_peaks(img).astype(np.float32)
    observations = surface.copy()
    obs_length = len(observations)
    window = max(1, obs_length // 20)
    for i, pt in enumerate(observations):
        start = max(0, i - window)
        end = min(obs_length, i + window + 1)
        local = observations[start:end]
        mu = float(np.mean(local))
        sigma = float(np.std(local))
        med = float(np.median(local))
        z = abs(float(pt) - mu) / sigma if sigma != 0.0 else abs(float(pt) - mu)
        if z > 0.5:
            observations[i] = med
    return observations.astype(np.float32)


def _detect_gf(raw: np.ndarray, background: np.ndarray | None = None) -> np.ndarray:
    base = _prepare_base_image(raw, background)
    smoothed = gaussian_filter(base, sigma=(1.0, 2.0), mode="nearest")
    score = _positive_vertical_gradient(smoothed)
    return _pick_column_peaks(score)


def _detect_grad_eng(raw: np.ndarray, background: np.ndarray | None = None) -> np.ndarray:
    base = _prepare_base_image(raw, background)
    base = _normalize01(base)
    base = _lowpass(base, 11, 5)
    grad = _weighted_gradient(base)
    grad = _lowpass(grad, 1, 3)
    score = _normalize01(base * grad)
    surface = _pick_column_peaks(score)
    surface = _median_filter_1d(surface, 15)
    surface = _ol_removal(surface)
    surface = _kalman_filter(surface)
    return surface.astype(np.float32)


def detect_surface_curve(method: str, raw: np.ndarray, background: np.ndarray | None = None) -> np.ndarray:
    method_key = resolve_method_key(method)
    if method_key not in CLASSICAL_METHOD_SPECS:
        raise ValueError(f"Unknown classical method {method!r}; expected one of {sorted(AVAILABLE_METHODS)}")

    raw_gray = _ensure_gray_u8(raw)
    bg_gray = _ensure_gray_u8(background) if background is not None else None

    if method_key == "gf":
        return _detect_gf(raw_gray, background=None)
    if method_key == "gf_b":
        if bg_gray is None:
            raise ValueError("gf_b requires a paired background image.")
        return _detect_gf(raw_gray, background=bg_gray)
    if method_key == "grad_eng":
        return _detect_grad_eng(raw_gray, background=bg_gray)
    if method_key == "legacy_sobel_dc":
        return _detect_legacy_sobel_dc(raw_gray)
    if method_key == "grad_sg":
        return _detect_grad_sg(raw_gray)

    raise AssertionError(f"Unhandled classical method {method_key!r}")


def apply_classical_method(method: str, raw: np.ndarray, *, background: np.ndarray | None = None) -> np.ndarray:
    return detect_surface_curve(method, raw, background=background)
