#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates
import torch
from torch import amp
from torch.utils.data import DataLoader, Dataset, Subset


def _find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").exists() and (candidate / "dinoct").is_dir():
            return candidate
    raise RuntimeError("Could not locate repo root from script path.")


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.baselines import (  # noqa: E402
    LEARNED_BASELINE_MODELS,
    build_learned_baseline_model,
    decode_paper_unet_logits,
    is_segmentation_baseline_model_type,
)
from dinoct.data import make_dataset  # noqa: E402
from dinoct.data.datasets import OCT  # noqa: E402
from dinoct.data.transforms import Ensure3CH, MaybeToTensor, PerImageZScore  # noqa: E402
from dinoct.eval import (  # noqa: E402
    DEFAULT_ACC_TOLERANCES,
    curve_metrics_batch,
    estimate_spike_kappa_from_curves,
    metric_name_for_tolerance,
)
from dinoct.train.post_train import (  # noqa: E402
    CurveLoss,
    LossCfg,
    ORIG_H,
    ORIG_W,
    _collate_oct,
    _make_oct_transform,
    soft_argmax_height,
    validate,
)
from dinoct.train.train import get_cfg, load_training_cfg, resolve_dataset_path  # noqa: E402
from dinoct.utils.utils import fix_random_seeds  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train a supervised learned OCT baseline (UNet or FCBR).")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "train" / "oct.yaml")
    parser.add_argument("--model-type", choices=list(LEARNED_BASELINE_MODELS), required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--min-lr-mult", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--head-channels", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--lambda-curve", type=float, default=None)
    parser.add_argument("--lambda-curv", type=float, default=None)
    parser.add_argument("--bg-weight", type=float, default=5.0)
    parser.add_argument("--eps-none", type=float, default=0.02)
    parser.add_argument("--curv-delta", type=float, default=1.0)
    parser.add_argument("--spike-kappa", type=float, default=None)
    parser.add_argument("--spike-kappa-quantile", type=float, default=0.99)
    parser.add_argument("--seg-band-radius", type=int, default=2)
    parser.add_argument("--crop-height", type=int, default=464)
    parser.add_argument("--crop-width", type=int, default=256)
    parser.add_argument("--elastic-prob", type=float, default=1.0)
    parser.add_argument("--elastic-max-disp", type=float, default=12.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _pick_device(device_arg: str) -> torch.device:
    want = str(device_arg).strip().lower()
    if want == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # pragma: no cover
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(want)


def _split_indices(dataset: OCT, *, seed: int) -> tuple[list[int], list[int]]:
    entries = dataset._get_entries()

    if "split" in entries.dtype.names:
        split_values = np.char.lower(entries["split"].astype(str))
        codes = entries["code"]
        train_idx_np = np.nonzero((split_values == "train") & np.isin(codes, (1, 2)))[0]
        val_idx_np = np.nonzero((split_values == "val") & np.isin(codes, (1, 2)))[0]
        if train_idx_np.size > 0 and val_idx_np.size > 0:
            return train_idx_np.tolist(), val_idx_np.tolist()

    curve_idx = np.nonzero(entries["code"] == 1)[0]
    bg_idx = np.nonzero(entries["code"] == 2)[0]
    if curve_idx.size == 0:
        raise ValueError("Supervised baselines require labeled curve samples (entries with code==1).")

    rng = np.random.default_rng(int(seed))
    rng.shuffle(curve_idx)
    val_frac = 0.1
    val_curve = max(1, min(int(round(curve_idx.size * val_frac)), int(curve_idx.size) - 1))

    if bg_idx.size == 0:
        raise ValueError("Supervised baselines require background samples (entries with code==2).")
    rng.shuffle(bg_idx)
    val_bg = max(1, min(int(round(bg_idx.size * val_frac)), int(bg_idx.size) - 1))
    train_idx = np.concatenate([curve_idx[val_curve:], bg_idx[val_bg:]]).tolist()
    val_idx = np.concatenate([curve_idx[:val_curve], bg_idx[:val_bg]]).tolist()
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def _estimate_spike_kappa_for_indices(dataset: OCT, indices: list[int], *, quantile: float) -> float:
    curves: list[np.ndarray] = []
    for idx in indices:
        _image_np, y_curve = dataset[int(idx)]
        if y_curve is None:
            continue
        curves.append(np.asarray(y_curve, dtype=np.float32))
    if not curves:
        raise ValueError("Could not estimate spike kappa: validation indices contain no labeled curves.")
    return float(estimate_spike_kappa_from_curves(curves, quantile=float(quantile)))

def _image_to_tensor(image_np: np.ndarray) -> torch.Tensor:
    image = Image.fromarray(np.asarray(image_np, dtype=np.uint8), mode="L")
    tensor = MaybeToTensor()(image)
    tensor = Ensure3CH()(tensor)
    tensor = PerImageZScore(eps=1e-6)(tensor)
    return tensor


def _build_paper_seg_target(y_curve: np.ndarray | None, *, is_bg: bool, band_radius: int) -> np.ndarray:
    target = np.zeros((ORIG_H, ORIG_W), dtype=np.int64)
    if is_bg or y_curve is None:
        return target

    y_i = np.rint(np.asarray(y_curve, dtype=np.float32)).astype(np.int64)
    y_i = np.clip(y_i, 0, ORIG_H - 1)
    rows = np.arange(ORIG_H, dtype=np.int64)[:, None]
    top = np.clip(y_i[None, :] - int(band_radius), 0, ORIG_H - 1)
    bot = np.clip(y_i[None, :] + int(band_radius), 0, ORIG_H - 1)
    target[rows > bot] = 2
    band_mask = (rows >= top) & (rows <= bot)
    target[band_mask] = 1
    return target


def _curve_from_seg_target(seg_target: np.ndarray) -> np.ndarray:
    h, w = seg_target.shape
    y = np.zeros((w,), dtype=np.float32)
    band = np.asarray(seg_target == 1, dtype=bool)
    if not band.any():
        return y
    rows = np.arange(h, dtype=np.int64)[:, None]
    top = band.argmax(axis=0)
    bottom = h - 1 - np.flipud(band).argmax(axis=0)
    present = band.any(axis=0)
    y[present] = 0.5 * (top[present].astype(np.float32) + bottom[present].astype(np.float32))
    return y


def _elastic_piecewise_affine(
    image: np.ndarray,
    target: np.ndarray,
    *,
    rng: np.random.Generator,
    max_disp: float,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape
    grid_y = np.linspace(0.0, float(h - 1), 3, dtype=np.float32)
    grid_x = np.linspace(0.0, float(w - 1), 3, dtype=np.float32)
    disp_y = rng.uniform(-float(max_disp), float(max_disp), size=(3, 3)).astype(np.float32)
    disp_x = rng.uniform(-float(max_disp), float(max_disp), size=(3, 3)).astype(np.float32)

    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    points = np.stack([yy.reshape(-1), xx.reshape(-1)], axis=-1)
    interp_y = RegularGridInterpolator((grid_y, grid_x), disp_y, method="linear", bounds_error=False, fill_value=0.0)
    interp_x = RegularGridInterpolator((grid_y, grid_x), disp_x, method="linear", bounds_error=False, fill_value=0.0)
    flow_y = interp_y(points).reshape(h, w)
    flow_x = interp_x(points).reshape(h, w)

    coords = np.stack([yy + flow_y, xx + flow_x], axis=0)
    warped_image = map_coordinates(image.astype(np.float32), coords, order=1, mode="reflect").reshape(h, w)
    warped_target = map_coordinates(target.astype(np.float32), coords, order=0, mode="nearest").reshape(h, w)
    return np.clip(warped_image, 0.0, 255.0).astype(np.uint8), warped_target.astype(np.int64)


def _random_crop(
    image: np.ndarray,
    target: np.ndarray,
    *,
    rng: np.random.Generator,
    crop_h: int,
    crop_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape
    crop_h = min(int(crop_h), h)
    crop_w = min(int(crop_w), w)
    top = int(rng.integers(0, max(h - crop_h + 1, 1)))
    left = int(rng.integers(0, max(w - crop_w + 1, 1)))
    return image[top : top + crop_h, left : left + crop_w], target[top : top + crop_h, left : left + crop_w]


class PaperUNetSegDataset(Dataset):
    def __init__(
        self,
        base: OCT,
        indices: list[int],
        *,
        band_radius: int,
        train: bool,
        crop_height: int,
        crop_width: int,
        elastic_prob: float,
        elastic_max_disp: float,
        seed: int,
    ) -> None:
        self.base = base
        self.indices = [int(idx) for idx in indices]
        self.entries = base._get_entries()
        self.root = Path(base.root)
        self.band_radius = int(band_radius)
        self.train = bool(train)
        self.crop_height = int(crop_height)
        self.crop_width = int(crop_width)
        self.elastic_prob = float(elastic_prob)
        self.elastic_max_disp = float(elastic_max_disp)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        dataset_idx = int(self.indices[int(index)])
        entry = self.entries[dataset_idx]
        image = np.asarray(Image.open(self.root / str(entry["filename"])).convert("L"), dtype=np.uint8)
        y_curve = self.base.get_target(dataset_idx)
        is_bg = bool(y_curve is None or np.asarray(y_curve, dtype=np.float32).sum() == 0.0)
        seg_target = _build_paper_seg_target(y_curve, is_bg=is_bg, band_radius=self.band_radius)

        if self.train:
            call_seed = int(np.random.randint(0, 2**31 - 1))
            rng = np.random.default_rng(self.seed + dataset_idx * 1009 + call_seed)
            if rng.random() < 0.5:
                image = np.ascontiguousarray(np.fliplr(image))
                seg_target = np.ascontiguousarray(np.fliplr(seg_target))
            if rng.random() < self.elastic_prob:
                image, seg_target = _elastic_piecewise_affine(
                    image,
                    seg_target,
                    rng=rng,
                    max_disp=self.elastic_max_disp,
                )
            image, seg_target = _random_crop(
                image,
                seg_target,
                rng=rng,
                crop_h=self.crop_height,
                crop_w=self.crop_width,
            )

        image_t = _image_to_tensor(image)
        seg_t = torch.from_numpy(np.asarray(seg_target, dtype=np.int64))
        y_t = torch.from_numpy(_curve_from_seg_target(seg_target))
        return {
            "image": image_t,
            "seg_target": seg_t,
            "y": y_t,
            "is_bg": torch.tensor(1 if is_bg else 0, dtype=torch.long),
        }


def _collate_seg(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "seg_target": torch.stack([item["seg_target"] for item in batch], dim=0),
        "y": torch.stack([item["y"] for item in batch], dim=0),
        "is_bg": torch.stack([item["is_bg"] for item in batch], dim=0),
    }


def _compute_seg_class_weights(dataset: OCT, indices: list[int], *, band_radius: int) -> torch.Tensor:
    counts = np.zeros((3,), dtype=np.float64)
    full_cols = ORIG_H * ORIG_W
    for idx in indices:
        y_curve = dataset.get_target(int(idx))
        is_bg = bool(y_curve is None or np.asarray(y_curve, dtype=np.float32).sum() == 0.0)
        if is_bg:
            counts[0] += full_cols
            continue
        target = _build_paper_seg_target(y_curve, is_bg=False, band_radius=band_radius)
        bincount = np.bincount(target.reshape(-1), minlength=3).astype(np.float64)
        counts += bincount
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(counts) * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights.astype(np.float32))


def _train_step_segmentation(
    *,
    batch: dict[str, torch.Tensor],
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    images = batch["image"].to(device, non_blocking=True)
    seg_target = batch["seg_target"].to(device, non_blocking=True)
    y = batch["y"].to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    use_amp = bool(device.type == "cuda" and scaler.is_enabled())
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    optimizer_stepped = False
    with amp.autocast(device_type=autocast_device, enabled=use_amp):
        seg_logits = model(images, orig_hw=tuple(seg_target.shape[-2:]))
        loss = criterion(seg_logits, seg_target)

    if use_amp:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        prev_scale = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        optimizer_stepped = float(scaler.get_scale()) >= prev_scale
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer_stepped = True

    with torch.no_grad():
        y_hat, presence_logits = decode_paper_unet_logits(seg_logits)
        mask = (1 - batch["is_bg"].to(device).float())
        mae = ((y_hat - y).abs().mean(dim=1) * mask).sum() / (mask.sum() + 1e-8)
    return {
        "loss": float(loss.detach().cpu()),
        "mae_px": float(mae.detach().cpu()),
        "loss_col_ce": float(loss.detach().cpu()),
        "loss_smooth": 0.0,
        "p_curve": float(torch.sigmoid(presence_logits).mean().detach().cpu()),
        "optimizer_stepped": float(1.0 if optimizer_stepped else 0.0),
    }


def _validate_segmentation(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
    *,
    acc_tolerances: tuple[float, ...],
    spike_kappa: float,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    n_batches = 0
    metric_names = ["mae_px", "p95_px", "bias_px", "abs_bias_px"] + [
        metric_name_for_tolerance(tau) for tau in acc_tolerances
    ] + ["spike_rate"]
    metric_lists: dict[str, list[torch.Tensor]] = {name: [] for name in metric_names}

    with torch.inference_mode():
        for batch in val_loader:
            images = batch["image"].to(device, non_blocking=True)
            seg_target = batch["seg_target"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            is_bg = batch["is_bg"].to(device, non_blocking=True)

            seg_logits = model(images, orig_hw=tuple(seg_target.shape[-2:]))
            loss = criterion(seg_logits, seg_target)
            loss_sum += float(loss.detach().cpu())
            n_batches += 1

            y_hat, _presence = decode_paper_unet_logits(seg_logits)
            non_bg = (is_bg == 0)
            if not bool(non_bg.any()):
                continue
            batch_metrics = curve_metrics_batch(
                y_hat[non_bg],
                y[non_bg],
                acc_tolerances=acc_tolerances,
                spike_kappa=spike_kappa,
            )
            for name, values in batch_metrics.items():
                metric_lists[name].append(values.detach().cpu())

    out = {"val_loss": loss_sum / max(n_batches, 1)}
    for name in metric_names:
        if metric_lists[name]:
            out[f"val_{name}"] = float(torch.cat(metric_lists[name], dim=0).mean().item())
        else:
            out[f"val_{name}"] = float("nan")
    return out


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    model_type: str,
    model_kwargs: dict[str, Any],
    dataset_path: str,
    step: int,
    val_metrics: dict[str, float] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": str(model_type),
            "model_kwargs": model_kwargs,
            "dataset_path": dataset_path,
            "step": int(step),
            "model": model.state_dict(),
            "val_metrics": val_metrics or {},
        },
        path,
    )


def _train_step(
    *,
    batch: dict[str, torch.Tensor],
    model: torch.nn.Module,
    criterion: CurveLoss,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    images = batch["image"].to(device, non_blocking=True)
    is_bg = batch["is_bg"].to(device, non_blocking=True).long()
    y = batch["y"].to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    use_amp = bool(device.type == "cuda" and scaler.is_enabled())
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    with amp.autocast(device_type=autocast_device, enabled=use_amp):
        presence_logits, curve_logits = model(images, orig_hw=(ORIG_H, ORIG_W))
        loss, metrics = criterion(curve_logits, y, is_bg)

    if use_amp:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        prev_scale = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        optimizer_stepped = float(scaler.get_scale()) >= prev_scale
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    with torch.no_grad():
        p_curve = torch.sigmoid(presence_logits)
        mask = (1 - is_bg).float()
        y_hat = soft_argmax_height(curve_logits[:, :-1, :])
        mae = ((y_hat - y).abs().mean(dim=1) * mask).sum() / (mask.sum() + 1e-8)
    return {
        "loss": float(loss.detach().cpu()),
        "mae_px": float(mae.detach().cpu()),
        **{k: float(v.detach().cpu()) for k, v in metrics.items()},
        "p_curve": float(p_curve.mean().detach().cpu()),
        "optimizer_stepped": float(1.0 if optimizer_stepped else 0.0),
    }


def main() -> None:
    args = parse_args()
    cfg = load_training_cfg(args.config)
    post_cfg = cfg.get("post_train", {})
    fix_random_seeds(int(args.seed))
    model_type = str(args.model_type).strip().lower()
    is_segmentation_model = is_segmentation_baseline_model_type(model_type)

    device = _pick_device(args.device)
    dataset_str = resolve_dataset_path(str(get_cfg(cfg, ("train", "dataset_path"), "OCT:root=data/oct:extra=data/oct/extra")))
    dataset_transform = None if is_segmentation_model else _make_oct_transform()
    dataset_full = make_dataset(dataset_str=dataset_str, transform=dataset_transform)
    if not isinstance(dataset_full, OCT):
        raise TypeError(f"Expected OCT dataset, got {type(dataset_full)}")

    train_idx, val_idx = _split_indices(dataset_full, seed=int(args.seed))
    if is_segmentation_model:
        ds_train: Dataset[Any] = PaperUNetSegDataset(
            dataset_full,
            train_idx,
            band_radius=int(args.seg_band_radius),
            train=True,
            crop_height=int(args.crop_height),
            crop_width=int(args.crop_width),
            elastic_prob=float(args.elastic_prob),
            elastic_max_disp=float(args.elastic_max_disp),
            seed=int(args.seed),
        )
        ds_val: Dataset[Any] = PaperUNetSegDataset(
            dataset_full,
            val_idx,
            band_radius=int(args.seg_band_radius),
            train=False,
            crop_height=int(args.crop_height),
            crop_width=int(args.crop_width),
            elastic_prob=0.0,
            elastic_max_disp=float(args.elastic_max_disp),
            seed=int(args.seed),
        )
        collate_fn = _collate_seg
    else:
        ds_train = Subset(dataset_full, train_idx)
        ds_val = Subset(dataset_full, val_idx)
        collate_fn = _collate_oct

    batch_size_default = 4 if is_segmentation_model else int(post_cfg.get("batch_size", 64))
    batch_size = int(args.batch_size or batch_size_default)
    steps = int(args.steps or post_cfg.get("steps", 1500))
    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, int(args.num_workers) // 2),
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
    )

    model_kwargs = {
        "in_chans": 3,
        "base_channels": int(args.base_channels),
        "head_channels": int(args.head_channels) if args.head_channels is not None else None,
    }
    if is_segmentation_model:
        model_kwargs = {
            "in_chans": 3,
            "base_channels": int(args.base_channels),
        }
    model = build_learned_baseline_model(model_type, **model_kwargs).to(device)

    if is_segmentation_model:
        class_weights = _compute_seg_class_weights(dataset_full, train_idx, band_radius=int(args.seg_band_radius)).to(device)
        criterion: Any = torch.nn.CrossEntropyLoss(weight=class_weights)
        lr = float(args.lr if args.lr is not None else 1e-2)
        weight_decay = float(args.weight_decay if args.weight_decay is not None else 0.0)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(args.momentum),
            weight_decay=weight_decay,
            nesterov=False,
        )
    else:
        criterion = CurveLoss(
            LossCfg(
                sigma=float(args.sigma if args.sigma is not None else post_cfg.get("sigma", 2.0)),
                lambda_curve=float(args.lambda_curve if args.lambda_curve is not None else post_cfg.get("lambda_curve", 1.0)),
                lambda_curv=float(args.lambda_curv if args.lambda_curv is not None else post_cfg.get("lambda_curv", 0.05)),
                bg_weight=float(args.bg_weight),
                eps_none=float(args.eps_none),
                curv_delta=float(args.curv_delta),
            )
        )
        lr = float(args.lr if args.lr is not None else 1e-3)
        weight_decay = float(args.weight_decay if args.weight_decay is not None else 5e-4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = amp.GradScaler("cuda", enabled=device.type == "cuda")

    warmup_steps = min(max(int(args.warmup_steps), 0), max(int(steps), 1))
    min_lr_mult = float(args.min_lr_mult)

    def _lr_mult(step_num: int) -> float:
        step_num = max(int(step_num), 1)
        if warmup_steps > 0 and step_num <= warmup_steps:
            return step_num / warmup_steps
        t = (step_num - warmup_steps) / max(1, int(steps) - warmup_steps)
        return min_lr_mult + (1.0 - min_lr_mult) * 0.5 * (1.0 + math.cos(math.pi * t))

    lr_mult_1 = float(_lr_mult(1))
    if not math.isfinite(lr_mult_1) or lr_mult_1 <= 0.0:
        lr_mult_1 = 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = float(param_group.get("lr", 0.0)) * lr_mult_1
    scheduler = (
        torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: _lr_mult(step + 2) / lr_mult_1)
        if steps > 1
        else None
    )

    output_dir = args.output_dir or (REPO_ROOT / "outputs" / "learned_baselines" / str(args.model_type))
    output_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt = output_dir / "curve_final.pth"
    best_ckpt = output_dir / "curve_best.pth"
    metrics_path = output_dir / "metrics.csv"

    with metrics_path.open("w", newline="") as metrics_fh:
        writer = csv.writer(metrics_fh)
        writer.writerow(
            [
                "step",
                "loss",
                "mae_px",
                "loss_col_ce",
                "loss_smooth",
                "p_curve",
                "val_loss",
                "val_mae_px",
                "val_p95_px",
                "val_acc_2px",
                "val_spike_rate",
                "lr",
            ]
        )

        data_iter = iter(train_loader)
        best_val_mae = float("inf")
        best_val_metrics: dict[str, float] | None = None
        spike_kappa_value = (
            float(args.spike_kappa)
            if args.spike_kappa is not None
            else _estimate_spike_kappa_for_indices(dataset_full, val_idx, quantile=float(args.spike_kappa_quantile))
        )

        for step in range(1, steps + 1):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            if is_segmentation_model:
                train_stats = _train_step_segmentation(
                    batch=batch,
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    scaler=scaler,
                    device=device,
                )
            else:
                train_stats = _train_step(
                    batch=batch,
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    scaler=scaler,
                    device=device,
                )
            if scheduler is not None and step < steps and bool(train_stats.get("optimizer_stepped", 0.0) > 0.5):
                scheduler.step()

            current_lr = float(optimizer.param_groups[0].get("lr", 0.0)) if optimizer.param_groups else 0.0
            if step % int(args.log_every) == 0 or step == 1:
                print(
                    f"[{args.model_type} {step}/{steps}] loss={train_stats['loss']:.4f} "
                    f"mae_px={train_stats['mae_px']:.3f} "
                    f"Lcol={train_stats.get('loss_col_ce', 0.0):.4f} "
                    f"Lsmooth={train_stats.get('loss_smooth', 0.0):.4f} "
                    f"lr={current_lr:.2e}"
                )

            val_metrics: dict[str, float] | None = None
            if step % int(args.eval_every) == 0 or step == steps:
                if is_segmentation_model:
                    val_metrics = _validate_segmentation(
                        model,
                        val_loader,
                        device,
                        criterion,
                        acc_tolerances=DEFAULT_ACC_TOLERANCES,
                        spike_kappa=spike_kappa_value,
                    )
                else:
                    val_metrics = validate(
                        model,
                        val_loader,
                        device,
                        criterion,
                        acc_tolerances=DEFAULT_ACC_TOLERANCES,
                    )
                writer.writerow(
                    [
                        step,
                        train_stats.get("loss", 0.0),
                        train_stats.get("mae_px", 0.0),
                        train_stats.get("loss_col_ce", 0.0),
                        train_stats.get("loss_smooth", 0.0),
                        train_stats.get("p_curve", 0.0),
                        val_metrics.get("val_loss", float("nan")),
                        val_metrics.get("val_mae_px", float("nan")),
                        val_metrics.get("val_p95_px", float("nan")),
                        val_metrics.get("val_acc_2px", float("nan")),
                        val_metrics.get("val_spike_rate", float("nan")),
                        current_lr,
                    ]
                )
                metrics_fh.flush()
                print(
                    f"[{args.model_type} val] step={step} "
                    f"val_loss={val_metrics.get('val_loss', float('nan')):.4f} "
                    f"val_mae={val_metrics.get('val_mae_px', float('nan')):.3f} "
                    f"val_p95={val_metrics.get('val_p95_px', float('nan')):.3f} "
                    f"val_acc2={val_metrics.get('val_acc_2px', float('nan')):.3f}"
                )
                val_mae = float(val_metrics.get("val_mae_px", float("inf")))
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_val_metrics = val_metrics
                    _save_checkpoint(
                        best_ckpt,
                        model=model,
                        model_type=model_type,
                        model_kwargs=model_kwargs,
                        dataset_path=dataset_str,
                        step=step,
                        val_metrics=val_metrics,
                    )

        _save_checkpoint(
            final_ckpt,
            model=model,
            model_type=model_type,
            model_kwargs=model_kwargs,
            dataset_path=dataset_str,
            step=steps,
            val_metrics=best_val_metrics,
        )

    summary = {
        "model_type": model_type,
        "model_kwargs": model_kwargs,
        "dataset_path": dataset_str,
        "steps": int(steps),
        "batch_size": int(batch_size),
        "device": str(device),
        "spike_kappa": spike_kappa_value,
        "best_val_metrics": best_val_metrics,
        "checkpoints": {
            "best": str(best_ckpt),
            "final": str(final_ckpt),
        },
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    best_mae = float(best_val_metrics.get("val_mae_px", float("nan"))) if best_val_metrics else float("nan")
    best_p95 = float(best_val_metrics.get("val_p95_px", float("nan"))) if best_val_metrics else float("nan")
    best_acc2 = float(best_val_metrics.get("val_acc_2px", float("nan"))) if best_val_metrics else float("nan")
    print(
        f"[{args.model_type}] done. best_val_mae={best_mae:.3f} "
        f"best_val_p95={best_p95:.3f} best_val_acc2={best_acc2:.3f}"
    )
    print(f"[{args.model_type}] wrote {best_ckpt}")
    print(f"[{args.model_type}] wrote {final_ckpt}")
    print(f"[{args.model_type}] wrote {output_dir / 'train_summary.json'}")


if __name__ == "__main__":
    main()
