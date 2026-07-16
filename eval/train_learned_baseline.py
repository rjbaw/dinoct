#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import amp
from torch.utils.data import DataLoader


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
)
from dinoct.data import make_dataset  # noqa: E402
from dinoct.data.datasets import OCT  # noqa: E402
from dinoct.eval import (  # noqa: E402
    DEFAULT_ACC_TOLERANCES,
    estimate_spike_kappa_from_curves,
)
from dinoct.train.post_train import (  # noqa: E402
    CurveLoss,
    LossCfg,
    ORIG_H,
    ORIG_W,
    _OCTCodeSubset,
    _collate_oct,
    _make_oct_transform,
    soft_argmax_height,
    validate,
)
from dinoct.train.train import get_cfg, load_training_cfg, resolve_dataset_path  # noqa: E402
from dinoct.utils.utils import fix_random_seeds, seed_worker  # noqa: E402


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
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--head-channels", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--lambda-curve", type=float, default=None)
    parser.add_argument("--lambda-curv", type=float, default=None)
    parser.add_argument(
        "--bg-weight",
        type=float,
        default=None,
        help="Background CE weight; falls back to post_train.bg_weight in the config, then 5.0.",
    )
    parser.add_argument("--eps-none", type=float, default=0.02)
    parser.add_argument("--curv-delta", type=float, default=1.0)
    parser.add_argument("--spike-kappa", type=float, default=None)
    parser.add_argument("--spike-kappa-quantile", type=float, default=0.99)
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

    logging.getLogger("dinoct").warning(
        "No explicit train/val split found (extra/manifest.csv + extra/splits.csv): using a "
        "seeded RANDOM 90/10 image-level split. This is fine for general training on your own "
        "data, but frames from the same recording can land in both partitions, so val is NOT "
        "leak-safe. For recording-level splits (the paper protocol) generate metadata with "
        "tools/data/build_oct_manifest.py + tools/data/build_oct_splits.py."
    )
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
        _image_np, z_curve = dataset[int(idx)]
        if z_curve is None:
            continue
        curves.append(np.asarray(z_curve, dtype=np.float32))
    if not curves:
        raise ValueError("Could not estimate spike kappa: validation indices contain no labeled curves.")
    return float(estimate_spike_kappa_from_curves(curves, quantile=float(quantile)))

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
    z = batch["z"].to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    use_amp = bool(device.type == "cuda" and scaler.is_enabled())
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    with amp.autocast(device_type=autocast_device, enabled=use_amp):
        presence_logits, curve_logits = model(images, orig_hw=(ORIG_H, ORIG_W))
        loss, metrics = criterion(curve_logits, z, is_bg)

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
        optimizer_stepped = True  # no GradScaler skip on the non-AMP (CPU/MPS) path

    with torch.no_grad():
        p_curve = torch.sigmoid(presence_logits)
        mask = (1 - is_bg).float()
        z_hat = soft_argmax_height(curve_logits[:, :-1, :])
        mae = ((z_hat - z).abs().mean(dim=1) * mask).sum() / (mask.sum() + 1e-8)
    return {
        "loss": float(loss.detach().cpu()),
        "mae_px": float(mae.detach().cpu()),
        **{k: float(v.detach().cpu()) for k, v in metrics.items()},
        "p_curve": float(p_curve.mean().detach().cpu()),
        "optimizer_stepped": float(1.0 if optimizer_stepped else 0.0),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    cfg = load_training_cfg(args.config)
    post_cfg = cfg.get("post_train", {})
    fix_random_seeds(int(args.seed))
    model_type = str(args.model_type).strip().lower()

    device = _pick_device(args.device)
    dataset_str = resolve_dataset_path(str(get_cfg(cfg, ("train", "dataset_path"), "OCT:root=data/oct:extra=data/oct/extra")))

    dataset_full = make_dataset(dataset_str=dataset_str, transform=_make_oct_transform())
    if not isinstance(dataset_full, OCT):
        raise TypeError(f"Expected OCT dataset, got {type(dataset_full)}")

    train_idx, val_idx = _split_indices(dataset_full, seed=int(args.seed))
    ds_train = _OCTCodeSubset(dataset_full, train_idx)
    ds_val = _OCTCodeSubset(dataset_full, val_idx)
    collate_fn = _collate_oct

    batch_size_default = int(post_cfg.get("batch_size", 64))
    batch_size = int(args.batch_size or batch_size_default)
    steps = int(args.steps or post_cfg.get("steps", 1500))
    train_generator = torch.Generator()
    train_generator.manual_seed(int(args.seed))
    val_generator = torch.Generator()
    val_generator.manual_seed(int(args.seed) + 1)
    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
        generator=train_generator,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, int(args.num_workers) // 2),
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
        generator=val_generator,
        worker_init_fn=seed_worker,
    )

    model_kwargs = {
        "in_chans": 3,
        "base_channels": int(args.base_channels),
        "head_channels": int(args.head_channels) if args.head_channels is not None else None,
    }
    model = build_learned_baseline_model(model_type, **model_kwargs).to(device)

    criterion = CurveLoss(
        LossCfg(
            sigma=float(args.sigma if args.sigma is not None else post_cfg.get("sigma", 2.0)),
            lambda_curve=float(args.lambda_curve if args.lambda_curve is not None else post_cfg.get("lambda_curve", 1.0)),
            lambda_curv=float(args.lambda_curv if args.lambda_curv is not None else post_cfg.get("lambda_curv", 0.05)),
            bg_weight=float(args.bg_weight if args.bg_weight is not None else post_cfg.get("bg_weight", 5.0)),
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
