#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset


def _find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").exists() and (candidate / "dinoct").is_dir():
            return candidate
    raise RuntimeError("Could not locate repo root from script path.")


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dinoct.data import make_dataset  # noqa: E402
from dinoct.data.datasets import OCT  # noqa: E402
from dinoct.eval import (  # noqa: E402
    DEFAULT_ACC_TOLERANCES,
    average_metric_rows,
    curve_metrics_batch,
    estimate_spike_kappa_from_curves,
    metric_name_for_tolerance,
    summarize_metric_rows,
)
from eval.baselines import (  # noqa: E402
    LEARNED_BASELINE_MODELS,
    build_learned_baseline_model,
    decode_paper_unet_logits,
    infer_model_type_from_checkpoint,
    is_segmentation_baseline_model_type,
)
from eval.corruptions import (  # noqa: E402
    CORRUPTION_SEVERITIES,
    CORRUPTION_TYPES,
    apply_oct_corruption,
    corruption_output_suffix,
)
from eval.evalset import DirectoryCurveEvalDataset, DirectoryEvalConfig, split_rows_for_directory_dataset  # noqa: E402
from dinoct.models import build_backbone  # noqa: E402
from dinoct.train.post_train import (  # noqa: E402
    CurveModel,
    ORIG_H,
    ORIG_W,
    _make_oct_transform,
    soft_argmax_height,
)
from dinoct.train.train import get_cfg, load_training_cfg, resolve_dataset_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate a fused OCT curve checkpoint on a dataset split.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "train" / "oct.yaml")
    parser.add_argument("--curve-ckpt", type=Path, default=None, help="Path to fused_curve_best.pth or fused_curve.pth")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--eval-dir", type=Path, default=None, help="Optional directory of .jpg/.txt pairs for a separate eval subset")
    parser.add_argument("--kappa-split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--spike-kappa", type=float, default=None, help="Explicit spike-rate threshold in pixels")
    parser.add_argument(
        "--spike-kappa-quantile",
        type=float,
        default=0.99,
        help="Reference second-difference quantile used when --spike-kappa is not set",
    )
    parser.add_argument("--acc-tolerances", type=float, nargs="*", default=list(DEFAULT_ACC_TOLERANCES))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--warmup-batches", type=int, default=0, help="Skip these batches when timing inference")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--corruption", choices=list(CORRUPTION_TYPES), default="clean")
    parser.add_argument("--severity", choices=list(CORRUPTION_SEVERITIES), default="medium")
    parser.add_argument("--corruption-seed", type=int, default=0)
    parser.add_argument("--write-overlays", action="store_true", help="Write raw-image overlays with prediction and reference curves")
    parser.add_argument("--overlay-limit", type=int, default=100, help="Maximum number of overlay images to write")
    parser.add_argument("--model-type", choices=["auto", "dinoct", *LEARNED_BASELINE_MODELS], default="auto")
    parser.add_argument("--backbone", default="auto", help="Override backbone if checkpoint inference fails")
    parser.add_argument("--patch-size", type=int, default=14, help="Fallback patch size if checkpoint inference fails")
    parser.add_argument("--lora-blocks", type=int, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.add_argument("--lora-use-mlp", action="store_true")
    return parser.parse_args()


def _parse_dataset_path(dataset_str: str) -> tuple[str, dict[str, str]]:
    parts = dataset_str.split(":")
    tokens: dict[str, str] = {}
    for token in parts[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        tokens[key] = value
    return parts[0], tokens


def _format_dataset_path(name: str, tokens: dict[str, str]) -> str:
    return ":".join([name] + [f"{key}={value}" for key, value in tokens.items()])


def _with_split(dataset_str: str, split: str) -> str:
    name, tokens = _parse_dataset_path(dataset_str)
    tokens["split"] = split
    return _format_dataset_path(name, tokens)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _pick_device(device_arg: str) -> torch.device:
    want = str(device_arg).strip().lower()
    if want == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # pragma: no cover
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(want)


def _infer_backbone_spec_from_state_dict(state: dict[str, Any]) -> tuple[str | None, int | None]:
    keys = list(state.keys())
    if any(k.startswith("backbone.stages.") or k.startswith("backbone.downsample_layers.") for k in keys):
        stage2_re = re.compile(r"^backbone\.stages\.2\.(\d+)\.")
        stage2_idx: list[int] = []
        for key in keys:
            match = stage2_re.match(key)
            if match:
                stage2_idx.append(int(match.group(1)))
        if stage2_idx:
            depth2 = max(stage2_idx) + 1
            return ("convnext_small" if depth2 >= 27 else "convnext_tiny"), None
        return "convnext_tiny", None

    embed_to_name = {384: "small"}

    def _try_weight(value: Any) -> tuple[str | None, int | None]:
        if not isinstance(value, torch.Tensor) or value.ndim != 4:
            return None, None
        patch = int(value.shape[-1])
        embed_dim = int(value.shape[0])
        return embed_to_name.get(embed_dim), patch

    for key in ("backbone.patch_embed.proj.weight", "patch_embed.proj.weight"):
        name, patch = _try_weight(state.get(key))
        if patch is not None:
            return name, patch

    for key, value in state.items():
        if key.endswith("patch_embed.proj.weight"):
            name, patch = _try_weight(value)
            if patch is not None:
                return name, patch

    return None, None


def _default_curve_ckpt() -> Path:
    best = REPO_ROOT / "outputs" / "post_train" / "fused_curve_best.pth"
    final = REPO_ROOT / "outputs" / "post_train" / "fused_curve.pth"
    if best.exists():
        return best
    if final.exists():
        return final
    raise FileNotFoundError(
        "Could not find a default curve checkpoint. Pass --curve-ckpt outputs/.../post_train/fused_curve_best.pth"
    )


def _load_curve_model(
    *,
    ckpt_path: Path,
    device: torch.device,
    model_type: str,
    backbone_name: str,
    patch_size: int,
    lora_blocks: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_use_mlp: bool,
) -> tuple[torch.nn.Module, str]:
    ckpt = _torch_load(ckpt_path)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported curve checkpoint format: {type(state)}")

    inferred_model_type = infer_model_type_from_checkpoint(ckpt if isinstance(ckpt, dict) else {})
    resolved_model_type = inferred_model_type or (None if model_type == "auto" else str(model_type).strip().lower())
    if resolved_model_type in LEARNED_BASELINE_MODELS:
        model_kwargs = ckpt.get("model_kwargs", {}) if isinstance(ckpt, dict) else {}
        if not isinstance(model_kwargs, dict):
            model_kwargs = {}
        model = build_learned_baseline_model(resolved_model_type, **model_kwargs).to(device)
        model.load_state_dict(state, strict=False)
        model.eval()
        return model, resolved_model_type

    inferred_name, inferred_patch = _infer_backbone_spec_from_state_dict(state)
    model_backbone = inferred_name or (None if backbone_name == "auto" else backbone_name)
    if model_backbone is None:
        raise ValueError("Could not infer backbone from checkpoint; pass --backbone explicitly.")
    model_patch = int(inferred_patch or patch_size)

    backbone = build_backbone(model_backbone, patch_size=model_patch, device=device)
    model = CurveModel(
        backbone=backbone,
        patch_size=model_patch,
        lora_cfg={
            "blocks": lora_blocks,
            "r": lora_r,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "use_mlp": lora_use_mlp,
        },
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, "dinoct"


class IndexedDataset(Dataset):
    def __init__(
        self,
        base: OCT,
        *,
        corruption: str,
        severity: str,
        corruption_seed: int,
    ) -> None:
        self.base = base
        self.entries = base._get_entries()
        self.root = Path(base.root)
        self.transform = _make_oct_transform()
        self.corruption = str(corruption)
        self.severity = str(severity)
        self.corruption_seed = int(corruption_seed)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor, np.ndarray | None]:
        entry = self.entries[int(index)]
        image = np.asarray(Image.open(self.root / str(entry["filename"])).convert("L"), dtype=np.uint8)
        image = apply_oct_corruption(
            image,
            corruption=self.corruption,
            severity=self.severity,
            sample_key=str(entry["filename"]),
            seed=self.corruption_seed,
        )
        image_t = self.transform(Image.fromarray(image, mode="L"))
        target = self.base.get_target(int(index))
        if target is not None:
            target = np.asarray(target, dtype=np.float32)
        return int(index), image_t, target


def _collate_eval(batch: list[tuple[int, torch.Tensor, np.ndarray | None]]) -> dict[str, torch.Tensor]:
    indices: list[int] = []
    images: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    is_bgs: list[int] = []
    for index, image, target in batch:
        indices.append(int(index))
        images.append(image)
        if target is None:
            is_bgs.append(1)
            ys.append(torch.zeros(ORIG_W, dtype=torch.float32))
        else:
            target_t = torch.from_numpy(np.asarray(target, dtype=np.float32))
            is_bgs.append(0 if target_t.sum() != 0 else 1)
            ys.append(target_t)
    return {
        "indices": torch.tensor(indices, dtype=torch.long),
        "image": torch.stack(images, dim=0),
        "y": torch.stack(ys, dim=0),
        "is_bg": torch.tensor(is_bgs, dtype=torch.long),
    }


def _read_split_rows(extra_root: Path) -> dict[str, dict[str, str]]:
    splits_path = extra_root / "splits.csv"
    if not splits_path.exists():
        return {}
    with splits_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        out: dict[str, dict[str, str]] = {}
        for row in reader:
            group_id = str(row.get("group_id", "") or "").strip()
            if not group_id:
                continue
            out[group_id] = {
                "recording_id": str(row.get("recording_id", "") or group_id).strip() or group_id,
                "split": str(row.get("split", "") or "").strip(),
                "acquisition_mode": str(row.get("acquisition_mode", "") or "").strip(),
            }
        return out


def _estimate_spike_kappa(dataset_str: str, split: str, quantile: float) -> float:
    dataset = make_dataset(dataset_str=_with_split(dataset_str, split), transform=None)
    if not isinstance(dataset, OCT):
        raise TypeError(f"Expected OCT dataset, got {type(dataset)}")
    entries = dataset._get_entries()
    curves: list[np.ndarray] = []
    for idx, entry in enumerate(entries):
        if int(entry["code"]) != 1:
            continue
        target = dataset.get_target(int(idx))
        if target is None:
            continue
        curves.append(np.asarray(target, dtype=np.float32))
    if not curves:
        raise ValueError(f"No labeled curves found in split={split!r} for spike-kappa estimation.")
    return estimate_spike_kappa_from_curves(curves, quantile=quantile)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _load_gray(root: Path, relpath: str) -> np.ndarray:
    path = root / relpath
    if not path.exists():
        raise FileNotFoundError(path)
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def _draw_curve_overlay(
    raw: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    path: Path,
    *,
    pred_color: tuple[int, int, int] = (255, 64, 64),
    true_color: tuple[int, int, int] = (64, 255, 64),
) -> None:
    image = Image.fromarray(np.asarray(raw, dtype=np.uint8), mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)

    def _curve_points(curve: np.ndarray) -> list[tuple[int, int]]:
        pts: list[tuple[int, int]] = []
        for x, y in enumerate(np.asarray(curve, dtype=np.float32)):
            yi = int(round(float(y)))
            yi = max(0, min(image.height - 1, yi))
            pts.append((int(x), yi))
        return pts

    true_pts = _curve_points(y_true)
    pred_pts = _curve_points(y_pred)
    if len(true_pts) >= 2:
        draw.line(true_pts, fill=true_color, width=2)
    if len(pred_pts) >= 2:
        draw.line(pred_pts, fill=pred_color, width=2)

    image.save(path)


def main() -> None:
    args = parse_args()
    cfg = load_training_cfg(args.config)
    post_cfg = cfg.get("post_train", {})
    dataset_str = resolve_dataset_path(str(get_cfg(cfg, ("train", "dataset_path"), "OCT:root=data/oct:extra=data/oct/extra")))
    eval_split_name = "real_hard" if args.eval_dir is not None else str(args.split)
    if args.eval_dir is None:
        dataset_eval = make_dataset(dataset_str=_with_split(dataset_str, args.split), transform=_make_oct_transform())
        if not isinstance(dataset_eval, OCT):
            raise TypeError(f"Expected OCT dataset, got {type(dataset_eval)}")
        split_rows: dict[str, dict[str, str]]
    else:
        dataset_eval = DirectoryCurveEvalDataset(DirectoryEvalConfig(eval_dir=Path(args.eval_dir), split_name=eval_split_name))
    entries = dataset_eval._get_entries()

    dataset_name, dataset_tokens = _parse_dataset_path(dataset_str)
    del dataset_name
    extra_root = Path(dataset_tokens.get("extra", str(Path(dataset_tokens["root"]) / "extra")))
    split_rows = _read_split_rows(extra_root) if args.eval_dir is None else split_rows_for_directory_dataset(dataset_eval)

    device = _pick_device(args.device)
    ckpt_path = args.curve_ckpt or _default_curve_ckpt()
    output_suffix = corruption_output_suffix(args.corruption, args.severity)
    output_dir = args.output_dir or (ckpt_path.parent / f"eval_{eval_split_name}{output_suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "overlays"
    if args.write_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    acc_tolerances = tuple(float(value) for value in (args.acc_tolerances or list(DEFAULT_ACC_TOLERANCES)))
    quality_metric_names = ["mae_px", "p95_px", "bias_px", "abs_bias_px"] + [
        metric_name_for_tolerance(tau) for tau in acc_tolerances
    ]
    if args.spike_kappa is None:
        spike_kappa = _estimate_spike_kappa(dataset_str, args.kappa_split, float(args.spike_kappa_quantile))
        kappa_source = f"reference_quantile:{args.kappa_split}"
    else:
        spike_kappa = float(args.spike_kappa)
        kappa_source = "explicit"
    quality_metric_names.append("spike_rate")
    all_metric_names = [*quality_metric_names, "runtime_ms"]

    model, resolved_model_type = _load_curve_model(
        ckpt_path=ckpt_path,
        device=device,
        model_type=str(args.model_type),
        backbone_name=str(args.backbone),
        patch_size=int(args.patch_size),
        lora_blocks=int(args.lora_blocks or post_cfg.get("lora_blocks", 3)),
        lora_r=int(args.lora_r or post_cfg.get("lora_r", 8)),
        lora_alpha=int(args.lora_alpha or post_cfg.get("lora_alpha", 16)),
        lora_dropout=float(args.lora_dropout or post_cfg.get("lora_dropout", 0.05)),
        lora_use_mlp=bool(args.lora_use_mlp or post_cfg.get("lora_use_mlp", False)),
    )

    data_loader = DataLoader(
        IndexedDataset(
            dataset_eval,
            corruption=str(args.corruption),
            severity=str(args.severity),
            corruption_seed=int(args.corruption_seed),
        ),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=_collate_eval,
    )

    sample_rows: list[dict[str, Any]] = []
    overlays_written = 0
    root = Path(dataset_eval.root)
    model.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(data_loader):
            images = batch["image"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            if is_segmentation_baseline_model_type(resolved_model_type):
                seg_logits = model(images, orig_hw=(ORIG_H, ORIG_W))
                _presence_logits = None
                curve_logits = None
                y_hat, _presence_logits = decode_paper_unet_logits(seg_logits)
            else:
                _presence_logits, curve_logits = model(images)
                y_hat = soft_argmax_height(curve_logits[:, :-1, :])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            runtime_per_sample = (
                elapsed_ms / max(int(images.shape[0]), 1) if batch_idx >= int(args.warmup_batches) else float("nan")
            )

            batch_metrics = curve_metrics_batch(y_hat, y, acc_tolerances=acc_tolerances, spike_kappa=spike_kappa)
            batch_indices = batch["indices"].tolist()
            for pos, dataset_idx in enumerate(batch_indices):
                entry = entries[int(dataset_idx)]
                if int(entry["code"]) != 1:
                    continue
                group_id = str(entry["group_id"])
                split_meta = split_rows.get(group_id, {})
                row: dict[str, Any] = {
                    "sample_id": str(entry["sample_id"]) if "sample_id" in entry.dtype.names else str(entry["filename"]),
                    "filename": str(entry["filename"]),
                    "stem": Path(str(entry["filename"])).stem,
                    "group_id": group_id,
                    "recording_id": split_meta.get("recording_id", group_id),
                    "acquisition_mode": split_meta.get("acquisition_mode", ""),
                    "split": split_meta.get("split", str(entry["split"]) if "split" in entry.dtype.names else eval_split_name),
                    "corruption": str(args.corruption),
                    "severity": str(args.severity),
                    "runtime_ms": runtime_per_sample,
                }
                for metric_name, metric_values in batch_metrics.items():
                    row[metric_name] = float(metric_values[pos].detach().cpu().item())
                sample_rows.append(row)

                if args.write_overlays and overlays_written < int(args.overlay_limit):
                    filename = str(entry["filename"])
                    raw = _load_gray(root, filename)
                    raw = apply_oct_corruption(
                        raw,
                        corruption=str(args.corruption),
                        severity=str(args.severity),
                        sample_key=filename,
                        seed=int(args.corruption_seed),
                    )
                    _draw_curve_overlay(
                        raw,
                        y_hat[pos].detach().cpu().numpy(),
                        y[pos].detach().cpu().numpy(),
                        overlay_dir / f"{Path(filename).stem}_overlay.png",
                    )
                    overlays_written += 1

    if not sample_rows:
        raise ValueError(f"No labeled samples found for split={args.split!r}.")

    recording_rows = average_metric_rows(sample_rows, group_key="recording_id", metric_names=all_metric_names)

    per_scan_path = output_dir / "per_scan_metrics.csv"
    per_recording_path = output_dir / "per_recording_metrics.csv"
    summary_path = output_dir / "summary.json"

    _write_csv(
        per_scan_path,
        sample_rows,
        [
            "sample_id",
            "filename",
            "stem",
            "group_id",
            "recording_id",
            "acquisition_mode",
            "split",
            "corruption",
            "severity",
            *all_metric_names,
        ],
    )
    _write_csv(
        per_recording_path,
        recording_rows,
        ["recording_id", "acquisition_mode", "split", "num_samples", *all_metric_names],
    )

    per_scan_summary = summarize_metric_rows(sample_rows, all_metric_names)
    per_recording_summary = summarize_metric_rows(recording_rows, all_metric_names)
    table_metrics_mean = {
        "mae_px": per_recording_summary["mae_px"]["mean"],
        "p95_px": per_recording_summary["p95_px"]["mean"],
        "bias_px": per_recording_summary["bias_px"]["mean"],
        "abs_bias_px": per_recording_summary["abs_bias_px"]["mean"],
        "acc_2px": per_recording_summary.get("acc_2px", {}).get("mean", float("nan")),
        "acc_4px": per_recording_summary.get("acc_4px", {}).get("mean", float("nan")),
        "spike_rate": per_recording_summary["spike_rate"]["mean"],
        "runtime_ms": per_scan_summary["runtime_ms"]["mean"],
    }
    table_metrics_std = {
        "mae_px": per_recording_summary["mae_px"]["std"],
        "p95_px": per_recording_summary["p95_px"]["std"],
        "bias_px": per_recording_summary["bias_px"]["std"],
        "abs_bias_px": per_recording_summary["abs_bias_px"]["std"],
        "acc_2px": per_recording_summary.get("acc_2px", {}).get("std", float("nan")),
        "acc_4px": per_recording_summary.get("acc_4px", {}).get("std", float("nan")),
        "spike_rate": per_recording_summary["spike_rate"]["std"],
        "runtime_ms": per_scan_summary["runtime_ms"]["std"],
    }

    summary = {
        "checkpoint": str(ckpt_path),
        "model_type": resolved_model_type,
        "dataset_path": str(args.eval_dir) if args.eval_dir is not None else _with_split(dataset_str, args.split),
        "split": eval_split_name,
        "corruption": str(args.corruption),
        "severity": str(args.severity),
        "acc_tolerances_px": list(acc_tolerances),
        "spike_kappa": float(spike_kappa),
        "spike_kappa_source": kappa_source,
        "counts": {
            "labeled_bscans": len(sample_rows),
            "recordings": len(recording_rows),
            "timed_bscans": int(per_scan_summary["runtime_ms"]["count"]),
            "overlay_images_written": int(overlays_written),
        },
        "table_metrics_per_recording_mean": table_metrics_mean,
        "table_metrics_per_recording_std": table_metrics_std,
        "per_scan": per_scan_summary,
        "per_recording": per_recording_summary,
        "runtime_ms_per_bscan": per_scan_summary["runtime_ms"],
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(
        f"[eval] model={resolved_model_type} split={eval_split_name} corruption={args.corruption} severity={args.severity} "
        f"labeled={len(sample_rows)} recordings={len(recording_rows)} "
        f"mae={table_metrics_mean['mae_px']:.3f}+-{table_metrics_std['mae_px']:.3f} "
        f"p95={table_metrics_mean['p95_px']:.3f}+-{table_metrics_std['p95_px']:.3f} "
        f"bias={table_metrics_mean['bias_px']:.3f}+-{table_metrics_std['bias_px']:.3f} "
        f"abs_bias={table_metrics_mean['abs_bias_px']:.3f}+-{table_metrics_std['abs_bias_px']:.3f} "
        f"acc2={table_metrics_mean['acc_2px']:.3f}+-{table_metrics_std['acc_2px']:.3f} "
        f"acc4={table_metrics_mean['acc_4px']:.3f}+-{table_metrics_std['acc_4px']:.3f} "
        f"spike={table_metrics_mean['spike_rate']:.4f}+-{table_metrics_std['spike_rate']:.4f} "
        f"runtime_ms={table_metrics_mean['runtime_ms']:.3f}+-{table_metrics_std['runtime_ms']:.3f}"
    )
    print(f"[eval] wrote {per_scan_path}")
    print(f"[eval] wrote {per_recording_path}")
    print(f"[eval] wrote {summary_path}")


if __name__ == "__main__":
    main()
