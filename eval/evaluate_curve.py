#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
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
from dinoct.provenance import file_md5, git_state, runtime_versions  # noqa: E402
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
    infer_model_type_from_checkpoint,
)
from eval.corruptions import (  # noqa: E402
    CORRUPTION_SEVERITIES,
    CORRUPTION_TYPES,
    apply_oct_corruption,
    corruption_output_suffix,
)
from eval.evalset import DirectoryCurveEvalDataset, DirectoryEvalConfig, split_rows_for_directory_dataset  # noqa: E402
from dinoct.models import build_backbone, native_patch_size_for_backbone  # noqa: E402
from dinoct.train.post_train import (  # noqa: E402
    CurveModel,
    ORIG_H,
    ORIG_W,
    _make_oct_transform,
    remap_legacy_curve_head_keys,
    soft_argmax_height,
)
from dinoct.train.train import get_cfg, load_training_cfg, resolve_dataset_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate a fused OCT curve checkpoint on a dataset split.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "train" / "oct.yaml")
    parser.add_argument("--curve-ckpt", type=Path, default=None, help="Path to fused_curve_best.pth or fused_curve.pth")
    parser.add_argument(
        "--ensemble-curve-ckpt",
        type=Path,
        action="append",
        default=[],
        help="Additional fused curve checkpoint to average with --curve-ckpt. Can be passed multiple times.",
    )
    parser.add_argument(
        "--ensemble-weights",
        type=float,
        nargs="+",
        default=None,
        help="Optional weights for primary plus ensemble checkpoints; omitted uses equal weights.",
    )
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
    parser.add_argument(
        "--patch-size",
        type=int,
        default=None,
        help="Optional patch-size override; omitted uses checkpoint/native backbone metadata",
    )
    parser.add_argument("--lora-blocks", type=int, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.add_argument("--lora-use-mlp", action="store_true")
    parser.add_argument("--curve-head-mid", type=int, default=None)
    parser.add_argument("--feature-layers", type=int, default=None)
    parser.add_argument(
        "--tta-hflip",
        action="store_true",
        help="Average predictions with a horizontally flipped input, reversing columns back before scoring.",
    )
    parser.add_argument(
        "--curve-z-offset",
        type=float,
        default=0.0,
        help="Add a constant vertical pixel offset to predicted curves before scoring.",
    )
    parser.add_argument(
        "--curve-smooth-window",
        type=int,
        default=1,
        help="Odd-width moving-average smoothing window for predicted curves before scoring; 1 disables smoothing.",
    )
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
        stem_weight = state.get("backbone.downsample_layers.0.0.weight")
        stem_is_s2 = isinstance(stem_weight, torch.Tensor) and stem_weight.ndim == 4 and int(stem_weight.shape[-1]) == 2
        stage2_re = re.compile(r"^backbone\.stages\.2\.(\d+)\.")
        stage2_idx: list[int] = []
        for key in keys:
            match = stage2_re.match(key)
            if match:
                stage2_idx.append(int(match.group(1)))
        if stage2_idx:
            depth2 = max(stage2_idx) + 1
            if depth2 >= 27:
                return ("convnext_small_s2" if stem_is_s2 else "convnext_small"), None
            return ("convnext_tiny_s2" if stem_is_s2 else "convnext_tiny"), None
        return ("convnext_tiny_s2" if stem_is_s2 else "convnext_tiny"), None

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


def _infer_lora_spec_from_state_dict(
    state: dict[str, Any],
    *,
    default_blocks: int,
    default_r: int,
    default_use_mlp: bool,
) -> tuple[int, int, bool]:
    lora_a_keys = [key for key, value in state.items() if key.endswith(".lora_A") and torch.is_tensor(value)]
    if not lora_a_keys:
        # No LoRA adapter keys => the checkpoint is a frozen/fused model with ZERO LoRA blocks.
        # Returning default_blocks (e.g. 3) here was a bug: it rebuilt a LoRA-wrapped model whose
        # last blocks' base weights then failed to load (renamed) and stayed RANDOM-INIT, silently
        # corrupting every frozen eval. Absence of lora_A is positive evidence of 0 blocks.
        return 0, int(default_r), bool(default_use_mlp)

    first = state[lora_a_keys[0]]
    inferred_r = int(first.shape[0]) if torch.is_tensor(first) and first.ndim == 2 else int(default_r)
    convnext_blocks: set[str] = set()
    vit_blocks: set[str] = set()
    inferred_use_mlp = bool(default_use_mlp)
    convnext_re = re.compile(r"^(backbone\.stages\.\d+\.\d+)\.")
    vit_re = re.compile(r"^(backbone\.blocks(?:\.\d+)?\.\d+)\.")
    for key in lora_a_keys:
        convnext_match = convnext_re.match(key)
        if convnext_match:
            convnext_blocks.add(convnext_match.group(1))
        vit_match = vit_re.match(key)
        if vit_match:
            vit_blocks.add(vit_match.group(1))
        if ".mlp." in key:
            inferred_use_mlp = True

    inferred_blocks = len(convnext_blocks or vit_blocks) or int(default_blocks)
    return int(inferred_blocks), int(inferred_r), bool(inferred_use_mlp)


def _infer_input_aa_strength_from_state_dict(state: dict[str, Any], default_strength: float) -> float:
    value = state.get("input_aa.aa_strength")
    if torch.is_tensor(value) and value.numel() == 1:
        return float(value.detach().cpu().item())
    return float(default_strength)


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
    patch_size: int | None,
    lora_blocks: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_use_mlp: bool,
    curve_head_mid: int,
    feature_layers: int,
    input_aa_strength: float,
) -> tuple[torch.nn.Module, str, dict[str, Any]]:
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
        state = remap_legacy_curve_head_keys(state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        _warn_state_mismatch(missing, unexpected)
        model.eval()
        return model, resolved_model_type, _load_info(
            missing=missing,
            unexpected=unexpected,
            patch_size=None,
            lora_blocks=None,
            lora_r=None,
            lora_alpha=None,
            lora_dropout=None,
            lora_use_mlp=None,
            curve_head_mid=None,
            feature_layers=None,
            input_aa_strength=None,
        )

    inferred_name, inferred_patch = _infer_backbone_spec_from_state_dict(state)
    model_backbone = inferred_name or (None if backbone_name == "auto" else backbone_name)
    if model_backbone is None:
        raise ValueError("Could not infer backbone from checkpoint; pass --backbone explicitly.")
    model_patch = int(
        inferred_patch
        if inferred_patch is not None
        else patch_size
        if patch_size is not None
        else native_patch_size_for_backbone(model_backbone)
    )

    backbone = build_backbone(model_backbone, patch_size=model_patch, device=device)
    base_channels = getattr(backbone, "embed_dim", backbone.num_features if hasattr(backbone, "num_features") else 768)
    model_lora_blocks, model_lora_r, model_lora_use_mlp = _infer_lora_spec_from_state_dict(
        state,
        default_blocks=int(lora_blocks),
        default_r=int(lora_r),
        default_use_mlp=bool(lora_use_mlp),
    )
    model_input_aa_strength = _infer_input_aa_strength_from_state_dict(state, float(input_aa_strength))
    model_curve_head_mid = int(curve_head_mid)
    model_feature_layers = int(feature_layers)
    proj_weight = state.get("curve_head.proj.weight")
    if torch.is_tensor(proj_weight) and proj_weight.ndim >= 2:
        model_curve_head_mid = int(proj_weight.shape[0])
        inferred_in_channels = int(proj_weight.shape[1])
        if int(base_channels) > 0 and inferred_in_channels % int(base_channels) == 0:
            model_feature_layers = max(1, inferred_in_channels // int(base_channels))

    model = CurveModel(
        backbone=backbone,
        patch_size=model_patch,
        lora_cfg={
            "blocks": model_lora_blocks,
            "r": model_lora_r,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "use_mlp": model_lora_use_mlp,
        },
        curve_head_mid=model_curve_head_mid,
        feature_layers=model_feature_layers,
        input_aa_strength=model_input_aa_strength,
    ).to(device)
    state = remap_legacy_curve_head_keys(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    _warn_state_mismatch(missing, unexpected)
    # Guard: refuse to eval a backbone that didn't actually load (would be partly random-init).
    bb_keys = [k for k in model.state_dict() if k.startswith("backbone.")]
    bb_missing = [k for k in missing if k.startswith("backbone.")]
    if bb_keys and len(bb_missing) > 0.05 * len(bb_keys):
        raise RuntimeError(
            f"evaluate_curve: {len(bb_missing)}/{len(bb_keys)} backbone params did NOT load from the "
            f"checkpoint (missing>5%) — model would be partly random-init (e.g. lora_blocks mismatch). "
            f"first missing: {bb_missing[0]}. Refusing to eval a corrupted backbone."
        )
    model.eval()
    return model, "dinoct", _load_info(
        missing=missing,
        unexpected=unexpected,
        patch_size=model_patch,
        lora_blocks=model_lora_blocks,
        lora_r=model_lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_use_mlp=model_lora_use_mlp,
        curve_head_mid=model_curve_head_mid,
        feature_layers=model_feature_layers,
        input_aa_strength=model_input_aa_strength,
    )


def _load_info(
    *,
    missing: list[str],
    unexpected: list[str],
    patch_size: int | None,
    lora_blocks: int | None,
    lora_r: int | None,
    lora_alpha: int | None,
    lora_dropout: float | None,
    lora_use_mlp: bool | None,
    curve_head_mid: int | None,
    feature_layers: int | None,
    input_aa_strength: float | None,
) -> dict[str, Any]:
    return {
        "missing_key_count": len(missing),
        "unexpected_key_count": len(unexpected),
        "first_missing_key": missing[0] if missing else None,
        "first_unexpected_key": unexpected[0] if unexpected else None,
        "patch_size": patch_size,
        "lora_blocks": lora_blocks,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_use_mlp": lora_use_mlp,
        "curve_head_mid": curve_head_mid,
        "feature_layers": feature_layers,
        "input_aa_strength": input_aa_strength,
    }


def _warn_state_mismatch(missing: list[str], unexpected: list[str]) -> None:
    if not missing and not unexpected:
        return
    print(
        "[eval] warning: checkpoint/model key mismatch: "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    if missing:
        print(f"[eval] warning: first missing key: {missing[0]}")
    if unexpected:
        print(f"[eval] warning: first unexpected key: {unexpected[0]}")


def _predict_curve_z_hat(
    model: torch.nn.Module,
    resolved_model_type: str,
    images: torch.Tensor,
    *,
    tta_hflip: bool,
) -> torch.Tensor:
    _, curve_logits = model(images)
    z_hat = soft_argmax_height(curve_logits[:, :-1, :])
    if tta_hflip:
        _, curve_logits_flip = model(images.flip(dims=(-1,)))
        z_hat_flip = soft_argmax_height(curve_logits_flip[:, :-1, :])
        z_hat = 0.5 * (z_hat + z_hat_flip.flip(dims=(-1,)))
    return z_hat


def _smooth_curve_z_hat(z_hat: torch.Tensor, window: int) -> torch.Tensor:
    window_i = int(window)
    if window_i <= 1:
        return z_hat
    if window_i % 2 == 0:
        raise ValueError("--curve-smooth-window must be odd when greater than 1")
    pad = window_i // 2
    z = z_hat.float().unsqueeze(1)
    z = F.pad(z, (pad, pad), mode="replicate")
    z = F.avg_pool1d(z, kernel_size=window_i, stride=1)
    return z.squeeze(1).to(dtype=z_hat.dtype)


def _arg_or_cfg(value, cfg: dict, key: str, default):
    return value if value is not None else cfg.get(key, default)


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

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor, np.ndarray | None, int]:
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
        return int(index), image_t, target, int(entry["code"])


def _collate_eval(batch: list[tuple[int, torch.Tensor, np.ndarray | None, int]]) -> dict[str, torch.Tensor]:
    indices: list[int] = []
    images: list[torch.Tensor] = []
    zs: list[torch.Tensor] = []
    is_bgs: list[int] = []
    for index, image, target, code in batch:
        indices.append(int(index))
        images.append(image)
        is_bg = int(code) == 2
        if target is None:
            is_bgs.append(1)
            zs.append(torch.zeros(ORIG_W, dtype=torch.float32))
        else:
            target_t = torch.from_numpy(np.asarray(target, dtype=np.float32))
            is_bgs.append(1 if is_bg else 0)
            zs.append(target_t)
    return {
        "indices": torch.tensor(indices, dtype=torch.long),
        "image": torch.stack(images, dim=0),
        "z": torch.stack(zs, dim=0),
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
    z_pred: np.ndarray,
    z_true: np.ndarray,
    path: Path,
    *,
    pred_color: tuple[int, int, int] = (255, 64, 64),
    true_color: tuple[int, int, int] = (64, 255, 64),
) -> None:
    image = Image.fromarray(np.asarray(raw, dtype=np.uint8), mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)

    def _curve_points(curve: np.ndarray) -> list[tuple[int, int]]:
        pts: list[tuple[int, int]] = []
        for x, z in enumerate(np.asarray(curve, dtype=np.float32)):
            zi = int(round(float(z)))
            zi = max(0, min(image.height - 1, zi))
            pts.append((int(x), zi))
        return pts

    true_pts = _curve_points(z_true)
    pred_pts = _curve_points(z_pred)
    if len(true_pts) >= 2:
        draw.line(true_pts, fill=true_color, width=2)
    if len(pred_pts) >= 2:
        draw.line(pred_pts, fill=pred_color, width=2)

    image.save(path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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
    if args.eval_dir is None and isinstance(dataset_eval, OCT):
        extra_root = Path(dataset_eval._extra_root)
    else:
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

    effective_lora_blocks = int(_arg_or_cfg(args.lora_blocks, post_cfg, "lora_blocks", 3))
    effective_lora_r = int(_arg_or_cfg(args.lora_r, post_cfg, "lora_r", 8))
    effective_lora_alpha = int(_arg_or_cfg(args.lora_alpha, post_cfg, "lora_alpha", 16))
    effective_lora_dropout = float(_arg_or_cfg(args.lora_dropout, post_cfg, "lora_dropout", 0.05))
    effective_lora_use_mlp = bool(args.lora_use_mlp or post_cfg.get("lora_use_mlp", False))
    effective_curve_head_mid = int(_arg_or_cfg(args.curve_head_mid, post_cfg, "curve_head_mid", 128))
    effective_feature_layers = int(_arg_or_cfg(args.feature_layers, post_cfg, "feature_layers", 1))
    effective_input_aa_strength = float(post_cfg.get("input_aa_strength", 0.0))
    configured_patch_size = get_cfg(cfg, ("student", "patch_size"), None)
    effective_patch_size = (
        int(args.patch_size)
        if args.patch_size is not None
        else int(configured_patch_size)
        if configured_patch_size is not None
        else None
    )

    model, resolved_model_type, model_load = _load_curve_model(
        ckpt_path=ckpt_path,
        device=device,
        model_type=str(args.model_type),
        backbone_name=str(args.backbone),
        patch_size=effective_patch_size,
        lora_blocks=effective_lora_blocks,
        lora_r=effective_lora_r,
        lora_alpha=effective_lora_alpha,
        lora_dropout=effective_lora_dropout,
        lora_use_mlp=effective_lora_use_mlp,
        curve_head_mid=effective_curve_head_mid,
        feature_layers=effective_feature_layers,
        input_aa_strength=effective_input_aa_strength,
    )
    model_entries: list[tuple[torch.nn.Module, str, Path]] = [(model, resolved_model_type, ckpt_path)]
    ensemble_model_loads: list[dict[str, Any]] = []
    for extra_ckpt in args.ensemble_curve_ckpt or []:
        extra_model, extra_model_type, extra_model_load = _load_curve_model(
            ckpt_path=Path(extra_ckpt),
            device=device,
            model_type=str(args.model_type),
            backbone_name=str(args.backbone),
            patch_size=effective_patch_size,
            lora_blocks=effective_lora_blocks,
            lora_r=effective_lora_r,
            lora_alpha=effective_lora_alpha,
            lora_dropout=effective_lora_dropout,
            lora_use_mlp=effective_lora_use_mlp,
            curve_head_mid=effective_curve_head_mid,
            feature_layers=effective_feature_layers,
            input_aa_strength=effective_input_aa_strength,
        )
        model_entries.append((extra_model, extra_model_type, Path(extra_ckpt)))
        ensemble_model_loads.append({"checkpoint": str(extra_ckpt), "model_type": extra_model_type, **extra_model_load})
    if args.ensemble_weights is None:
        ensemble_weights = [1.0 / float(len(model_entries))] * len(model_entries)
    else:
        if len(args.ensemble_weights) != len(model_entries):
            raise ValueError(
                "--ensemble-weights must provide one weight for the primary --curve-ckpt plus each "
                "--ensemble-curve-ckpt"
            )
        raw_weights = [float(value) for value in args.ensemble_weights]
        weight_sum = float(sum(raw_weights))
        if weight_sum <= 0.0:
            raise ValueError("--ensemble-weights must sum to a positive value")
        ensemble_weights = [float(value) / weight_sum for value in raw_weights]

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
            z = batch["z"].to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            z_hat = None
            for weight, (entry_model, entry_model_type, _entry_ckpt) in zip(ensemble_weights, model_entries):
                pred = _predict_curve_z_hat(entry_model, entry_model_type, images, tta_hflip=bool(args.tta_hflip))
                z_hat = pred.mul(float(weight)) if z_hat is None else z_hat.add(pred, alpha=float(weight))
            if z_hat is None:
                raise RuntimeError("No curve model predictions were produced.")
            if float(args.curve_z_offset) != 0.0:
                z_hat = (z_hat + float(args.curve_z_offset)).clamp_(0.0, float(ORIG_H - 1))
            if int(args.curve_smooth_window) > 1:
                z_hat = _smooth_curve_z_hat(z_hat, int(args.curve_smooth_window)).clamp_(0.0, float(ORIG_H - 1))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            runtime_per_sample = (
                elapsed_ms / max(int(images.shape[0]), 1) if batch_idx >= int(args.warmup_batches) else float("nan")
            )

            batch_metrics = curve_metrics_batch(z_hat, z, acc_tolerances=acc_tolerances, spike_kappa=spike_kappa)
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
                        z_hat[pos].detach().cpu().numpy(),
                        z[pos].detach().cpu().numpy(),
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

    def _repo_rel(path: Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(REPO_ROOT))
        except ValueError:
            return str(path)

    provenance = {
        "git": git_state(REPO_ROOT),
        "versions": runtime_versions(),
        "checkpoint_md5": {_repo_rel(path): file_md5(path) for _, _, path in model_entries},
        "manifest_md5": file_md5(extra_root / "manifest.csv") if args.eval_dir is None else None,
        "splits_md5": file_md5(extra_root / "splits.csv") if args.eval_dir is None else None,
        "eval_dir": str(args.eval_dir) if args.eval_dir is not None else None,
    }

    summary = {
        "checkpoint": str(ckpt_path),
        "model_type": resolved_model_type,
        "dataset_path": str(args.eval_dir) if args.eval_dir is not None else _with_split(dataset_str, args.split),
        "split": eval_split_name,
        "corruption": str(args.corruption),
        "severity": str(args.severity),
        "corruption_seed": int(args.corruption_seed),
        "model_load": model_load,
        "ensemble_curve_checkpoints": [str(path) for _, _, path in model_entries],
        "ensemble_weights": [float(weight) for weight in ensemble_weights],
        "ensemble_model_loads": ensemble_model_loads,
        "tta_hflip": bool(args.tta_hflip),
        "curve_z_offset": float(args.curve_z_offset),
        "curve_smooth_window": int(args.curve_smooth_window),
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
        "provenance": provenance,
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
