#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def _find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").exists() and (candidate / "dinoct").is_dir():
            return candidate
    raise RuntimeError("Could not locate repo root from script path.")


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.classical import (  # noqa: E402
    AVAILABLE_METHODS,
    CLASSICAL_METHODS,
    detect_surface_curve,
    resolve_method_key,
)
from eval.corruptions import (  # noqa: E402
    CORRUPTION_SEVERITIES,
    CORRUPTION_TYPES,
    apply_oct_corruption,
    corruption_output_suffix,
)
from eval.evalset import DirectoryCurveEvalDataset, DirectoryEvalConfig, split_rows_for_directory_dataset  # noqa: E402
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
from dinoct.train.train import get_cfg, load_training_cfg, resolve_dataset_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate classical OCT baselines on a dataset split.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "train" / "oct.yaml")
    parser.add_argument("--method", choices=[*AVAILABLE_METHODS, "all"], default="all")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--eval-dir", type=Path, default=None, help="Optional directory of .jpg/.txt pairs for a separate eval subset")
    parser.add_argument("--kappa-split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--spike-kappa", type=float, default=None)
    parser.add_argument("--spike-kappa-quantile", type=float, default=0.99)
    parser.add_argument("--acc-tolerances", type=float, nargs="*", default=list(DEFAULT_ACC_TOLERANCES))
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of labeled samples to evaluate")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--corruption", choices=list(CORRUPTION_TYPES), default="clean")
    parser.add_argument("--severity", choices=list(CORRUPTION_SEVERITIES), default="medium")
    parser.add_argument("--corruption-seed", type=int, default=0)
    parser.add_argument(
        "--fallback-background",
        type=Path,
        default=None,
        help="Fallback background image used by gf_b when a sample has no matched background",
    )
    parser.add_argument("--write-overlays", action="store_true", help="Write raw-image overlays with prediction and reference curves")
    parser.add_argument("--overlay-limit", type=int, default=100, help="Maximum number of overlay images to write per method")
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


def _estimate_spike_kappa(dataset: OCT, quantile: float) -> float:
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
        raise ValueError("No labeled curves found for spike-kappa estimation.")
    return estimate_spike_kappa_from_curves(curves, quantile=float(quantile))


def _load_gray(root: Path, relpath: str | Path) -> np.ndarray:
    path = root / relpath
    if not path.exists():
        raise FileNotFoundError(path)
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


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


def _evaluate_method(
    *,
    method: str,
    dataset: Any,
    split_rows: dict[str, dict[str, str]],
    output_dir: Path,
    acc_tolerances: tuple[float, ...],
    spike_kappa: float,
    limit: int | None,
    fallback_background: np.ndarray | None,
    corruption: str,
    severity: str,
    corruption_seed: int,
    write_overlays: bool,
    overlay_limit: int,
) -> dict[str, Any]:
    method_key = resolve_method_key(method)
    method_spec = CLASSICAL_METHODS[method_key]
    entries = dataset._get_entries()
    root = Path(dataset.root)
    out_dir = output_dir / method_key
    out_dir.mkdir(parents=True, exist_ok=True)
    per_scan_path = out_dir / "per_scan_metrics.csv"
    per_recording_path = out_dir / "per_recording_metrics.csv"
    summary_path = out_dir / "summary.json"
    overlay_dir = out_dir / "overlays"
    if write_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    quality_metric_names = ["mae_px", "p95_px", "bias_px", "abs_bias_px"] + [
        metric_name_for_tolerance(tau) for tau in acc_tolerances
    ] + ["spike_rate"]
    runtime_metric_names = ["runtime_ms"]
    all_metric_names = [*quality_metric_names, *runtime_metric_names]

    sample_rows: list[dict[str, Any]] = []
    overlays_written = 0
    counts = {
        "labeled_bscans_total": 0,
        "evaluated_bscans": 0,
        "skipped_missing_background": 0,
        "used_fallback_background": 0,
    }

    for idx, entry in enumerate(entries):
        if int(entry["code"]) != 1:
            continue
        counts["labeled_bscans_total"] += 1
        if limit is not None and counts["evaluated_bscans"] >= int(limit):
            break

        filename = str(entry["filename"])
        raw = _load_gray(root, filename)
        raw = apply_oct_corruption(
            raw,
            corruption=corruption,
            severity=severity,
            sample_key=filename,
            seed=int(corruption_seed),
        )
        background_relpath = str(entry["background_relpath"]) if "background_relpath" in entry.dtype.names else ""
        background = _load_gray(root, background_relpath) if background_relpath else None
        used_fallback_background = False
        if method_spec.requires_background and background is None and fallback_background is not None:
            background = fallback_background
            used_fallback_background = True
        if method_spec.requires_background and background is None:
            counts["skipped_missing_background"] += 1
            continue

        y_true = dataset.get_target(int(idx))
        if y_true is None:
            continue
        t0 = time.perf_counter()
        y_pred = detect_surface_curve(method_key, raw, background=background)
        runtime_ms = (time.perf_counter() - t0) * 1000.0
        metrics = curve_metrics_batch(
            np.asarray(y_pred[None, :], dtype=np.float32),
            np.asarray(y_true, dtype=np.float32)[None, :],
            acc_tolerances=acc_tolerances,
            spike_kappa=spike_kappa,
        )

        group_id = str(entry["group_id"])
        split_meta = split_rows.get(group_id, {})
        row: dict[str, Any] = {
            "sample_id": str(entry["sample_id"]) if "sample_id" in entry.dtype.names else filename,
            "filename": filename,
            "stem": Path(filename).stem,
            "group_id": group_id,
            "recording_id": split_meta.get("recording_id", group_id),
            "acquisition_mode": split_meta.get("acquisition_mode", ""),
            "split": split_meta.get("split", str(entry["split"]) if "split" in entry.dtype.names else ""),
            "corruption": str(corruption),
            "severity": str(severity),
            "used_background": int(background is not None),
            "used_fallback_background": int(used_fallback_background),
        }
        for metric_name, metric_value in metrics.items():
            row[metric_name] = float(np.asarray(metric_value, dtype=np.float32)[0])
        row["runtime_ms"] = float(runtime_ms)
        sample_rows.append(row)
        counts["evaluated_bscans"] += 1
        if used_fallback_background:
            counts["used_fallback_background"] += 1
        if write_overlays and overlays_written < max(int(overlay_limit), 0):
            overlay_path = overlay_dir / f"{Path(filename).stem}_overlay.png"
            _draw_curve_overlay(
                raw,
                np.asarray(y_pred, dtype=np.float32),
                np.asarray(y_true, dtype=np.float32),
                overlay_path,
            )
            overlays_written += 1

    if not sample_rows:
        summary = {
            "method": method_key,
            "display_name": method_spec.display_name,
            "description": method_spec.description,
            "requires_background": method_spec.requires_background,
            "split": str(getattr(dataset, "_split", "") or ""),
            "corruption": str(corruption),
            "severity": str(severity),
            "spike_kappa": float(spike_kappa),
            "acc_tolerances_px": list(acc_tolerances),
            "counts": {
                **counts,
                "recordings": 0,
                "evaluated_with_background": 0,
                "overlay_images_written": 0,
            },
            "skipped": True,
            "skip_reason": (
                "no labeled samples with usable background were available for this split"
                if method_spec.requires_background
                else "no evaluated labeled samples were available for this split"
            ),
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        print(
            f"[classical:{method_key}] skipped on split={getattr(dataset, '_split', '') or ''} "
            f"corruption={corruption} severity={severity}: "
            f"{summary['skip_reason']} (labeled_total={counts['labeled_bscans_total']}, "
            f"skipped_missing_background={counts['skipped_missing_background']})"
        )
        print(f"[classical:{method_key}] wrote {summary_path}")
        return summary

    recording_rows = average_metric_rows(sample_rows, group_key="recording_id", metric_names=all_metric_names)
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
        "runtime_ms": per_recording_summary["runtime_ms"]["mean"],
    }
    table_metrics_std = {
        "mae_px": per_recording_summary["mae_px"]["std"],
        "p95_px": per_recording_summary["p95_px"]["std"],
        "bias_px": per_recording_summary["bias_px"]["std"],
        "abs_bias_px": per_recording_summary["abs_bias_px"]["std"],
        "acc_2px": per_recording_summary.get("acc_2px", {}).get("std", float("nan")),
        "acc_4px": per_recording_summary.get("acc_4px", {}).get("std", float("nan")),
        "spike_rate": per_recording_summary["spike_rate"]["std"],
        "runtime_ms": per_recording_summary["runtime_ms"]["std"],
    }

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
            "used_background",
            "used_fallback_background",
            *all_metric_names,
        ],
    )
    _write_csv(
        per_recording_path,
        recording_rows,
        ["recording_id", "acquisition_mode", "split", "num_samples", *all_metric_names],
    )

    summary = {
        "method": method_key,
        "display_name": method_spec.display_name,
        "description": method_spec.description,
        "requires_background": method_spec.requires_background,
        "split": str(sample_rows[0].get("split", "")),
        "corruption": str(corruption),
        "severity": str(severity),
        "spike_kappa": float(spike_kappa),
        "acc_tolerances_px": list(acc_tolerances),
        "counts": {
            **counts,
            "recordings": len(recording_rows),
            "evaluated_with_background": int(sum(int(row["used_background"]) for row in sample_rows)),
            "overlay_images_written": overlays_written,
        },
        "table_metrics_per_recording_mean": table_metrics_mean,
        "table_metrics_per_recording_std": table_metrics_std,
        "runtime_ms_per_bscan": {
            "per_scan": per_scan_summary["runtime_ms"],
            "per_recording": per_recording_summary["runtime_ms"],
        },
        "per_scan": per_scan_summary,
        "per_recording": per_recording_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(
        f"[classical:{method_key}] split={sample_rows[0].get('split', '') or getattr(dataset, '_split', '')} "
        f"corruption={corruption} severity={severity} "
        f"labeled={counts['evaluated_bscans']} recordings={len(recording_rows)} "
        f"mae={table_metrics_mean['mae_px']:.3f}+-{table_metrics_std['mae_px']:.3f} "
        f"p95={table_metrics_mean['p95_px']:.3f}+-{table_metrics_std['p95_px']:.3f} "
        f"bias={table_metrics_mean['bias_px']:.3f}+-{table_metrics_std['bias_px']:.3f} "
        f"abs_bias={table_metrics_mean['abs_bias_px']:.3f}+-{table_metrics_std['abs_bias_px']:.3f} "
        f"acc2={table_metrics_mean['acc_2px']:.3f}+-{table_metrics_std['acc_2px']:.3f} "
        f"acc4={table_metrics_mean['acc_4px']:.3f}+-{table_metrics_std['acc_4px']:.3f} "
        f"spike={table_metrics_mean['spike_rate']:.4f}+-{table_metrics_std['spike_rate']:.4f} "
        f"runtime_ms={table_metrics_mean['runtime_ms']:.3f}+-{table_metrics_std['runtime_ms']:.3f}"
    )
    print(f"[classical:{method_key}] wrote {per_scan_path}")
    print(f"[classical:{method_key}] wrote {per_recording_path}")
    print(f"[classical:{method_key}] wrote {summary_path}")

    return summary


def main() -> None:
    args = parse_args()
    cfg = load_training_cfg(args.config)
    dataset_str = resolve_dataset_path(str(get_cfg(cfg, ("train", "dataset_path"), "OCT:root=data/oct:extra=data/oct/extra")))
    eval_split_name = "real_hard" if args.eval_dir is not None else str(args.split)

    _, dataset_tokens = _parse_dataset_path(dataset_str)
    extra_root = Path(dataset_tokens.get("extra", str(Path(dataset_tokens["root"]) / "extra")))
    split_rows = _read_split_rows(extra_root) if args.eval_dir is None else {}

    if args.eval_dir is None:
        dataset_eval = make_dataset(dataset_str=_with_split(dataset_str, args.split), transform=None)
        if not isinstance(dataset_eval, OCT):
            raise TypeError(f"Expected OCT dataset, got {type(dataset_eval)}")
    else:
        dataset_eval = DirectoryCurveEvalDataset(DirectoryEvalConfig(eval_dir=Path(args.eval_dir), split_name=eval_split_name))
        split_rows = split_rows_for_directory_dataset(dataset_eval)
    dataset_kappa = make_dataset(dataset_str=_with_split(dataset_str, args.kappa_split), transform=None)
    if not isinstance(dataset_kappa, OCT):
        raise TypeError(f"Expected OCT dataset, got {type(dataset_kappa)}")

    acc_tolerances = tuple(float(value) for value in (args.acc_tolerances or list(DEFAULT_ACC_TOLERANCES)))
    spike_kappa = (
        float(args.spike_kappa)
        if args.spike_kappa is not None
        else _estimate_spike_kappa(dataset_kappa, quantile=float(args.spike_kappa_quantile))
    )
    output_suffix = corruption_output_suffix(args.corruption, args.severity)
    output_dir = args.output_dir or (REPO_ROOT / "outputs" / "classical_eval" / f"{eval_split_name}{output_suffix}")
    methods = list(CLASSICAL_METHODS) if args.method == "all" else [resolve_method_key(str(args.method))]
    dataset_root = Path(dataset_tokens["root"])
    default_fallback_background = dataset_root / "background" / "background.jpg"
    fallback_background_path = (
        args.fallback_background
        if args.fallback_background is not None
        else (default_fallback_background if default_fallback_background.exists() else None)
    )
    fallback_background = _load_gray(Path("."), fallback_background_path) if fallback_background_path is not None else None

    for method in methods:
        _evaluate_method(
            method=method,
            dataset=dataset_eval,
            split_rows=split_rows,
            output_dir=output_dir,
            acc_tolerances=acc_tolerances,
            spike_kappa=spike_kappa,
            limit=args.limit,
            fallback_background=fallback_background,
            corruption=str(args.corruption),
            severity=str(args.severity),
            corruption_seed=int(args.corruption_seed),
            write_overlays=bool(args.write_overlays),
            overlay_limit=int(args.overlay_limit),
        )


if __name__ == "__main__":
    main()
