#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def _find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").exists() and (candidate / "dinoct").is_dir():
            return candidate
    raise RuntimeError("Could not locate repo root from script path.")


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.corruptions import CORRUPTION_SEVERITIES, CORRUPTION_TYPES  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "train" / "oct.yaml"
DEFAULT_CKPT = REPO_ROOT / "outputs" / "paper_results" / "checkpoints" / "post_train" / "fused_curve_best.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paper robustness suite for a learned curve checkpoint.")
    parser.add_argument("--name", default=None, help="Optional run name. Defaults to the checkpoint stem.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--curve-ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--kappa-split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--corruptions",
        nargs="*",
        choices=[c for c in CORRUPTION_TYPES if c != "clean"],
        default=["stripe", "ghost", "dropout"],
        help="Synthetic corruption families to evaluate.",
    )
    parser.add_argument(
        "--severities",
        nargs="*",
        choices=list(CORRUPTION_SEVERITIES),
        default=["medium", "severe"],
        help="Severity levels to evaluate.",
    )
    parser.add_argument("--include-clean", action="store_true", help="Also evaluate the clean condition.")
    parser.add_argument("--corruption-seed", type=int, default=0)
    parser.add_argument("--write-overlays", action="store_true")
    parser.add_argument("--overlay-limit", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _conditions(*, include_clean: bool, corruptions: list[str], severities: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = [("clean", "medium")] if include_clean else []
    for corruption in corruptions:
        for severity in severities:
            out.append((str(corruption), str(severity)))
    return out


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _slugify_name(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value).strip())
    text = text.strip("._-")
    return text or "run"


def _default_run_name(args: argparse.Namespace) -> str:
    if args.name:
        return _slugify_name(str(args.name))
    stem = str(Path(args.curve_ckpt).stem).strip().lower()
    if stem.startswith("fused_curve"):
        return "dinoct"
    return _slugify_name(stem)


def _condition_name(corruption: str, severity: str) -> str:
    return "clean" if corruption == "clean" else f"{corruption}_{severity}"


def main() -> None:
    args = parse_args()
    run_name = _default_run_name(args)
    base_dir = args.output_dir or (REPO_ROOT / "outputs" / "paper_results" / "robustness" / run_name)
    base_dir.mkdir(parents=True, exist_ok=True)

    aggregate_rows: list[dict[str, Any]] = []
    for corruption, severity in _conditions(
        include_clean=bool(args.include_clean),
        corruptions=[str(value) for value in (args.corruptions or [])],
        severities=[str(value) for value in (args.severities or [])],
    ):
        run_dir = base_dir / _condition_name(corruption, severity) / run_name
        cmd = [
            sys.executable,
            str(REPO_ROOT / "eval" / "evaluate_curve.py"),
            "--config", str(args.config),
            "--curve-ckpt", str(args.curve_ckpt),
            "--split", str(args.split),
            "--kappa-split", str(args.kappa_split),
            "--corruption", corruption,
            "--severity", severity,
            "--corruption-seed", str(args.corruption_seed),
            "--output-dir", str(run_dir),
            "--device", str(args.device),
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
        ]
        if args.write_overlays:
            cmd.append("--write-overlays")
            cmd.extend(["--overlay-limit", str(args.overlay_limit)])
        summary_path = run_dir / "summary.json"
        if args.resume and summary_path.exists():
            summary = _read_json(summary_path)
        else:
            print(f"[robustness] running: {' '.join(cmd)}")
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
            summary = _read_json(summary_path)
        table_mean = summary.get("table_metrics_per_recording_mean", {})
        table_std = summary.get("table_metrics_per_recording_std", {})
        aggregate_rows.append(
            {
                "mode": "curve",
                "model_or_method": str(summary.get("model_type", "curve")),
                "corruption": corruption,
                "severity": severity,
                **table_mean,
                **{f"{key}_std": value for key, value in table_std.items()},
                "labeled_bscans": summary.get("counts", {}).get("labeled_bscans", 0),
                "recordings": summary.get("counts", {}).get("recordings", 0),
            }
        )

    if not aggregate_rows:
        raise ValueError("No robustness summaries were produced.")

    fieldnames = [
        "mode",
        "model_or_method",
        "corruption",
        "severity",
        "mae_px",
        "mae_px_std",
        "p95_px",
        "p95_px_std",
        "bias_px",
        "bias_px_std",
        "abs_bias_px",
        "abs_bias_px_std",
        "acc_2px",
        "acc_2px_std",
        "acc_4px",
        "acc_4px_std",
        "spike_rate",
        "spike_rate_std",
        "runtime_ms",
        "runtime_ms_std",
        "labeled_bscans",
        "recordings",
    ]
    csv_path = base_dir / "robustness_suite_summary.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregate_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    json_path = base_dir / "robustness_suite_summary.json"
    json_path.write_text(json.dumps(aggregate_rows, indent=2) + "\n")
    print(f"[robustness] wrote {csv_path}")
    print(f"[robustness] wrote {json_path}")


if __name__ == "__main__":
    main()
