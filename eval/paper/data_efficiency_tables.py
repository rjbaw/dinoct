#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

BUDGET_ROWS = [
    "50 images",
    "100 images",
    "250 images",
    "500 images",
    "1000 images",
    "full train set",
]

METHOD_SPECS = [
    ("DINOCT", "Data efficiency performance DINOCT (mean $\\pm$ std)", "tab:lowdata_dinoct"),
    ("UNET", "Data efficiency performance UNET (mean $\\pm$ std)", "tab:lowdata_unet"),
    ("FCBR", "Data efficiency performance FCBR (mean $\\pm$ std)", "tab:lowdata_fcbr"),
]


def _find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").exists() and (candidate / "dinoct").is_dir():
            return candidate
    raise RuntimeError("Could not locate repo root from script path.")


REPO_ROOT = _find_repo_root()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render LaTeX data-efficiency tables from a manifest of summary.json paths."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "outputs" / "paper_tables" / "data_efficiency_manifest.csv",
        help="CSV with columns: method,budget,summary_path,notes",
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=REPO_ROOT / "outputs" / "paper_tables" / "data_efficiency_tables.tex",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "outputs" / "paper_tables" / "data_efficiency_explicit.csv",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any listed summary path is missing or malformed.",
    )
    return parser.parse_args()


def _write_template_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["method", "budget", "summary_path", "notes"])
        writer.writeheader()
        for method, _caption, _label in METHOD_SPECS:
            for budget in BUDGET_ROWS:
                writer.writerow({"method": method, "budget": budget, "summary_path": "", "notes": ""})


def _resolve_summary_path(raw: str) -> Path | None:
    value = raw.strip()
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if path.is_dir():
        path = path / "summary.json"
    return path


def _load_metrics(summary_path: Path) -> dict[str, float]:
    payload = json.loads(summary_path.read_text())
    mean_block = payload["table_metrics_per_recording_mean"]
    std_block = payload["table_metrics_per_recording_std"]
    return {
        "mae_mean": float(mean_block["mae_px"]),
        "mae_std": float(std_block["mae_px"]),
        "p95_mean": float(mean_block["p95_px"]),
        "p95_std": float(std_block["p95_px"]),
        "bias_mean": float(mean_block["bias_px"]),
        "bias_std": float(std_block["bias_px"]),
        "acc2_mean": float(mean_block["acc_2px"]),
        "acc2_std": float(std_block["acc_2px"]),
        "spike_mean": 100.0 * float(mean_block["spike_rate"]),
        "spike_std": 100.0 * float(std_block["spike_rate"]),
    }


def _fmt_pm(mean: float, std: float, *, digits: int = 3) -> str:
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _blank_row(budget: str) -> str:
    return f"{budget} &  &  &  &  &     \\\\" 


def _render_row(budget: str, metrics: dict[str, float] | None) -> str:
    if metrics is None:
        return _blank_row(budget)
    return (
        f"{budget} & "
        f"{_fmt_pm(metrics['mae_mean'], metrics['mae_std'])} & "
        f"{_fmt_pm(metrics['p95_mean'], metrics['p95_std'])} & "
        f"{_fmt_pm(metrics['bias_mean'], metrics['bias_std'])} & "
        f"{_fmt_pm(metrics['acc2_mean'], metrics['acc2_std'])} & "
        f"{_fmt_pm(metrics['spike_mean'], metrics['spike_std'], digits=2)}     \\\\" 
    )


def _render_table(caption: str, label: str, rows: dict[str, dict[str, float] | None]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        r"\begin{tabularx}{\textwidth}{@{}Xccccc@{}}",
        r"\toprule",
        r"Method & MAE (px) \(\downarrow\) & P95 error (px) \(\downarrow\) & Bias (px) & Acc@2px \(\uparrow\) & SpikeRate@\(\kappa\) (\%) \(\downarrow\) \\",
        r"\midrule",
    ]
    for budget in BUDGET_ROWS:
        lines.append(_render_row(budget, rows.get(budget)))
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            f"\\label{{{label}}}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest
    if not manifest_path.exists():
        _write_template_manifest(manifest_path)
        raise SystemExit(
            f"Manifest not found. Wrote template to {manifest_path}. Fill summary_path values and rerun."
        )

    rows_by_method: dict[str, dict[str, dict[str, float] | None]] = {
        method: {budget: None for budget in BUDGET_ROWS} for method, _caption, _label in METHOD_SPECS
    }
    explicit_rows: list[dict[str, Any]] = []
    missing: list[str] = []

    with manifest_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            method = str(row.get("method", "")).strip().upper()
            budget = str(row.get("budget", "")).strip()
            notes = str(row.get("notes", "")).strip()
            if method not in rows_by_method or budget not in rows_by_method[method]:
                continue

            summary_path = _resolve_summary_path(str(row.get("summary_path", "")))
            metrics: dict[str, float] | None = None
            if summary_path is not None:
                if summary_path.exists():
                    metrics = _load_metrics(summary_path)
                else:
                    missing.append(str(summary_path))
                    if args.strict:
                        raise FileNotFoundError(summary_path)

            rows_by_method[method][budget] = metrics
            explicit_rows.append(
                {
                    "method": method,
                    "budget": budget,
                    "summary_path": "" if summary_path is None else str(summary_path),
                    "mae_mean": "" if metrics is None else metrics["mae_mean"],
                    "mae_std": "" if metrics is None else metrics["mae_std"],
                    "p95_mean": "" if metrics is None else metrics["p95_mean"],
                    "p95_std": "" if metrics is None else metrics["p95_std"],
                    "bias_mean": "" if metrics is None else metrics["bias_mean"],
                    "bias_std": "" if metrics is None else metrics["bias_std"],
                    "acc2_mean": "" if metrics is None else metrics["acc2_mean"],
                    "acc2_std": "" if metrics is None else metrics["acc2_std"],
                    "spike_percent_mean": "" if metrics is None else metrics["spike_mean"],
                    "spike_percent_std": "" if metrics is None else metrics["spike_std"],
                    "notes": notes,
                }
            )

    tex_blocks = []
    for method, caption, label in METHOD_SPECS:
        tex_blocks.append(_render_table(caption, label, rows_by_method[method]))
        tex_blocks.append("")

    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text("\n".join(tex_blocks).rstrip() + "\n")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "method",
                "budget",
                "summary_path",
                "mae_mean",
                "mae_std",
                "p95_mean",
                "p95_std",
                "bias_mean",
                "bias_std",
                "acc2_mean",
                "acc2_std",
                "spike_percent_mean",
                "spike_percent_std",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(explicit_rows)

    print(f"[lowdata] wrote {args.output_tex}")
    print(f"[lowdata] wrote {args.output_csv}")
    if missing:
        print(f"[lowdata] warning: {len(missing)} listed summary paths were missing", file=sys.stderr)
        for path in missing:
            print(f"[lowdata] missing: {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
