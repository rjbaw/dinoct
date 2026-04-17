#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


LEARNED_ROWS = [
    ("FCBR", "fcbr"),
    ("UNet", "unet"),
    ("DINOCT", "dinoct"),
]
CLASSICAL_REAL_HARD_ROWS = [
    ("GF", "gf"),
    ("GF-B", "gf_b"),
    ("GRAD-SG", "grad_sg"),
    ("GRAD-ENG", "grad_eng"),
    ("TUNED-SOBEL-DC", "legacy_sobel_dc"),
]
SEVERE_KEYS = [
    ("stripe", "severe"),
    ("ghost", "severe"),
    ("dropout", "severe"),
]


def _find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").exists() and (candidate / "dinoct").is_dir():
            return candidate
    raise RuntimeError("Could not locate repo root from script path.")


REPO_ROOT = _find_repo_root()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render aggregate stress summaries, degradation ratios, and failure-rate tables "
            "from paper robustness outputs."
        )
    )
    parser.add_argument(
        "--robustness-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "paper_results" / "robustness",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "paper_results" / "robustness",
    )
    parser.add_argument(
        "--classical-real-hard-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "classical_eval" / "real_hard",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "paper_tables",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_suite_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["corruption"], row["severity"]): row for row in csv.DictReader(path.open())}


def _float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _render_tex_table(
    *,
    caption: str,
    label: str,
    header: list[str],
    rows: list[list[str]],
    column_spec: str,
    env: str = "table*",
) -> str:
    lines = [
        rf"\begin{{{env}}}[tb]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\begin{{tabularx}}{{\textwidth}}{{{column_spec}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabularx}",
        rf"\label{{{label}}}",
        rf"\end{{{env}}}",
    ])
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _failure_row_from_scan_csv(*, label: str, path: Path) -> dict[str, Any]:
    rows = list(csv.DictReader(path.open()))
    n = max(len(rows), 1)
    return {
        "method": label,
        "failure_rate_mae_gt_5": sum(float(row["mae_px"]) > 5.0 for row in rows) / n,
        "failure_rate_mae_gt_10": sum(float(row["mae_px"]) > 10.0 for row in rows) / n,
        "failure_rate_p95_gt_10": sum(float(row["p95_px"]) > 10.0 for row in rows) / n,
        "unsafe_spike_rate": sum(float(row["spike_rate"]) > 0.0 for row in rows) / n,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    stress_rows: list[dict[str, Any]] = []
    ratio_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []

    for label, slug in LEARNED_ROWS:
        suite_rows = _load_suite_rows(args.robustness_root / slug / "robustness_suite_summary.csv")
        hard_summary = _read_json(args.robustness_root / slug / "real_hard" / "summary.json")
        hard_scans = list(csv.DictReader((args.robustness_root / slug / "real_hard" / "per_scan_metrics.csv").open()))

        clean_mean = _float(suite_rows[("clean", "medium")], "mae_px")
        severe_means = [_float(suite_rows[key], "mae_px") for key in SEVERE_KEYS]
        severe_avg = _mean(severe_means)
        severe_worst = max(severe_means)
        hard_mean = float(hard_summary["table_metrics_per_recording_mean"]["mae_px"])
        hard_acc2 = float(hard_summary["table_metrics_per_recording_mean"]["acc_2px"])
        hard_spike = float(hard_summary["table_metrics_per_recording_mean"]["spike_rate"])

        stress_rows.append(
            {
                "method": label,
                "clean_mae": clean_mean,
                "avg_severe_synth_mae": severe_avg,
                "worst_severe_synth_mae": severe_worst,
                "real_hard_mae": hard_mean,
                "real_hard_acc2": hard_acc2,
                "real_hard_spike_rate": hard_spike,
            }
        )
        ratio_rows.append(
            {
                "method": label,
                "avg_severe_over_clean": severe_avg / clean_mean,
                "real_hard_over_clean": hard_mean / clean_mean,
            }
        )

        failure_rows.append(
            _failure_row_from_scan_csv(
                label=label,
                path=args.robustness_root / slug / "real_hard" / "per_scan_metrics.csv",
            )
        )

        for row in hard_scans:
            distribution_rows.append(
                {
                    "method": label,
                    "sample_id": row["sample_id"],
                    "recording_id": row["recording_id"],
                    "mae_px": float(row["mae_px"]),
                    "p95_px": float(row["p95_px"]),
                    "acc_2px": float(row["acc_2px"]),
                    "spike_rate": float(row["spike_rate"]),
                }
            )

    classical_failure_rows: list[dict[str, Any]] = []
    for label, slug in CLASSICAL_REAL_HARD_ROWS:
        scan_csv = args.classical_real_hard_root / slug / "per_scan_metrics.csv"
        if not scan_csv.exists():
            print(f"[stress-analysis] warning: missing classical real-hard scan CSV, skipping {scan_csv}")
            continue
        classical_failure_rows.append(_failure_row_from_scan_csv(label=label, path=scan_csv))

    failure_rows_all = classical_failure_rows + failure_rows

    stress_fields = [
        "method",
        "clean_mae",
        "avg_severe_synth_mae",
        "worst_severe_synth_mae",
        "real_hard_mae",
        "real_hard_acc2",
        "real_hard_spike_rate",
    ]
    ratio_fields = ["method", "avg_severe_over_clean", "real_hard_over_clean"]
    failure_fields = [
        "method",
        "failure_rate_mae_gt_5",
        "failure_rate_mae_gt_10",
        "failure_rate_p95_gt_10",
        "unsafe_spike_rate",
    ]
    distribution_fields = [
        "method",
        "sample_id",
        "recording_id",
        "mae_px",
        "p95_px",
        "acc_2px",
        "spike_rate",
    ]

    stress_csv = args.output_dir / "learned_stress_summary.csv"
    ratio_csv = args.output_dir / "learned_degradation_ratios.csv"
    failure_csv = args.output_dir / "learned_failure_rates_real_hard.csv"
    dist_csv = args.output_dir / "learned_real_hard_distribution.csv"
    _write_csv(stress_csv, stress_fields, stress_rows)
    _write_csv(ratio_csv, ratio_fields, ratio_rows)
    _write_csv(failure_csv, failure_fields, failure_rows_all)
    _write_csv(dist_csv, distribution_fields, distribution_rows)

    stress_tex = _render_tex_table(
        caption=(
            "Aggregate robustness summary for learned models. Severe synthetic MAE averages stripe-severe, "
            "ghost-severe, and dropout-severe; worst severe synthetic MAE is the maximum of those three conditions."
        ),
        label="tab:stress_summary",
        header=[
            "Method",
            "Clean MAE",
            "Avg severe synthetic MAE",
            "Worst severe synthetic MAE",
            "Real difficult MAE",
            "Real difficult Acc@2px",
            "Real difficult SpikeRate@\\(\\kappa\\) (\\%)",
        ],
        rows=[
            [
                row["method"],
                _fmt(row["clean_mae"]),
                _fmt(row["avg_severe_synth_mae"]),
                _fmt(row["worst_severe_synth_mae"]),
                _fmt(row["real_hard_mae"]),
                _fmt(row["real_hard_acc2"]),
                _fmt(100.0 * row["real_hard_spike_rate"]),
            ]
            for row in stress_rows
        ],
        column_spec="@{}Xcccccc@{}",
    )
    ratio_tex = _render_tex_table(
        caption="Robustness degradation ratios for learned models, relative to clean-test MAE.",
        label="tab:stress_ratios",
        header=["Method", "Avg severe synthetic / clean", "Real difficult / clean"],
        rows=[
            [
                row["method"],
                f"{row['avg_severe_over_clean']:.2f}$\\times$",
                f"{row['real_hard_over_clean']:.2f}$\\times$",
            ]
            for row in ratio_rows
        ],
        column_spec="@{}Xcc@{}",
        env="table",
    )
    failure_tex = _render_tex_table(
        caption="Real-difficult catastrophic failure rates computed from per-B-scan predictions.",
        label="tab:failure_rates_realhard",
        header=["Method", "FailureRate@5px", "FailureRate@10px", "P95FailureRate@10px", "Unsafe spike rate"],
        rows=[
            [
                row["method"],
                _fmt(100.0 * row["failure_rate_mae_gt_5"]),
                _fmt(100.0 * row["failure_rate_mae_gt_10"]),
                _fmt(100.0 * row["failure_rate_p95_gt_10"]),
                _fmt(100.0 * row["unsafe_spike_rate"]),
            ]
            for row in failure_rows_all
        ],
        column_spec="@{}Xcccc@{}",
        env="table",
    )

    for out in [args.output_dir / "learned_stress_summary.tex", args.table_dir / "learned_stress_summary.tex"]:
        out.write_text(stress_tex)
    for out in [args.output_dir / "learned_degradation_ratios.tex", args.table_dir / "learned_degradation_ratios.tex"]:
        out.write_text(ratio_tex)
    for out in [args.output_dir / "learned_failure_rates_real_hard.tex", args.table_dir / "learned_failure_rates_real_hard.tex"]:
        out.write_text(failure_tex)

    print(f"[stress-analysis] wrote {stress_csv}")
    print(f"[stress-analysis] wrote {ratio_csv}")
    print(f"[stress-analysis] wrote {failure_csv}")
    print(f"[stress-analysis] wrote {dist_csv}")
    print(f"[stress-analysis] wrote {args.output_dir / 'learned_stress_summary.tex'}")
    print(f"[stress-analysis] wrote {args.output_dir / 'learned_degradation_ratios.tex'}")
    print(f"[stress-analysis] wrote {args.output_dir / 'learned_failure_rates_real_hard.tex'}")


if __name__ == "__main__":
    main()
