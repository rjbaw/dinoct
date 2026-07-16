#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


VARIANTS = [
    {
        "key": "all_backbone_norms",
        "label": "All backbone norms",
        "change": "Reference",
        "norm_mode": "all_norms",
    },
    {
        "key": "final_norm_only",
        "label": "Final norm only",
        "change": r"all backbone norms $\rightarrow$ final norm only",
        "norm_mode": "final_only",
    },
    {
        "key": "frozen_backbone_norms",
        "label": "Frozen backbone norms",
        "change": r"all backbone norms $\rightarrow$ frozen",
        "norm_mode": "none",
    },
]

SEVERE_CORRUPTIONS = ("stripe", "ghost", "dropout")
AGGREGATE_METRIC_KEYS = [
    "clean_test_mae",
    "stripe_severe_mae",
    "ghost_severe_mae",
    "dropout_severe_mae",
    "severe_mean_mae",
    "worst_severe_mae",
    "real_hard_mae",
    "real_hard_acc2",
]


def repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "dinoct").is_dir():
            return parent
    raise RuntimeError("Could not locate repo root")


REPO_ROOT = repo_root()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run backbone-normalization adaptation sensitivity analysis."
    )
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "train" / "oct.yaml")
    parser.add_argument(
        "--pretrained-backbone",
        type=Path,
        default=REPO_ROOT / "outputs" / "pretrain" / "dinov3_pretrain.pth",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "paper_results" / "ablations" / "backbone_norm_sensitivity",
    )
    parser.add_argument("--real-hard-dir", type=Path, default=REPO_ROOT / "data" / "oct" / "eval" / "hard")
    parser.add_argument(
        "--paper-table-path",
        type=Path,
        default=REPO_ROOT / "outputs" / "paper_tables" / "norm_adaptation_ablation.tex",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-num-workers", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    return json.loads(path.read_text())


def run(cmd: list[str]) -> None:
    print("[norm-sensitivity] running: {}".format(" ".join(cmd)))
    env = os.environ.copy()
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", env["PYTORCH_ALLOC_CONF"])
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


def resolve_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return [int(seed) for seed in args.seeds]
    return [int(args.seed)]


def variant_run_dir(*, output_root: Path, variant_key: str, seed: int, multi_seed: bool) -> Path:
    base_dir = output_root / variant_key
    if not multi_seed:
        return base_dir
    return base_dir / "seed_{}".format(seed)


def metadata_payload(args: argparse.Namespace, variant: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "script": "eval/paper/backbone_norm_sensitivity.py",
        "trainer": "python -m dinoct",
        "config": str(args.config),
        "pretrained_backbone": str(args.pretrained_backbone),
        "seed": int(seed),
        "batch_size": int(args.batch_size),
        "train_num_workers": int(args.train_num_workers),
        "eval_batch_size": int(args.eval_batch_size),
        "eval_num_workers": int(args.eval_num_workers),
        "device": str(args.device),
        "real_hard_dir": str(args.real_hard_dir),
        "backbone_norm_mode": str(variant["norm_mode"]),
        "backbone_checkpointing": True,
    }


def metadata_matches(path: Path, args: argparse.Namespace, variant: dict[str, Any], seed: int) -> bool:
    if not path.exists():
        return False
    try:
        return read_json(path) == metadata_payload(args, variant, seed)
    except Exception:
        return False


def metrics_from_summary(path: Path) -> dict[str, float]:
    summary = read_json(path)
    assert isinstance(summary, dict)
    model_load = summary.get("model_load")
    if isinstance(model_load, dict):
        missing = int(model_load.get("missing_key_count", 0) or 0)
        unexpected = int(model_load.get("unexpected_key_count", 0) or 0)
        if missing or unexpected:
            raise ValueError(
                "Checkpoint load mismatch in {}: missing_key_count={} unexpected_key_count={}".format(
                    path,
                    missing,
                    unexpected,
                )
            )
    mean_block = summary["table_metrics_per_recording_mean"]
    return {
        "mae": float(mean_block["mae_px"]),
        "acc2": float(mean_block["acc_2px"]),
    }


def robustness_metrics_from_summary(path: Path) -> dict[str, float]:
    rows = read_json(path)
    assert isinstance(rows, list)
    severe_rows = [row for row in rows if str(row.get("severity", "")).lower() == "severe"]
    if not severe_rows:
        raise ValueError("No severe rows found in {}".format(path))

    out: dict[str, float] = {}
    severe_values: list[float] = []
    for corruption in SEVERE_CORRUPTIONS:
        row = next((item for item in severe_rows if str(item.get("corruption", "")).lower() == corruption), None)
        if row is None:
            raise ValueError("Missing severe {} row in {}".format(corruption, path))
        value = float(row["mae_px"])
        out["{}_severe_mae".format(corruption)] = value
        severe_values.append(value)
    out["severe_mean_mae"] = sum(severe_values) / float(len(severe_values))
    out["worst_severe_mae"] = max(severe_values)
    return out


def seed_mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot aggregate an empty metric list")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.fmean(values)), float(statistics.pstdev(values))  # ddof=0: unified std convention (2026-07-13)


def aggregate_variant_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot aggregate an empty variant row list")
    aggregated: dict[str, Any] = {
        "recipe_key": rows[0]["recipe_key"],
        "label": rows[0]["label"],
        "change": rows[0]["change"],
        "norm_mode": rows[0]["norm_mode"],
        "num_seeds": len(rows),
        "seeds": "|".join(str(int(row["seed"])) for row in rows),
        "train_dirs": "|".join(str(row["train_dir"]) for row in rows),
    }
    for key in AGGREGATE_METRIC_KEYS:
        mean_value, std_value = seed_mean_std([float(row[key]) for row in rows])
        aggregated[key] = mean_value
        aggregated["{}_seed_std".format(key)] = std_value
    return aggregated


def fmt(mean_value: float, std_value: float, *, bold: bool, multi_seed: bool) -> str:
    value = "{:.3f}".format(mean_value)
    if bold:
        value = r"\textbf{" + value + "}"
    if not multi_seed:
        return value
    return value + r" $\pm$ " + "{:.3f}".format(std_value)


def render_norm_table(rows: list[dict[str, Any]]) -> str:
    multi_seed = any(int(row.get("num_seeds", 1)) > 1 for row in rows)
    min_keys = ["clean_test_mae", "severe_mean_mae", "worst_severe_mae", "real_hard_mae"]
    max_keys = ["real_hard_acc2"]
    mins = {key: min(float(row[key]) for row in rows) for key in min_keys}
    maxs = {key: max(float(row[key]) for row in rows) for key in max_keys}

    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Backbone normalization adaptation sensitivity analysis across three seeds. Values are mean $\pm$ std across seeds.}",
        r"\begin{footnotesize}",
        r"\begin{tabularx}{\textwidth}{@{}lXccccc@{}}",
        r"\toprule",
        r"Configuration & Change & Clean MAE & Avg. severe synthetic MAE & Worst severe synthetic MAE & Real-artifact MAE & Real-artifact Acc@2px \\",
        r"\midrule",
    ]
    for row in rows:
        cells = [
            str(row["label"]),
            str(row["change"]),
            fmt(
                float(row["clean_test_mae"]),
                float(row["clean_test_mae_seed_std"]),
                bold=float(row["clean_test_mae"]) == mins["clean_test_mae"],
                multi_seed=multi_seed,
            ),
            fmt(
                float(row["severe_mean_mae"]),
                float(row["severe_mean_mae_seed_std"]),
                bold=float(row["severe_mean_mae"]) == mins["severe_mean_mae"],
                multi_seed=multi_seed,
            ),
            fmt(
                float(row["worst_severe_mae"]),
                float(row["worst_severe_mae_seed_std"]),
                bold=float(row["worst_severe_mae"]) == mins["worst_severe_mae"],
                multi_seed=multi_seed,
            ),
            fmt(
                float(row["real_hard_mae"]),
                float(row["real_hard_mae_seed_std"]),
                bold=float(row["real_hard_mae"]) == mins["real_hard_mae"],
                multi_seed=multi_seed,
            ),
            fmt(
                float(row["real_hard_acc2"]),
                float(row["real_hard_acc2_seed_std"]),
                bold=float(row["real_hard_acc2"]) == maxs["real_hard_acc2"],
                multi_seed=multi_seed,
            ),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{footnotesize}",
            r"\label{tab:norm_adaptation_ablation}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def run_variant_seed(args: argparse.Namespace, variant: dict[str, Any], seed: int, multi_seed: bool) -> dict[str, Any]:
    variant_dir = variant_run_dir(
        output_root=args.output_root,
        variant_key=str(variant["key"]),
        seed=seed,
        multi_seed=multi_seed,
    )
    train_dir = variant_dir / "train"
    eval_dir = variant_dir / "eval"
    metadata_path = variant_dir / "run_metadata.json"
    ckpt = train_dir / "post_train" / "fused_curve_best.pth"
    clean_summary = eval_dir / "clean_test" / "summary.json"
    hard_summary = eval_dir / "real_hard" / "summary.json"
    robustness_summary = eval_dir / "robustness" / "robustness_suite_summary.json"

    ready = (
        ckpt.exists()
        and clean_summary.exists()
        and hard_summary.exists()
        and robustness_summary.exists()
        and metadata_matches(metadata_path, args, variant, seed)
    )
    if not (args.resume and ready):
        train_cmd = [
            sys.executable,
            "-m",
            "dinoct",
            "--config",
            str(args.config),
            "--output-dir",
            str(train_dir),
            "--seed",
            str(seed),
            "--post-train-only",
            "--pretrained-backbone",
            str(args.pretrained_backbone),
            "--post-train-batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.train_num_workers),
            "--post-train-backbone-norm-mode",
            str(variant["norm_mode"]),
        ]
        if not (args.resume and ckpt.exists() and metadata_matches(metadata_path, args, variant, seed)):
            run(train_cmd)

        clean_cmd = [
            sys.executable,
            str(REPO_ROOT / "eval" / "evaluate_curve.py"),
            "--config",
            str(args.config),
            "--curve-ckpt",
            str(ckpt),
            "--split",
            "test",
            "--output-dir",
            str(eval_dir / "clean_test"),
            "--device",
            str(args.device),
            "--batch-size",
            str(args.eval_batch_size),
            "--num-workers",
            str(args.eval_num_workers),
        ]
        run(clean_cmd)

        hard_cmd = [
            sys.executable,
            str(REPO_ROOT / "eval" / "evaluate_curve.py"),
            "--config",
            str(args.config),
            "--curve-ckpt",
            str(ckpt),
            "--eval-dir",
            str(args.real_hard_dir),
            "--output-dir",
            str(eval_dir / "real_hard"),
            "--device",
            str(args.device),
            "--batch-size",
            str(args.eval_batch_size),
            "--num-workers",
            str(args.eval_num_workers),
        ]
        run(hard_cmd)

        robust_cmd = [
            sys.executable,
            str(REPO_ROOT / "eval" / "paper" / "robustness.py"),
            "--name",
            "{}_seed_{}".format(variant["key"], seed),
            "--config",
            str(args.config),
            "--curve-ckpt",
            str(ckpt),
            "--split",
            "test",
            "--corruptions",
            "stripe",
            "ghost",
            "dropout",
            "--severities",
            "severe",
            "--output-dir",
            str(eval_dir / "robustness"),
            "--device",
            str(args.device),
            "--batch-size",
            str(args.eval_batch_size),
            "--num-workers",
            str(args.eval_num_workers),
        ]
        run(robust_cmd)

        metadata_path.write_text(json.dumps(metadata_payload(args, variant, seed), indent=2) + "\n")

    clean = metrics_from_summary(clean_summary)
    hard = metrics_from_summary(hard_summary)
    robust = robustness_metrics_from_summary(robustness_summary)
    return {
        "recipe_key": str(variant["key"]),
        "label": str(variant["label"]),
        "change": str(variant["change"]),
        "norm_mode": str(variant["norm_mode"]),
        "seed": int(seed),
        "clean_test_mae": clean["mae"],
        "stripe_severe_mae": robust["stripe_severe_mae"],
        "ghost_severe_mae": robust["ghost_severe_mae"],
        "dropout_severe_mae": robust["dropout_severe_mae"],
        "severe_mean_mae": robust["severe_mean_mae"],
        "worst_severe_mae": robust["worst_severe_mae"],
        "real_hard_mae": hard["mae"],
        "real_hard_acc2": hard["acc2"],
        "train_dir": str(train_dir),
    }


def main() -> None:
    args = parse_args()
    if not args.pretrained_backbone.exists():
        raise SystemExit("Missing pretrained backbone: {}".format(args.pretrained_backbone))
    args.output_root.mkdir(parents=True, exist_ok=True)

    seeds = resolve_seeds(args)
    multi_seed = len(seeds) > 1

    per_seed_rows: list[dict[str, Any]] = []
    aggregated_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        variant_rows = [run_variant_seed(args, variant, seed, multi_seed) for seed in seeds]
        per_seed_rows.extend(variant_rows)
        aggregated_rows.append(aggregate_variant_rows(variant_rows))

    per_seed_csv_path = args.output_root / "backbone_norm_sensitivity_per_seed.csv"
    with per_seed_csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_seed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_seed_rows)

    aggregated_csv_path = args.output_root / "backbone_norm_sensitivity.csv"
    with aggregated_csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(aggregated_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregated_rows)

    tex = render_norm_table(aggregated_rows)
    tex_path = args.output_root / "backbone_norm_sensitivity.tex"
    tex_path.write_text(tex)
    if args.paper_table_path:
        args.paper_table_path.parent.mkdir(parents=True, exist_ok=True)
        args.paper_table_path.write_text(tex)

    print("[norm-sensitivity] wrote {}".format(per_seed_csv_path))
    print("[norm-sensitivity] wrote {}".format(aggregated_csv_path))
    print("[norm-sensitivity] wrote {}".format(tex_path))
    if args.paper_table_path:
        print("[norm-sensitivity] wrote {}".format(args.paper_table_path))


if __name__ == "__main__":
    main()
