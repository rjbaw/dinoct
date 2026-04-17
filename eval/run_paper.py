#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def _find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").exists() and (candidate / "dinoct").is_dir():
            return candidate
    raise RuntimeError("Could not locate repo root from script path.")


REPO_ROOT = _find_repo_root()
PAPER_RESULTS_ROOT = REPO_ROOT / "outputs" / "paper_results"
PAPER_CHECKPOINT_ROOT = PAPER_RESULTS_ROOT / "checkpoints"
PAPER_ROBUSTNESS_ROOT = PAPER_RESULTS_ROOT / "robustness"
PAPER_ABLATION_ROOT = PAPER_RESULTS_ROOT / "ablations" / "final_component_ablation"
PAPER_DATA_EFF_ROOT = PAPER_RESULTS_ROOT / "data_efficiency"
PAPER_CLASSICAL_ROOT = PAPER_RESULTS_ROOT / "classical_eval"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "train" / "oct.yaml"
DEFAULT_REAL_HARD_DIR = REPO_ROOT / "data" / "oct" / "eval" / "hard"
DEFAULT_BACKBONE = REPO_ROOT / "outputs" / "pretrain" / "dinov3_pretrain.pth"
DEFAULT_FULL_TRAIN_REFERENCE_CSV = PAPER_DATA_EFF_ROOT / "full_train_references.csv"


def _add_bundle_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint-root", type=Path, default=PAPER_CHECKPOINT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pretrained-backbone", type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument("--real-hard-dir", type=Path, default=DEFAULT_REAL_HARD_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-num-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")


def _add_classical_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--real-hard-dir", type=Path, default=DEFAULT_REAL_HARD_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--write-overlays", action="store_true")


def _add_ablation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pretrained-backbone", type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument("--real-hard-dir", type=Path, default=DEFAULT_REAL_HARD_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-num-workers", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--output-root", type=Path, default=PAPER_ABLATION_ROOT)
    parser.add_argument("--resume", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper replication entrypoint.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("main", help="Train the canonical DINOCT/UNet/FCBR checkpoints and evaluate clean-test + real-hard.")
    _add_bundle_args(p)

    p = sub.add_parser("low-data", help="Run the paper low-data comparison and write the data-efficiency outputs.")
    p.add_argument("--resume", action="store_true")

    p = sub.add_parser("ablations", help="Run the final single-component ablation table.")
    _add_ablation_args(p)

    p = sub.add_parser(
        "classical",
        help="Run the classical OCT baselines into outputs/paper_results/classical_eval.",
    )
    _add_classical_args(p)

    p = sub.add_parser(
        "robustness",
        help="Run the classical and learned-model robustness suite into outputs/paper_results/.",
    )
    _add_bundle_args(p)
    p.add_argument("--write-overlays", action="store_true")

    p = sub.add_parser("all", help="Run the full OCT paper pipeline: main checkpoints, classical baselines, robustness, ablations, and low-data.")
    _add_bundle_args(p)
    p.add_argument("--ablation-seeds", type=int, nargs="+", default=None)
    p.add_argument("--ablation-batch-size", type=int, default=256)
    p.add_argument("--write-overlays", action="store_true")

    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    print(f"[paper] running: {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _paper_checkpoint_paths(checkpoint_root: Path) -> dict[str, Path]:
    return {
        "dinoct": checkpoint_root / "post_train" / "fused_curve_best.pth",
        "unet": checkpoint_root / "unet" / "curve_best.pth",
        "fcbr": checkpoint_root / "fcbr" / "curve_best.pth",
    }


def _maybe_run(cmd: list[str], *, done_path: Path | None, resume: bool) -> None:
    if resume and done_path is not None and done_path.exists():
        return
    _run(cmd)


def _write_full_train_references(checkpoint_root: Path) -> None:
    DEFAULT_FULL_TRAIN_REFERENCE_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "method": "DINOCT",
            "checkpoint_path": str((checkpoint_root / "post_train" / "fused_curve_best.pth").relative_to(REPO_ROOT)),
            "summary_path": str((checkpoint_root / "eval" / "clean_test" / "summary.json").relative_to(REPO_ROOT)),
            "notes": "reused paper main checkpoint",
        },
        {
            "method": "UNET",
            "checkpoint_path": str((checkpoint_root / "unet" / "curve_best.pth").relative_to(REPO_ROOT)),
            "summary_path": str((checkpoint_root / "unet" / "eval" / "clean_test" / "summary.json").relative_to(REPO_ROOT)),
            "notes": "reused paper main checkpoint",
        },
        {
            "method": "FCBR",
            "checkpoint_path": str((checkpoint_root / "fcbr" / "curve_best.pth").relative_to(REPO_ROOT)),
            "summary_path": str((checkpoint_root / "fcbr" / "eval" / "clean_test" / "summary.json").relative_to(REPO_ROOT)),
            "notes": "reused paper main checkpoint",
        },
    ]
    with DEFAULT_FULL_TRAIN_REFERENCE_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["method", "checkpoint_path", "summary_path", "notes"])
        writer.writeheader()
        writer.writerows(rows)


def _run_classical_suite(args: argparse.Namespace) -> None:
    PAPER_CLASSICAL_ROOT.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, list[str]]] = [
        ("test", ["--split", "test"]),
        ("test_stripe_medium", ["--split", "test", "--corruption", "stripe", "--severity", "medium"]),
        ("test_stripe_severe", ["--split", "test", "--corruption", "stripe", "--severity", "severe"]),
        ("test_ghost_medium", ["--split", "test", "--corruption", "ghost", "--severity", "medium"]),
        ("test_ghost_severe", ["--split", "test", "--corruption", "ghost", "--severity", "severe"]),
        ("test_dropout_medium", ["--split", "test", "--corruption", "dropout", "--severity", "medium"]),
        ("test_dropout_severe", ["--split", "test", "--corruption", "dropout", "--severity", "severe"]),
        ("real_hard", ["--eval-dir", str(args.real_hard_dir)]),
    ]
    method_slugs = ["gf", "gf_b", "grad_sg", "grad_eng", "legacy_sobel_dc"]
    for output_name, extra_args in jobs:
        output_dir = PAPER_CLASSICAL_ROOT / output_name
        done_paths = [output_dir / method / "summary.json" for method in method_slugs]
        if args.resume and all(path.exists() for path in done_paths):
            continue
        cmd = [
            sys.executable,
            str(REPO_ROOT / "eval" / "classical" / "evaluate.py"),
            "--config", str(args.config),
            "--method", "all",
            "--output-dir", str(output_dir),
            *extra_args,
        ]
        if getattr(args, "write_overlays", False):
            cmd.append("--write-overlays")
        _run(cmd)


def _run_main_bundle(args: argparse.Namespace) -> None:
    checkpoint_root = args.checkpoint_root
    config = args.config
    backbone = args.pretrained_backbone
    if not backbone.exists():
        raise SystemExit(f"Missing pretrained backbone: {backbone}")

    checkpoint_root.mkdir(parents=True, exist_ok=True)

    dinoct_ckpt = checkpoint_root / "post_train" / "fused_curve_best.pth"
    _maybe_run(
        [
            sys.executable,
            "-m",
            "dinoct",
            "--config", str(config),
            "--output-dir", str(checkpoint_root),
            "--seed", str(args.seed),
            "--post-train-only",
            "--pretrained-backbone", str(backbone),
            "--num-workers", str(args.train_num_workers),
        ],
        done_path=dinoct_ckpt,
        resume=args.resume,
    )

    unet_ckpt = checkpoint_root / "unet" / "curve_best.pth"
    _maybe_run(
        [
            sys.executable,
            str(REPO_ROOT / "eval" / "train_learned_baseline.py"),
            "--config", str(config),
            "--model-type", "unet",
            "--output-dir", str(checkpoint_root / "unet"),
            "--seed", str(args.seed),
            "--batch-size", "12",
            "--num-workers", str(args.train_num_workers),
            "--device", str(args.device),
        ],
        done_path=unet_ckpt,
        resume=args.resume,
    )

    fcbr_ckpt = checkpoint_root / "fcbr" / "curve_best.pth"
    _maybe_run(
        [
            sys.executable,
            str(REPO_ROOT / "eval" / "train_learned_baseline.py"),
            "--config", str(config),
            "--model-type", "fcbr",
            "--output-dir", str(checkpoint_root / "fcbr"),
            "--seed", str(args.seed),
            "--batch-size", "32",
            "--num-workers", str(args.train_num_workers),
            "--device", str(args.device),
        ],
        done_path=fcbr_ckpt,
        resume=args.resume,
    )

    eval_jobs = [
        (
            dinoct_ckpt,
            checkpoint_root / "eval" / "clean_test" / "summary.json",
            [
                sys.executable,
                str(REPO_ROOT / "eval" / "evaluate_curve.py"),
                "--config", str(config),
                "--curve-ckpt", str(dinoct_ckpt),
                "--split", "test",
                "--output-dir", str(checkpoint_root / "eval" / "clean_test"),
                "--device", str(args.device),
                "--batch-size", str(args.eval_batch_size),
                "--num-workers", str(args.eval_num_workers),
            ],
        ),
        (
            dinoct_ckpt,
            checkpoint_root / "eval" / "real_hard" / "summary.json",
            [
                sys.executable,
                str(REPO_ROOT / "eval" / "evaluate_curve.py"),
                "--config", str(config),
                "--curve-ckpt", str(dinoct_ckpt),
                "--eval-dir", str(args.real_hard_dir),
                "--output-dir", str(checkpoint_root / "eval" / "real_hard"),
                "--device", str(args.device),
                "--batch-size", str(args.eval_batch_size),
                "--num-workers", str(args.eval_num_workers),
            ],
        ),
        (
            unet_ckpt,
            checkpoint_root / "unet" / "eval" / "clean_test" / "summary.json",
            [
                sys.executable,
                str(REPO_ROOT / "eval" / "evaluate_curve.py"),
                "--config", str(config),
                "--curve-ckpt", str(unet_ckpt),
                "--split", "test",
                "--output-dir", str(checkpoint_root / "unet" / "eval" / "clean_test"),
                "--device", str(args.device),
                "--batch-size", str(args.eval_batch_size),
                "--num-workers", str(args.eval_num_workers),
            ],
        ),
        (
            unet_ckpt,
            checkpoint_root / "unet" / "eval" / "real_hard" / "summary.json",
            [
                sys.executable,
                str(REPO_ROOT / "eval" / "evaluate_curve.py"),
                "--config", str(config),
                "--curve-ckpt", str(unet_ckpt),
                "--eval-dir", str(args.real_hard_dir),
                "--output-dir", str(checkpoint_root / "unet" / "eval" / "real_hard"),
                "--device", str(args.device),
                "--batch-size", str(args.eval_batch_size),
                "--num-workers", str(args.eval_num_workers),
            ],
        ),
        (
            fcbr_ckpt,
            checkpoint_root / "fcbr" / "eval" / "clean_test" / "summary.json",
            [
                sys.executable,
                str(REPO_ROOT / "eval" / "evaluate_curve.py"),
                "--config", str(config),
                "--curve-ckpt", str(fcbr_ckpt),
                "--split", "test",
                "--output-dir", str(checkpoint_root / "fcbr" / "eval" / "clean_test"),
                "--device", str(args.device),
                "--batch-size", str(args.eval_batch_size),
                "--num-workers", str(args.eval_num_workers),
            ],
        ),
        (
            fcbr_ckpt,
            checkpoint_root / "fcbr" / "eval" / "real_hard" / "summary.json",
            [
                sys.executable,
                str(REPO_ROOT / "eval" / "evaluate_curve.py"),
                "--config", str(config),
                "--curve-ckpt", str(fcbr_ckpt),
                "--eval-dir", str(args.real_hard_dir),
                "--output-dir", str(checkpoint_root / "fcbr" / "eval" / "real_hard"),
                "--device", str(args.device),
                "--batch-size", str(args.eval_batch_size),
                "--num-workers", str(args.eval_num_workers),
            ],
        ),
    ]
    for _ckpt, done_path, cmd in eval_jobs:
        _maybe_run(cmd, done_path=done_path, resume=args.resume)

    _write_full_train_references(checkpoint_root)


def _run_robustness_suite(args: argparse.Namespace) -> None:
    _run_classical_suite(args)
    ckpts = _paper_checkpoint_paths(args.checkpoint_root)
    missing = [str(path) for path in ckpts.values() if not path.exists()]
    if missing:
        raise SystemExit("Missing paper checkpoints for robustness suite:\n" + "\n".join(missing))

    PAPER_ROBUSTNESS_ROOT.mkdir(parents=True, exist_ok=True)
    for method, ckpt in ckpts.items():
        output_dir = PAPER_ROBUSTNESS_ROOT / method
        robust_summary = output_dir / "robustness_suite_summary.csv"
        real_hard_summary = output_dir / "real_hard" / "summary.json"
        if args.resume and robust_summary.exists() and real_hard_summary.exists():
            continue

        robust_cmd = [
            sys.executable,
            str(REPO_ROOT / "eval" / "paper" / "robustness.py"),
            "--name", method,
            "--config", str(args.config),
            "--curve-ckpt", str(ckpt),
            "--include-clean",
            "--output-dir", str(output_dir),
            "--device", str(args.device),
            "--batch-size", str(args.eval_batch_size),
            "--num-workers", str(args.eval_num_workers),
        ]
        if args.resume:
            robust_cmd.append("--resume")
        if getattr(args, "write_overlays", False):
            robust_cmd.append("--write-overlays")
        _run(robust_cmd)

        if not (args.resume and real_hard_summary.exists()):
            hard_cmd = [
                sys.executable,
                str(REPO_ROOT / "eval" / "evaluate_curve.py"),
                "--config", str(args.config),
                "--curve-ckpt", str(ckpt),
                "--eval-dir", str(args.real_hard_dir),
                "--output-dir", str(output_dir / "real_hard"),
                "--device", str(args.device),
                "--batch-size", str(args.eval_batch_size),
                "--num-workers", str(args.eval_num_workers),
            ]
            _run(hard_cmd)

    _run([
        sys.executable,
        str(REPO_ROOT / "eval" / "paper" / "robustness_table.py"),
        "--robustness-root", str(PAPER_ROBUSTNESS_ROOT),
        "--classical-root", str(PAPER_CLASSICAL_ROOT),
    ])
    _run([
        sys.executable,
        str(REPO_ROOT / "eval" / "paper" / "stress_analysis.py"),
        "--robustness-root", str(PAPER_ROBUSTNESS_ROOT),
        "--classical-real-hard-root", str(PAPER_CLASSICAL_ROOT / "real_hard"),
        "--output-dir", str(PAPER_ROBUSTNESS_ROOT),
    ])


def _run_ablations(args: argparse.Namespace) -> None:
    output_root = getattr(args, "output_root", PAPER_ABLATION_ROOT)
    batch_size = int(getattr(args, "batch_size", getattr(args, "ablation_batch_size", 256)))
    seeds = getattr(args, "seeds", None)
    if seeds is None:
        seeds = getattr(args, "ablation_seeds", None)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "eval" / "paper" / "component_ablation.py"),
        "--output-root", str(output_root),
        "--config", str(args.config),
        "--pretrained-backbone", str(args.pretrained_backbone),
        "--real-hard-dir", str(args.real_hard_dir),
        "--device", str(args.device),
        "--batch-size", str(batch_size),
        "--train-num-workers", str(args.train_num_workers),
        "--eval-batch-size", str(args.eval_batch_size),
        "--eval-num-workers", str(args.eval_num_workers),
        "--seed", str(args.seed),
    ]
    if seeds:
        cmd.extend(["--seeds", *[str(seed) for seed in seeds]])
    if args.resume:
        cmd.append("--resume")
    _run(cmd)


def _run_low_data(resume: bool) -> None:
    cmd = [sys.executable, str(REPO_ROOT / "eval" / "paper" / "low_data.py")]
    if resume:
        cmd.append("--resume")
    _run(cmd)


def main() -> None:
    args = parse_args()
    if args.command == "main":
        _run_main_bundle(args)
        return
    if args.command == "classical":
        _run_classical_suite(args)
        return
    if args.command == "robustness":
        _run_robustness_suite(args)
        return
    if args.command == "ablations":
        _run_ablations(args)
        return
    if args.command == "low-data":
        _run_low_data(resume=args.resume)
        return
    if args.command == "all":
        _run_main_bundle(args)
        _run_robustness_suite(args)
        _run_ablations(args)
        _run_low_data(resume=args.resume)
        return
    raise SystemExit(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
