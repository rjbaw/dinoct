#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

import yaml


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
PAPER_NORM_ABLATION_ROOT = PAPER_RESULTS_ROOT / "ablations" / "backbone_norm_sensitivity"
PAPER_LORA_ABLATION_ROOT = PAPER_RESULTS_ROOT / "ablations" / "lora_placement_sensitivity"
PAPER_DATA_EFF_ROOT = PAPER_RESULTS_ROOT / "data_efficiency"
PAPER_CLASSICAL_ROOT = PAPER_RESULTS_ROOT / "classical_eval"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "train" / "oct.yaml"
DEFAULT_REAL_HARD_DIR = REPO_ROOT / "data" / "oct" / "eval" / "hard"
DEFAULT_BACKBONE = REPO_ROOT / "outputs" / "pretrain" / "dinov3_pretrain.pth"
DEFAULT_FULL_TRAIN_REFERENCE_CSV = PAPER_DATA_EFF_ROOT / "full_train_references.csv"
PAPER_SEEDS = [0, 1, 2, 3, 4]  # every paper table (main + ablations) uses these five seeds


def _add_bundle_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint-root", type=Path, default=PAPER_CHECKPOINT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pretrained-backbone", type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument("--real-hard-dir", type=Path, default=DEFAULT_REAL_HARD_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=PAPER_SEEDS,
        help="Training seeds trained and evaluated for the pooled main + robustness tables.",
    )
    parser.add_argument("--train-num-workers", type=int, default=0)
    parser.add_argument(
        "--baseline-budget-mode",
        choices=["matched-samples", "legacy-steps"],
        default="matched-samples",
        help=(
            "matched-samples trains UNet/FCBR for the same approximate post-train samples seen as DINOCT; "
            "legacy-steps preserves the old 1500-step baseline runs."
        ),
    )
    parser.add_argument("--baseline-eval-every", type=int, default=500)
    parser.add_argument("--unet-steps", type=int, default=None)
    parser.add_argument("--fcbr-steps", type=int, default=None)
    parser.add_argument("--resume", action="store_true")


def _add_classical_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--real-hard-dir", type=Path, default=DEFAULT_REAL_HARD_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--write-overlays", action="store_true")


def _add_pretrain_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")


def _add_low_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pretrained-backbone", type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument("--real-hard-dir", type=Path, default=DEFAULT_REAL_HARD_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-num-workers", type=int, default=0)
    parser.add_argument(
        "--baseline-budget-mode",
        choices=["matched-samples", "legacy-steps"],
        default="matched-samples",
        help=(
            "matched-samples trains UNet/FCBR for the same approximate post-train samples seen as DINOCT; "
            "legacy-steps preserves the old baseline runs."
        ),
    )
    parser.add_argument("--baseline-eval-every", type=int, default=500)
    parser.add_argument("--unet-steps", type=int, default=None)
    parser.add_argument("--fcbr-steps", type=int, default=None)
    parser.add_argument("--resume", action="store_true")


def _add_ablation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pretrained-backbone", type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument("--real-hard-dir", type=Path, default=DEFAULT_REAL_HARD_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-num-workers", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=PAPER_SEEDS)
    parser.add_argument("--output-root", type=Path, default=PAPER_ABLATION_ROOT)
    parser.add_argument("--resume", action="store_true")


def _add_norm_ablation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pretrained-backbone", type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument("--real-hard-dir", type=Path, default=DEFAULT_REAL_HARD_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-num-workers", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=PAPER_SEEDS)
    parser.add_argument("--output-root", type=Path, default=PAPER_NORM_ABLATION_ROOT)
    parser.add_argument("--resume", action="store_true")


def _add_lora_ablation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pretrained-backbone", type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument("--real-hard-dir", type=Path, default=DEFAULT_REAL_HARD_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-num-workers", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=PAPER_SEEDS)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--output-root", type=Path, default=PAPER_LORA_ABLATION_ROOT)
    parser.add_argument("--resume", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper replication entrypoint.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("main", help="Train the canonical DINOCT/UNet/FCBR checkpoints and evaluate clean-test + real-hard.")
    _add_bundle_args(p)

    p = sub.add_parser("low-data", help="Run the paper low-data comparison and write the data-efficiency outputs.")
    _add_low_data_args(p)

    p = sub.add_parser("pretrain", help="Run the paper SSL pretraining stage into outputs/pretrain/.")
    _add_pretrain_args(p)

    p = sub.add_parser("ablations", help="Run the final single-component ablation table.")
    _add_ablation_args(p)

    p = sub.add_parser("norms", help="Run the backbone-normalization sensitivity table.")
    _add_norm_ablation_args(p)

    p = sub.add_parser("lora-placement", help="Run the LoRA-placement sensitivity table.")
    _add_lora_ablation_args(p)

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
    p.add_argument("--ablation-seeds", type=int, nargs="+", default=PAPER_SEEDS)
    p.add_argument("--ablation-batch-size", type=int, default=128)
    p.add_argument("--norm-ablation-seeds", type=int, nargs="+", default=PAPER_SEEDS)
    p.add_argument("--norm-ablation-batch-size", type=int, default=128)
    p.add_argument("--lora-ablation-seeds", type=int, nargs="+", default=PAPER_SEEDS)
    p.add_argument("--lora-ablation-batch-size", type=int, default=128)
    p.add_argument("--lora-ablation-variants", nargs="+", default=None)
    p.add_argument("--skip-norms", action="store_true")
    p.add_argument("--skip-lora-placement", action="store_true")
    p.add_argument("--write-overlays", action="store_true")

    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    print(f"[paper] running: {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _seed_root(checkpoint_root: Path, seed: int) -> Path:
    """Per-seed checkpoint/eval root, e.g. checkpoints/seed3/. Keeps the pooled
    localization/robustness tables one directory per training seed."""
    return checkpoint_root / f"seed{seed}"


def _paper_checkpoint_paths(checkpoint_root: Path, seed: int) -> dict[str, Path]:
    root = _seed_root(checkpoint_root, seed)
    return {
        "dinoct": root / "post_train" / "fused_curve_best.pth",
        "unet": root / "unet" / "curve_best.pth",
        "fcbr": root / "fcbr" / "curve_best.pth",
    }


def _maybe_run(cmd: list[str], *, done_path: Path | None, resume: bool) -> None:
    if resume and done_path is not None and done_path.exists():
        return
    _run(cmd)


def _run_pretrain(args: argparse.Namespace) -> None:
    done_path = args.output_dir / "pretrain" / "dinov3_pretrain.pth"
    _maybe_run(
        [
            sys.executable,
            "-m",
            "dinoct",
            "--config", str(args.config),
            "--output-dir", str(args.output_dir),
            "--seed", str(args.seed),
            "--num-workers", str(args.num_workers),
            "--post-train-steps", "0",
        ],
        done_path=done_path,
        resume=args.resume,
    )


def _write_full_train_references(checkpoint_root: Path) -> None:
    DEFAULT_FULL_TRAIN_REFERENCE_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "method": "DINOCT",
            "checkpoint_path": str((checkpoint_root / "post_train" / "fused_curve_best.pth").relative_to(REPO_ROOT)),
            "summary_path": str((checkpoint_root / "eval" / "clean_test" / "summary.json").relative_to(REPO_ROOT)),
            "real_hard_summary_path": str((checkpoint_root / "eval" / "real_hard" / "summary.json").relative_to(REPO_ROOT)),
            "notes": "reused paper main checkpoint",
        },
        {
            "method": "UNET",
            "checkpoint_path": str((checkpoint_root / "unet" / "curve_best.pth").relative_to(REPO_ROOT)),
            "summary_path": str((checkpoint_root / "unet" / "eval" / "clean_test" / "summary.json").relative_to(REPO_ROOT)),
            "real_hard_summary_path": str((checkpoint_root / "unet" / "eval" / "real_hard" / "summary.json").relative_to(REPO_ROOT)),
            "notes": "reused paper main checkpoint",
        },
        {
            "method": "FCBR",
            "checkpoint_path": str((checkpoint_root / "fcbr" / "curve_best.pth").relative_to(REPO_ROOT)),
            "summary_path": str((checkpoint_root / "fcbr" / "eval" / "clean_test" / "summary.json").relative_to(REPO_ROOT)),
            "real_hard_summary_path": str((checkpoint_root / "fcbr" / "eval" / "real_hard" / "summary.json").relative_to(REPO_ROOT)),
            "notes": "reused paper main checkpoint",
        },
    ]
    with DEFAULT_FULL_TRAIN_REFERENCE_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["method", "checkpoint_path", "summary_path", "real_hard_summary_path", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _baseline_steps_for_paper(args: argparse.Namespace, *, model_type: str, batch_size: int) -> int | None:
    explicit = {"unet": args.unet_steps, "fcbr": args.fcbr_steps}.get(model_type)
    if explicit is not None:
        return int(explicit)
    if str(getattr(args, "baseline_budget_mode", "matched-samples")) == "legacy-steps":
        return None

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    post_cfg = cfg.get("post_train", {})
    dinoct_steps = int(post_cfg.get("steps", 1500))
    dinoct_batch_size = int(post_cfg.get("batch_size", 128))
    target_samples = int(dinoct_steps * dinoct_batch_size)
    return int(math.ceil(target_samples / max(int(batch_size), 1)))


def _dinoct_eval_lora_args(config: Path) -> list[str]:
    """DINOCT robustness/real-hard evals pin the post-train LoRA alpha (matches the paper
    provenance: --lora-alpha from the config). Empty for the supervised baselines."""
    cfg = yaml.safe_load(Path(config).read_text()) or {}
    alpha = cfg.get("post_train", {}).get("lora_alpha")
    return ["--lora-alpha", str(int(alpha))] if alpha is not None else []


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


def _train_and_eval_seed(args: argparse.Namespace, seed: int) -> None:
    checkpoint_root = _seed_root(args.checkpoint_root, seed)
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
            "--seed", str(seed),
            "--post-train-only",
            "--pretrained-backbone", str(backbone),
            "--num-workers", str(args.train_num_workers),
        ],
        done_path=checkpoint_root / "post_train" / "fused_curve.pth",  # final artifact = training completed
        resume=args.resume,
    )

    unet_ckpt = checkpoint_root / "unet" / "curve_best.pth"
    unet_batch_size = 12
    unet_steps = _baseline_steps_for_paper(args, model_type="unet", batch_size=unet_batch_size)
    unet_cmd = [
        sys.executable,
        str(REPO_ROOT / "eval" / "train_learned_baseline.py"),
        "--config", str(config),
        "--model-type", "unet",
        "--output-dir", str(checkpoint_root / "unet"),
        "--seed", str(seed),
        "--batch-size", str(unet_batch_size),
        "--eval-every", str(args.baseline_eval_every),
        "--num-workers", str(args.train_num_workers),
        "--device", str(args.device),
    ]
    if unet_steps is not None:
        unet_cmd.extend(["--steps", str(unet_steps)])
    _maybe_run(
        unet_cmd,
        done_path=checkpoint_root / "unet" / "curve_final.pth",  # final artifact = training completed
        resume=args.resume,
    )

    fcbr_ckpt = checkpoint_root / "fcbr" / "curve_best.pth"
    fcbr_batch_size = 32
    fcbr_steps = _baseline_steps_for_paper(args, model_type="fcbr", batch_size=fcbr_batch_size)
    fcbr_cmd = [
        sys.executable,
        str(REPO_ROOT / "eval" / "train_learned_baseline.py"),
        "--config", str(config),
        "--model-type", "fcbr",
        "--output-dir", str(checkpoint_root / "fcbr"),
        "--seed", str(seed),
        "--batch-size", str(fcbr_batch_size),
        "--eval-every", str(args.baseline_eval_every),
        "--num-workers", str(args.train_num_workers),
        "--device", str(args.device),
    ]
    if fcbr_steps is not None:
        fcbr_cmd.extend(["--steps", str(fcbr_steps)])
    _maybe_run(
        fcbr_cmd,
        done_path=checkpoint_root / "fcbr" / "curve_final.pth",  # final artifact = training completed
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


def _run_main_bundle(args: argparse.Namespace) -> None:
    seeds = getattr(args, "seeds", None) or [args.seed]
    for seed in seeds:
        print(f"[paper] === main bundle: training seed {seed} ===")
        _train_and_eval_seed(args, seed)
    # Full-train references (used by the data-efficiency comparison) point at the first seed.
    _write_full_train_references(_seed_root(args.checkpoint_root, seeds[0]))


def _run_robustness_suite(args: argparse.Namespace) -> None:
    _run_classical_suite(args)
    seeds = getattr(args, "seeds", None) or [args.seed]
    dinoct_lora = _dinoct_eval_lora_args(args.config)
    PAPER_ROBUSTNESS_ROOT.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        ckpts = _paper_checkpoint_paths(args.checkpoint_root, seed)
        missing = [str(path) for path in ckpts.values() if not path.exists()]
        if missing:
            raise SystemExit(f"Missing seed-{seed} checkpoints for robustness suite:\n" + "\n".join(missing))

        for method, ckpt in ckpts.items():
            # robustness.py writes {output_dir}/{condition}/{name}/... so the seed lives one
            # level up: robustness/{method}/seed{N}/{cond}/{method}/per_recording_metrics.csv,
            # exactly the layout downstream aggregation consumes.
            output_dir = PAPER_ROBUSTNESS_ROOT / method / f"seed{seed}"
            robust_done = output_dir / "robustness_suite_summary.json"  # written only after ALL conditions
            real_hard_summary = output_dir / "real_hard" / "summary.json"
            extra = dinoct_lora if method == "dinoct" else []

            if not (args.resume and robust_done.exists()):
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
                    *extra,
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
                    *extra,
                ]
                _run(hard_cmd)

    # The paper tables are derived from the seed-partitioned per_recording_metrics.csv
    # layout written above (mean +- std across recordings of the seed-averaged metric).


def _run_paper_ablation(
    args: argparse.Namespace,
    *,
    script_name: str,
    default_output_root: Path,
    all_batch_attr: str,
    all_seeds_attr: str,
    all_variants_attr: str | None = None,
) -> None:
    output_root = getattr(args, "output_root", None) or default_output_root
    batch_size_arg = getattr(args, "batch_size", None)
    if batch_size_arg is None:
        batch_size_arg = getattr(args, all_batch_attr, 256)
    batch_size = int(batch_size_arg)

    seeds = getattr(args, "seeds", None)
    if seeds is None:
        seeds = getattr(args, all_seeds_attr, None)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "eval" / "paper" / script_name),
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

    if all_variants_attr is not None:
        variants = getattr(args, "variants", None)
        if variants is None:
            variants = getattr(args, all_variants_attr, None)
        if variants:
            cmd.extend(["--variants", *[str(variant) for variant in variants]])

    if args.resume:
        cmd.append("--resume")
    _run(cmd)


def _run_ablations(args: argparse.Namespace) -> None:
    _run_paper_ablation(
        args,
        script_name="component_ablation.py",
        default_output_root=PAPER_ABLATION_ROOT,
        all_batch_attr="ablation_batch_size",
        all_seeds_attr="ablation_seeds",
    )


def _run_norm_ablations(args: argparse.Namespace) -> None:
    _run_paper_ablation(
        args,
        script_name="backbone_norm_sensitivity.py",
        default_output_root=PAPER_NORM_ABLATION_ROOT,
        all_batch_attr="norm_ablation_batch_size",
        all_seeds_attr="norm_ablation_seeds",
    )


def _run_lora_ablations(args: argparse.Namespace) -> None:
    _run_paper_ablation(
        args,
        script_name="lora_placement_sensitivity.py",
        default_output_root=PAPER_LORA_ABLATION_ROOT,
        all_batch_attr="lora_ablation_batch_size",
        all_seeds_attr="lora_ablation_seeds",
        all_variants_attr="lora_ablation_variants",
    )


def _run_low_data(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "eval" / "paper" / "low_data.py"),
        "--config", str(args.config),
        "--pretrained-backbone", str(args.pretrained_backbone),
        "--real-hard-dir", str(args.real_hard_dir),
        "--seed", str(args.seed),
        "--train-num-workers", str(args.train_num_workers),
        "--eval-batch-size", str(args.eval_batch_size),
        "--eval-num-workers", str(args.eval_num_workers),
        "--device", str(args.device),
        "--baseline-budget-mode", str(args.baseline_budget_mode),
        "--baseline-eval-every", str(args.baseline_eval_every),
    ]
    if getattr(args, "unet_steps", None) is not None:
        cmd.extend(["--unet-steps", str(args.unet_steps)])
    if getattr(args, "fcbr_steps", None) is not None:
        cmd.extend(["--fcbr-steps", str(args.fcbr_steps)])
    if args.resume:
        cmd.append("--resume")
    _run(cmd)


def main() -> None:
    args = parse_args()
    if args.command == "main":
        _run_main_bundle(args)
        return
    if args.command == "pretrain":
        _run_pretrain(args)
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
    if args.command == "norms":
        _run_norm_ablations(args)
        return
    if args.command == "lora-placement":
        _run_lora_ablations(args)
        return
    if args.command == "low-data":
        _run_low_data(args)
        return
    if args.command == "all":
        _run_main_bundle(args)
        _run_robustness_suite(args)
        _run_ablations(args)
        if not getattr(args, "skip_norms", False):
            _run_norm_ablations(args)
        if not getattr(args, "skip_lora_placement", False):
            _run_lora_ablations(args)
        _run_low_data(args)
        return
    raise SystemExit(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
