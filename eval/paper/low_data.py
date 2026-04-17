#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


METHOD_LABELS = {
    "dinoct": "DINOCT",
    "unet": "UNET",
    "fcbr": "FCBR",
}

BUDGET_SPECS = [
    (50, "50 images"),
    (100, "100 images"),
    (250, "250 images"),
    (500, "500 images"),
    (1000, "1000 images"),
    (None, "full train set"),
]

def _find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").exists() and (candidate / "dinoct").is_dir():
            return candidate
    raise RuntimeError("Could not locate repo root from script path.")


REPO_ROOT = _find_repo_root()
DEFAULT_CONFIG = REPO_ROOT / "configs" / "train" / "oct.yaml"
DEFAULT_EXTRA = REPO_ROOT / "data" / "oct" / "extra"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "paper_results" / "data_efficiency"
DEFAULT_MANIFEST = REPO_ROOT / "outputs" / "paper_results" / "data_efficiency_manifest.csv"
DEFAULT_TABLE_SCRIPT = REPO_ROOT / "eval" / "paper" / "data_efficiency_tables.py"
DEFAULT_BACKBONE = REPO_ROOT / "outputs" / "pretrain" / "dinov3_pretrain.pth"
DEFAULT_FULL_TRAIN_REFERENCE_CSV = REPO_ROOT / "outputs" / "paper_results" / "data_efficiency" / "full_train_references.csv"
DEFAULT_FULL_TRAIN_REFS = {
    "DINOCT": {
        "checkpoint_path": REPO_ROOT / "outputs" / "paper_results" / "checkpoints" / "post_train" / "fused_curve_best.pth",
        "summary_path": REPO_ROOT / "outputs" / "paper_results" / "checkpoints" / "eval" / "clean_test" / "summary.json",
        "notes": "reused paper main checkpoint",
    },
    "UNET": {
        "checkpoint_path": REPO_ROOT / "outputs" / "paper_results" / "checkpoints" / "unet" / "curve_best.pth",
        "summary_path": REPO_ROOT / "outputs" / "paper_results" / "checkpoints" / "unet" / "eval" / "clean_test" / "summary.json",
        "notes": "reused paper main checkpoint",
    },
    "FCBR": {
        "checkpoint_path": REPO_ROOT / "outputs" / "paper_results" / "checkpoints" / "fcbr" / "curve_best.pth",
        "summary_path": REPO_ROOT / "outputs" / "paper_results" / "checkpoints" / "fcbr" / "eval" / "clean_test" / "summary.json",
        "notes": "reused paper main checkpoint",
    },
}


def _slug(text: str) -> str:
    out = []
    for ch in text.strip().lower():
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def _parse_budget_token(token: str) -> tuple[int | None, str]:
    value = token.strip().lower()
    if value in {"full", "full_train_set", "full-train-set", "all"}:
        return None, "full train set"
    try:
        n = int(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported budget token: {token!r}") from exc
    return n, f"{n} images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run low-data train+eval experiments for DINOCT, UNET, and FCBR, then fill paper tables."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["dinoct", "unet", "fcbr"],
        choices=sorted(METHOD_LABELS),
        help="Which methods to run.",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        default=["50", "100", "250", "500", "1000", "full"],
        help="Train labeled-image budgets to evaluate. Use integers or 'full'.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-extra", type=Path, default=DEFAULT_EXTRA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pretrained-backbone", type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument("--seed", type=int, default=0, help="Training seed for all methods.")
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=0,
        help="Seed used to choose the labeled train subset for each low-data budget.",
    )
    parser.add_argument("--train-num-workers", type=int, default=0)
    parser.add_argument("--eval-num-workers", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--dinoct-batch-size", type=int, default=None)
    parser.add_argument("--baseline-batch-size", type=int, default=None)
    parser.add_argument("--unet-batch-size", type=int, default=12)
    parser.add_argument("--fcbr-batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=None, help="Override training/post-train steps for all methods.")
    parser.add_argument("--device", default="auto", help="Eval/baseline training device: auto, cpu, cuda, or mps.")
    parser.add_argument("--resume", action="store_true", help="Skip completed runs if their summary.json already exists.")
    parser.add_argument(
        "--overwrite-subsets",
        action="store_true",
        help="Regenerate budget subset splits even if a persisted subset draw already exists.",
    )
    parser.add_argument(
        "--full-train-reference-csv",
        type=Path,
        default=DEFAULT_FULL_TRAIN_REFERENCE_CSV,
        help="Optional CSV with columns: method,checkpoint_path,summary_path,notes. Used only for the 'full train set' row.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands but do not run them.")
    return parser.parse_args()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _budget_selection(rows: list[dict[str, str]], budget: int, seed: int) -> tuple[set[str], int]:
    train_labeled = [
        row
        for row in rows
        if row.get("split", "").strip().lower() == "train" and int(row.get("num_labeled_raw", "0") or 0) > 0
    ]
    if not train_labeled:
        return set(), 0

    rng = random.Random(int(seed))
    shuffled = train_labeled[:]
    rng.shuffle(shuffled)
    counts = [int(row.get("num_labeled_raw", "0") or 0) for row in shuffled]
    group_ids = [str(row.get("group_id", "")).strip() for row in shuffled]
    total = sum(counts)
    if budget >= total:
        return set(group_ids), total

    prev_sum = [-1] * (total + 1)
    prev_idx = [-1] * (total + 1)
    prev_sum[0] = 0
    for idx, count in enumerate(counts):
        for s in range(total - count, -1, -1):
            if prev_sum[s] == -1:
                continue
            ns = s + count
            if prev_sum[ns] == -1:
                prev_sum[ns] = s
                prev_idx[ns] = idx

    reachable = [s for s in range(total + 1) if prev_sum[s] != -1]
    best_sum = min(reachable, key=lambda s: (abs(s - budget), s < budget, s))
    selected_ids: set[str] = set()
    cursor = best_sum
    while cursor > 0:
        idx = prev_idx[cursor]
        if idx < 0:
            break
        selected_ids.add(group_ids[idx])
        cursor = prev_sum[cursor]
    return selected_ids, best_sum


def _materialize_budget_extra(
    *,
    base_extra: Path,
    output_extra: Path,
    budget: int | None,
    subset_seed: int,
    overwrite_existing: bool,
) -> dict[str, Any]:
    manifest_src = base_extra / "manifest.csv"
    splits_src = base_extra / "splits.csv"
    if not manifest_src.exists():
        raise FileNotFoundError(manifest_src)
    if not splits_src.exists():
        raise FileNotFoundError(splits_src)

    output_extra.mkdir(parents=True, exist_ok=True)
    manifest_dst = output_extra / "manifest.csv"
    splits_dst = output_extra / "splits.csv"
    summary_dst = output_extra / "budget_summary.json"
    base_manifest_sha256 = _sha256_file(manifest_src)
    base_splits_sha256 = _sha256_file(splits_src)

    if not overwrite_existing and manifest_dst.exists() and splits_dst.exists() and summary_dst.exists():
        existing = json.loads(summary_dst.read_text())
        same_budget = existing.get("budget_target_labeled_images") == budget
        same_seed = int(existing.get("subset_seed", existing.get("seed", -1))) == int(subset_seed)
        same_manifest = existing.get("base_manifest_sha256") == base_manifest_sha256
        same_splits = existing.get("base_splits_sha256") == base_splits_sha256
        legacy_summary = same_budget and same_seed and not existing.get("base_manifest_sha256") and not existing.get("base_splits_sha256")
        if same_budget and same_seed and same_manifest and same_splits:
            return existing
        if legacy_summary:
            existing["subset_seed"] = int(subset_seed)
            existing["seed"] = int(subset_seed)
            existing["base_manifest_sha256"] = base_manifest_sha256
            existing["base_splits_sha256"] = base_splits_sha256
            existing["splits_csv"] = str(splits_dst.resolve())
            existing["manifest_csv"] = str(manifest_dst.resolve())
            summary_dst.write_text(json.dumps(existing, indent=2) + "\n")
            return existing
        raise RuntimeError(
            "Existing low-data subset draw does not match the requested configuration. "
            f"budget={budget!r}, subset_seed={subset_seed}, output_extra={output_extra}. "
            "Use --overwrite-subsets to regenerate it."
        )

    shutil.copy2(manifest_src, manifest_dst)

    rows = _read_csv_rows(splits_src)
    fieldnames = list(rows[0].keys()) if rows else []
    train_labeled_total = sum(
        int(row.get("num_labeled_raw", "0") or 0)
        for row in rows
        if row.get("split", "").strip().lower() == "train"
    )

    if budget is None:
        selected_group_ids = {
            str(row.get("group_id", "")).strip()
            for row in rows
            if row.get("split", "").strip().lower() == "train" and int(row.get("num_labeled_raw", "0") or 0) > 0
        }
        actual_labeled = train_labeled_total
        out_rows = rows
    else:
        selected_group_ids, actual_labeled = _budget_selection(rows, budget, subset_seed)
        out_rows = []
        for row in rows:
            split = str(row.get("split", "")).strip().lower()
            labeled = int(row.get("num_labeled_raw", "0") or 0)
            new_row = dict(row)
            if split == "train" and labeled > 0:
                new_row["split"] = "train" if str(row.get("group_id", "")).strip() in selected_group_ids else "unused"
            out_rows.append(new_row)

    _write_csv_rows(splits_dst, fieldnames, out_rows)

    summary = {
        "budget_target_labeled_images": budget,
        "budget_actual_labeled_images": actual_labeled,
        "full_train_labeled_images": train_labeled_total,
        "selected_group_ids": sorted(selected_group_ids),
        "subset_seed": int(subset_seed),
        "seed": int(subset_seed),
        "base_manifest_sha256": base_manifest_sha256,
        "base_splits_sha256": base_splits_sha256,
        "splits_csv": str(splits_dst.resolve()),
        "manifest_csv": str(manifest_dst.resolve()),
    }
    summary_dst.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _write_config_with_extra(*, base_config: Path, output_config: Path, root_path: Path, extra_path: Path) -> None:
    cfg = yaml.safe_load(base_config.read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config: {base_config}")
    cfg.setdefault("train", {})["dataset_path"] = f"OCT:root={root_path.resolve()}:extra={extra_path.resolve()}"
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(yaml.safe_dump(cfg, sort_keys=False))


def _run(cmd: list[str], *, dry_run: bool) -> None:
    pretty = " ".join(str(part) for part in cmd)
    print(f"[lowdata] running: {pretty}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def _update_manifest(*, manifest_path: Path, method_label: str, budget_label: str, summary_path: Path, notes: str) -> None:
    rows = _read_csv_rows(manifest_path) if manifest_path.exists() else []
    if not rows:
        rows = [
            {"method": method, "budget": budget, "summary_path": "", "notes": ""}
            for method in ("DINOCT", "UNET", "FCBR")
            for _budget, budget in BUDGET_SPECS
        ]
    updated = False
    for row in rows:
        if str(row.get("method", "")).strip().upper() == method_label and str(row.get("budget", "")).strip() == budget_label:
            row["summary_path"] = _rel(summary_path)
            row["notes"] = notes
            updated = True
            break
    if not updated:
        rows.append({"method": method_label, "budget": budget_label, "summary_path": _rel(summary_path), "notes": notes})
    _write_csv_rows(manifest_path, ["method", "budget", "summary_path", "notes"], rows)


def _append_run_record(path: Path, record: dict[str, Any]) -> None:
    fieldnames = [
        "method",
        "budget_label",
        "budget_target_labeled_images",
        "budget_actual_labeled_images",
        "subset_seed",
        "run_dir",
        "summary_path",
        "checkpoint_path",
        "notes",
    ]
    rows = _read_csv_rows(path) if path.exists() else []
    rows = [row for row in rows if not (row.get("method") == record["method"] and row.get("budget_label") == record["budget_label"])]
    rows.append(record)
    _write_csv_rows(path, fieldnames, rows)


def _expected_checkpoint_path(method: str, run_dir: Path) -> Path:
    if method == "dinoct":
        return run_dir / "post_train" / "fused_curve_best.pth"
    return run_dir / "curve_best.pth"


def _load_full_train_references(path: Path) -> dict[str, dict[str, str]]:
    refs: dict[str, dict[str, str]] = {}
    if path.exists():
        for row in _read_csv_rows(path):
            method = str(row.get("method", "")).strip().upper()
            if not method:
                continue
            refs[method] = {
                "checkpoint_path": str(row.get("checkpoint_path", "")).strip(),
                "summary_path": str(row.get("summary_path", "")).strip(),
                "notes": str(row.get("notes", "")).strip(),
            }
    for method, spec in DEFAULT_FULL_TRAIN_REFS.items():
        if method in refs:
            continue
        ckpt = Path(spec["checkpoint_path"])
        summary = Path(spec["summary_path"])
        if ckpt.exists():
            refs[method] = {
                "checkpoint_path": _rel(ckpt),
                "summary_path": _rel(summary) if summary.exists() else "",
                "notes": str(spec["notes"]),
            }
    return refs


def _train_dinoct(*, run_dir: Path, config_path: Path, args: argparse.Namespace) -> Path:
    ckpt = run_dir / "post_train" / "fused_curve_best.pth"
    if args.resume and ckpt.exists():
        return ckpt
    cmd = [
        sys.executable,
        "-m",
        "dinoct",
        "--config",
        str(config_path),
        "--output-dir",
        str(run_dir),
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.train_num_workers),
        "--post-train-only",
        "--pretrained-backbone",
        str(args.pretrained_backbone),
    ]
    if args.steps is not None:
        cmd.extend(["--post-train-steps", str(args.steps)])
    if args.dinoct_batch_size is not None:
        cmd.extend(["--post-train-batch-size", str(args.dinoct_batch_size)])
    _run(cmd, dry_run=args.dry_run)
    return ckpt


def _baseline_batch_size(model_type: str, args: argparse.Namespace) -> int:
    if model_type == "unet":
        return int(args.unet_batch_size if args.unet_batch_size is not None else 4)
    if model_type == "fcbr":
        return int(args.fcbr_batch_size if args.fcbr_batch_size is not None else (args.baseline_batch_size or 64))
    return int(args.baseline_batch_size or 64)


def _train_baseline(*, model_type: str, run_dir: Path, config_path: Path, args: argparse.Namespace) -> Path:
    ckpt = run_dir / "curve_best.pth"
    if args.resume and ckpt.exists():
        return ckpt

    cmd = [
        sys.executable,
        str(REPO_ROOT / "eval" / "train_learned_baseline.py"),
        "--config",
        str(config_path),
        "--model-type",
        model_type,
        "--output-dir",
        str(run_dir),
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.train_num_workers),
        "--device",
        str(args.device),
        "--batch-size",
        str(_baseline_batch_size(model_type, args)),
    ]
    if args.steps is not None:
        cmd.extend(["--steps", str(args.steps)])
    _run(cmd, dry_run=args.dry_run)
    return ckpt


def _evaluate_curve(*, ckpt_path: Path, output_dir: Path, args: argparse.Namespace) -> Path:
    summary_path = output_dir / "summary.json"
    if args.resume and summary_path.exists():
        return summary_path
    cmd = [
        sys.executable,
        str(REPO_ROOT / "eval" / "evaluate_curve.py"),
        "--curve-ckpt",
        str(ckpt_path),
        "--split",
        "test",
        "--output-dir",
        str(output_dir),
        "--device",
        str(args.device),
        "--batch-size",
        str(args.eval_batch_size),
        "--num-workers",
        str(args.eval_num_workers),
    ]
    _run(cmd, dry_run=args.dry_run)
    return summary_path


def _resolve_reference_path(raw: str) -> Path | None:
    value = str(raw).strip()
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _materialize_reference_full_train(
    *,
    method_label: str,
    run_dir: Path,
    reference: dict[str, str],
    args: argparse.Namespace,
) -> tuple[Path, Path | None, str]:
    summary_ref = _resolve_reference_path(reference.get("summary_path", ""))
    ckpt_ref = _resolve_reference_path(reference.get("checkpoint_path", ""))
    ref_notes = str(reference.get("notes", "")).strip()

    target_eval_dir = run_dir / "eval" / "clean_test"
    target_summary = target_eval_dir / "summary.json"

    if ckpt_ref is not None and ckpt_ref.exists():
        eval_args = argparse.Namespace(**vars(args))
        eval_args.resume = False
        summary_path = _evaluate_curve(ckpt_path=ckpt_ref, output_dir=target_eval_dir, args=eval_args)
        return summary_path, ckpt_ref, ref_notes

    if args.resume and target_summary.exists():
        return target_summary, ckpt_ref if ckpt_ref and ckpt_ref.exists() else None, ref_notes

    if summary_ref is not None and summary_ref.exists():
        return summary_ref, ckpt_ref if ckpt_ref and ckpt_ref.exists() else None, ref_notes

    raise FileNotFoundError(
        f"Full-train reference for {method_label} is invalid: checkpoint={ckpt_ref!s}, summary={summary_ref!s}"
    )


def main() -> None:
    args = parse_args()
    budgets = [_parse_budget_token(token) for token in args.budgets]
    output_root = args.output_root
    split_root = output_root / "splits"
    runs_csv = output_root / "low_data_runs.csv"
    root_path = REPO_ROOT / "data" / "oct"

    if not args.dry_run:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        output_root.mkdir(parents=True, exist_ok=True)

    full_train_refs = _load_full_train_references(args.full_train_reference_csv)

    for budget_value, budget_label in budgets:
        budget_slug = _slug(budget_label)
        extra_dir = split_root / budget_slug / "extra"
        budget_summary = _materialize_budget_extra(
            base_extra=args.base_extra,
            output_extra=extra_dir,
            budget=budget_value,
            subset_seed=args.subset_seed,
            overwrite_existing=args.overwrite_subsets,
        )
        notes = f"actual_labeled={budget_summary['budget_actual_labeled_images']}; subset_seed={args.subset_seed}"

        for method in args.methods:
            method_label = METHOD_LABELS[method]
            run_dir = output_root / method / budget_slug
            config_path = run_dir / "train_config.yaml"
            _write_config_with_extra(
                base_config=args.config,
                output_config=config_path,
                root_path=root_path,
                extra_path=extra_dir,
            )

            summary_path = run_dir / "eval" / "clean_test" / "summary.json"
            ckpt_path_existing = _expected_checkpoint_path(method, run_dir)
            ckpt_path: Path | None = ckpt_path_existing if ckpt_path_existing.exists() else None
            extra_notes = ""

            if budget_value is None and method_label in full_train_refs:
                summary_path, ckpt_path, extra_notes = _materialize_reference_full_train(
                    method_label=method_label,
                    run_dir=run_dir,
                    reference=full_train_refs[method_label],
                    args=args,
                )
            elif args.resume and summary_path.exists():
                pass
            elif method == "dinoct":
                ckpt_path = _train_dinoct(run_dir=run_dir, config_path=config_path, args=args)
                summary_path = _evaluate_curve(ckpt_path=ckpt_path, output_dir=run_dir / "eval" / "clean_test", args=args)
            else:
                ckpt_path = _train_baseline(model_type=method, run_dir=run_dir, config_path=config_path, args=args)
                summary_path = _evaluate_curve(ckpt_path=ckpt_path, output_dir=run_dir / "eval" / "clean_test", args=args)

            notes_full = notes if not extra_notes else f"{notes}; {extra_notes}"
            if not args.dry_run:
                _update_manifest(
                    manifest_path=args.manifest,
                    method_label=method_label,
                    budget_label=budget_label,
                    summary_path=summary_path,
                    notes=notes_full,
                )
                _append_run_record(
                    runs_csv,
                    {
                        "method": method_label,
                        "budget_label": budget_label,
                        "budget_target_labeled_images": "" if budget_value is None else int(budget_value),
                        "budget_actual_labeled_images": int(budget_summary["budget_actual_labeled_images"]),
                        "subset_seed": int(args.subset_seed),
                        "run_dir": _rel(run_dir),
                        "summary_path": _rel(summary_path),
                        "checkpoint_path": "" if ckpt_path is None else _rel(ckpt_path),
                        "notes": notes_full,
                    },
                )

    if not args.dry_run:
        cmd = [sys.executable, str(DEFAULT_TABLE_SCRIPT), "--manifest", str(args.manifest)]
        _run(cmd, dry_run=False)
        print(f"[lowdata] wrote manifest: {args.manifest}")
        print(f"[lowdata] wrote run index: {runs_csv}")


if __name__ == "__main__":
    main()
