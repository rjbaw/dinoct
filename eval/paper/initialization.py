"""Aggregate archived initialization scores into historical paper-result CSV/JSON.

This does not train, infer, or rescore labels. Input MAE must already average
B-scans within recordings, then weight recordings equally. Output std is the
population std across the three downstream seeds, not across recordings.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/paper_results/initialization"
DEFAULT_INPUT = DEFAULT_OUTPUT_ROOT / "paired_seed_results.csv"
SCORING_VIEW = "submitted_protocol_reproduction"
RECIPES = ("full_finetune", "paper_lora")
ARMS = ("scratch", "ssl", "meta")
CONDITIONS = ("clean", "stripe_severe", "ghost_severe", "dropout_severe", "stress")
SEEDS = (1, 2, 3)


def read_scores(path: Path) -> tuple[dict[tuple[str, str, int], tuple[float, ...]], str]:
    """Validate the historical seed cells and return scores plus the input hash."""
    payload = path.read_bytes()
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    required = {
        "recipe", "scoring_view", "condition", "seed", "eligible_bscans",
        "scientific_recordings", *(f"{arm}_mae_px" for arm in ARMS),
    }
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        missing = required.difference(reader.fieldnames or ())
        raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
    if len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError("Duplicate CSV column names")

    scores = {}
    for row in reader:
        if (row["scoring_view"] != SCORING_VIEW or row["recipe"] not in RECIPES
                or row["condition"] not in CONDITIONS):
            continue
        try:
            seed = int(row["seed"])
            counts = (int(row["eligible_bscans"]), int(row["scientific_recordings"]))
            values = tuple(float(row[f"{arm}_mae_px"]) for arm in ARMS)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric field at CSV line {reader.line_num}") from exc
        key = (row["recipe"], row["condition"], seed)
        if seed not in SEEDS:
            raise ValueError(f"Unexpected downstream seed in {key}; expected {SEEDS}")
        if key in scores:
            raise ValueError(f"Duplicate historical seed cell: {key}")
        expected_counts = (151, 1) if row["condition"] == "stress" else (1022, 61)
        if counts != expected_counts:
            raise ValueError(f"Historical counts for {key}: got {counts}, expected {expected_counts}")
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError(f"MAE must be finite and nonnegative in {key}")
        scores[key] = values

    expected = {(recipe, condition, seed) for recipe in RECIPES for condition in CONDITIONS for seed in SEEDS}
    missing = expected.difference(scores)
    if missing:
        raise ValueError(f"Missing historical seed cells ({len(missing)}): {sorted(missing)}")
    return scores, hashlib.sha256(payload).hexdigest()


def aggregate_scores(scores: dict[tuple[str, str, int], tuple[float, ...]]) -> list[dict]:
    """Summarize seed MAEs without changing the historical scoring population."""
    rows = []
    for recipe in RECIPES:
        for arm_index, arm in enumerate(ARMS):
            for condition in CONDITIONS:
                values = [scores[recipe, condition, seed][arm_index] for seed in SEEDS]
                rows.append({
                    "recipe": recipe,
                    "initialization": arm,
                    "condition": condition,
                    "scoring_view": SCORING_VIEW,
                    "num_seeds": len(SEEDS),
                    "seeds": "|".join(str(seed) for seed in SEEDS),
                    "labeled_bscans": 151 if condition == "stress" else 1022,
                    "recordings": 1 if condition == "stress" else 61,
                    "mae_px": statistics.mean(values),
                    "mae_px_seed_std": statistics.pstdev(values),
                })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT,
                        help="archived per-seed CSV (default: released initialization scores)")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                        help="destination for initialization_summary.csv and .json")
    parser.add_argument("--resume", action="store_true", help="reuse only identical existing summaries")
    parser.add_argument("--check-only", action="store_true", help="validate inputs and destinations without writing")
    args = parser.parse_args(argv)
    try:
        scores, _digest = read_scores(args.input_csv)
        rows = aggregate_scores(scores)
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        outputs = {
            args.output_root / "initialization_summary.csv": buffer.getvalue().encode("utf-8"),
            args.output_root / "initialization_summary.json": (json.dumps(rows, indent=2, allow_nan=False) + "\n").encode("utf-8"),
        }
        existing = set()
        # Validate both destinations before writing either summary.
        for path, payload in outputs.items():
            if path.exists() or path.is_symlink():
                if not args.resume:
                    raise FileExistsError(f"File exists: {path}; use --resume or a new output root")
                if path.read_bytes() != payload:
                    raise ValueError(f"Existing summary differs: {path}; use a new output root")
                existing.add(path)
            else:
                for parent in path.parents:
                    if parent.exists() or parent.is_symlink():
                        if not parent.is_dir():
                            raise NotADirectoryError(f"Output parent is not a directory: {parent}")
                        if not os.access(parent, os.W_OK | os.X_OK):
                            raise PermissionError(f"Output parent is not writable/searchable: {parent}")
                        break
        if args.check_only:
            print("[initialization] inputs and destinations validated")
            return 0
        args.output_root.mkdir(parents=True, exist_ok=True)
        for path, payload in outputs.items():
            if path in existing:
                print(f"[initialization] reused {path}")
            else:
                with path.open("xb") as handle:
                    handle.write(payload)
                print(f"[initialization] wrote {path}")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
