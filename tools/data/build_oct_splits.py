#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path


def _find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").exists() and (candidate / "dinoct").is_dir():
            return candidate
    raise RuntimeError("Could not locate repo root from script path.")


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dinoct.oct_metadata import (  # noqa: E402
    default_acquisition_mode,
    read_manifest_csv,
    write_splits_csv,
)

EXCLUDED_CONTINUOUS_RECORDINGS = {
    "continuous:new",
    "continuous:new_capture",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create recording-level train/val/test splits for the OCT manifest."
    )
    parser.add_argument("--dir", type=Path, default=Path("data/oct"), help="Dataset root containing extra/manifest.csv")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest CSV path. Defaults to <dir>/extra/manifest.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Split CSV path. Defaults to <dir>/extra/splits.csv",
    )
    parser.add_argument("--train-frac", type=float, default=0.7, help="Target labeled-sample fraction for train")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Target labeled-sample fraction for val")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for group assignment")
    return parser.parse_args()


def _validate_fracs(train_frac: float, val_frac: float) -> dict[str, float]:
    train_f = float(train_frac)
    val_f = float(val_frac)
    test_f = 1.0 - train_f - val_f
    if train_f <= 0.0 or val_f <= 0.0 or test_f <= 0.0:
        raise ValueError("Expected positive train/val/test fractions with train + val < 1.")
    return {"train": train_f, "val": val_f, "test": test_f}


def _collect_group_stats(manifest_path: Path) -> dict[str, dict[str, object]]:
    rows = read_manifest_csv(manifest_path)
    groups: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "group_id": "",
            "num_labeled_raw": 0,
            "num_raw": 0,
            "num_background": 0,
            "num_paired_labeled_raw": 0,
            "modalities": set(),
        }
    )

    for row in rows:
        group = groups[row.group_id]
        group["group_id"] = row.group_id
        modalities = group["modalities"]
        assert isinstance(modalities, set)
        if row.modality:
            modalities.add(row.modality)

        if row.kind == "raw":
            group["num_raw"] = int(group["num_raw"]) + 1
            if row.code == 1:
                group["num_labeled_raw"] = int(group["num_labeled_raw"]) + 1
                if row.paired_background_relpath:
                    group["num_paired_labeled_raw"] = int(group["num_paired_labeled_raw"]) + 1
        elif row.kind == "background":
            group["num_background"] = int(group["num_background"]) + 1

    return groups


def _resolve_recordings(
    groups: dict[str, dict[str, object]],
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, str]]]:
    recordings: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "num_labeled_raw": 0,
            "num_raw": 0,
            "num_background": 0,
            "num_paired_labeled_raw": 0,
            "modalities": set(),
            "group_ids": [],
            "acquisition_modes": set(),
            "primary_acquisition_mode": "",
        }
    )
    group_map: dict[str, dict[str, str]] = {}

    for group_id, stats in groups.items():
        recording_id = group_id
        acquisition_mode = default_acquisition_mode(group_id)
        group_map[group_id] = {
            "recording_id": recording_id,
            "acquisition_mode": acquisition_mode,
        }

        rec = recordings[recording_id]
        rec["num_labeled_raw"] = int(rec["num_labeled_raw"]) + int(stats["num_labeled_raw"])
        rec["num_raw"] = int(rec["num_raw"]) + int(stats["num_raw"])
        rec["num_background"] = int(rec["num_background"]) + int(stats["num_background"])
        rec["num_paired_labeled_raw"] = int(rec["num_paired_labeled_raw"]) + int(stats["num_paired_labeled_raw"])

        rec_modalities = rec["modalities"]
        rec_group_ids = rec["group_ids"]
        rec_modes = rec["acquisition_modes"]
        src_modalities = stats["modalities"]
        assert isinstance(rec_modalities, set)
        assert isinstance(rec_group_ids, list)
        assert isinstance(rec_modes, set)
        assert isinstance(src_modalities, set)
        rec_modalities.update(str(item) for item in src_modalities)
        rec_group_ids.append(group_id)
        rec_modes.add(acquisition_mode)
        rec["primary_acquisition_mode"] = acquisition_mode if len(rec_modes) == 1 else "mixed"

    return recordings, group_map


def _assign_recording_subset(
    eligible: list[tuple[str, dict[str, object]]],
    fractions: dict[str, float],
    seed: int,
    weight_key: str,
) -> dict[str, str]:
    if len(eligible) < 3:
        return {recording_id: "train" for recording_id, _stats in eligible}

    total_weight = sum(int(stats[weight_key]) for _, stats in eligible)
    targets = {split: total_weight * frac for split, frac in fractions.items()}
    counts = {split: 0 for split in fractions}

    rng = random.Random(int(seed))
    rng.shuffle(eligible)
    eligible.sort(key=lambda item: int(item[1][weight_key]), reverse=True)

    assignments: dict[str, str] = {}
    split_names = list(fractions)

    for split_name, (recording_id, stats) in zip(split_names, eligible):
        assignments[recording_id] = split_name
        counts[split_name] += int(stats[weight_key])

    for recording_id, stats in eligible[len(split_names) :]:
        sample_weight = int(stats[weight_key])
        best_split = max(
            split_names,
            key=lambda split_name: (
                targets[split_name] - counts[split_name],
                -counts[split_name],
                split_name == "train",
            ),
        )
        assignments[recording_id] = best_split
        counts[best_split] += sample_weight

    return assignments


def _policy_split_for_recording(recording_id: str, stats: dict[str, object]) -> str | None:
    mode = str(stats.get("primary_acquisition_mode", "") or "other")
    if recording_id in EXCLUDED_CONTINUOUS_RECORDINGS:
        return "exclude"
    if mode == "continuous_sequence":
        return "train"
    if mode in {"background", "c_scan"}:
        return None
    return "exclude"


def _assign_recordings(
    recordings: dict[str, dict[str, object]],
    fractions: dict[str, float],
    seed: int,
) -> dict[str, str]:
    randomized = [
        (recording_id, stats)
        for recording_id, stats in recordings.items()
        if _policy_split_for_recording(recording_id, stats) is None and int(stats["num_labeled_raw"]) > 0
    ]
    if len(randomized) < 3:
        raise ValueError(
            "Need at least 3 labeled C-scan recordings to build train/val/test splits; "
            f"found {len(randomized)}."
        )

    assignments: dict[str, str] = {}
    assignments.update(_assign_recording_subset(randomized, fractions, seed=seed, weight_key="num_labeled_raw"))

    background_groups = [
        (recording_id, stats)
        for recording_id, stats in recordings.items()
        if str(stats.get("primary_acquisition_mode", "") or "") == "background" and int(stats["num_background"]) > 0
    ]
    if background_groups:
        assignments.update(_assign_recording_subset(background_groups, fractions, seed=seed + 1, weight_key="num_background"))

    for recording_id, stats in recordings.items():
        forced_split = _policy_split_for_recording(recording_id, stats)
        if forced_split is not None:
            assignments[recording_id] = forced_split
            continue
        assignments.setdefault(recording_id, "train")

    return assignments


def _build_rows(
    groups: dict[str, dict[str, object]],
    group_map: dict[str, dict[str, str]],
    recording_assignments: dict[str, str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for group_id in sorted(group_map):
        stats = groups[group_id]
        mapping = group_map[group_id]
        recording_id = mapping["recording_id"]
        modalities = stats["modalities"]
        assert isinstance(modalities, set)
        rows.append(
            {
                "group_id": group_id,
                "recording_id": recording_id,
                "split": recording_assignments[recording_id],
                "acquisition_mode": mapping["acquisition_mode"],
                "num_labeled_raw": str(int(stats["num_labeled_raw"])),
                "num_raw": str(int(stats["num_raw"])),
                "num_background": str(int(stats["num_background"])),
                "num_paired_labeled_raw": str(int(stats["num_paired_labeled_raw"])),
                "modalities": "|".join(sorted(str(item) for item in modalities)),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    dataset_root = args.dir.expanduser()
    manifest_path = args.manifest.expanduser() if args.manifest else dataset_root / "extra" / "manifest.csv"
    output_path = args.output.expanduser() if args.output else dataset_root / "extra" / "splits.csv"

    fractions = _validate_fracs(args.train_frac, args.val_frac)
    groups = _collect_group_stats(manifest_path)
    recordings, group_map = _resolve_recordings(groups)
    assignments = _assign_recordings(recordings, fractions, seed=args.seed)
    rows = _build_rows(groups, group_map, assignments)
    write_splits_csv(output_path, rows)

    print(f"[splits] wrote {len(rows)} group assignments to {output_path}")
    split_totals = defaultdict(int)
    labeled_totals = defaultdict(int)
    recording_totals = defaultdict(set)
    for row in rows:
        split_name = row["split"]
        split_totals[split_name] += 1
        labeled_totals[split_name] += int(row["num_labeled_raw"])
        recording_totals[split_name].add(row["recording_id"])
    for split_name in ("train", "val", "test", "exclude"):
        print(
            f"[splits] {split_name}: groups={split_totals[split_name]} "
            f"recordings={len(recording_totals[split_name])} "
            f"labeled_raw={labeled_totals[split_name]}"
        )


if __name__ == "__main__":
    main()
