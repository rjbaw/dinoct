#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


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
SPLIT_NAMES = ("train", "val", "test", "exclude")


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
    parser.add_argument(
        "--cscan-session-gap-sec",
        type=float,
        default=15.0,
        help=(
            "Consecutive timestamped C-scan captures separated by at most this many seconds are "
            "treated as one recording/split unit."
        ),
    )
    parser.add_argument(
        "--no-exact-dedup",
        action="store_true",
        help="Disable grouping C-scan captures that share byte-identical raw image files.",
    )
    parser.add_argument(
        "--dedup-ahash-threshold",
        type=int,
        default=1,
        help=(
            "Average-hash Hamming threshold for grouping visually similar C-scan captures. "
            "Default 1 is the audited paper setting."
        ),
    )
    parser.add_argument(
        "--no-ahash-dedup",
        action="store_true",
        help="Disable average-hash grouping of visually similar C-scan captures.",
    )
    parser.add_argument(
        "--dedup-ahash-size",
        type=int,
        default=16,
        help="Average-hash side length used with --dedup-ahash-threshold.",
    )
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
            "image_relpaths": [],
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
            image_relpaths = group["image_relpaths"]
            assert isinstance(image_relpaths, list)
            image_relpaths.append(row.image_relpath)
            if row.code == 1:
                group["num_labeled_raw"] = int(group["num_labeled_raw"]) + 1
                if row.paired_background_relpath:
                    group["num_paired_labeled_raw"] = int(group["num_paired_labeled_raw"]) + 1
        elif row.kind == "background":
            group["num_background"] = int(group["num_background"]) + 1

    return groups


def _parse_cscan_time(group_id: str) -> datetime | None:
    if not group_id.startswith("cscan:"):
        return None
    value = group_id.split(":", 1)[1]
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H-%M-%S")
    except ValueError:
        return None


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {str(value): str(value) for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _recording_id_for_cscan_component(group_ids: list[str]) -> str:
    # Keep the split unit private in shared artifacts.  The exact group_id rows
    # remain in splits.csv, but the derived recording/session cluster is only a
    # stable opaque key used for assignment and per-recording aggregation.
    digest = hashlib.sha256("\n".join(sorted(group_ids)).encode("utf-8")).hexdigest()[:16]
    return "cscan_session:{}".format(digest)


def _group_exact_duplicate_cscans(
    *,
    dataset_root: Path,
    cscan_group_ids: list[str],
    groups: dict[str, dict[str, object]],
    union_find: _UnionFind,
) -> None:
    groups_by_hash: dict[str, list[str]] = defaultdict(list)
    for group_id in cscan_group_ids:
        image_relpaths = groups[group_id].get("image_relpaths", [])
        assert isinstance(image_relpaths, list)
        for relpath in image_relpaths:
            path = dataset_root / str(relpath)
            if not path.exists():
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            groups_by_hash[digest].append(group_id)

    for group_ids in groups_by_hash.values():
        unique_ids = sorted(set(group_ids))
        if len(unique_ids) < 2:
            continue
        first = unique_ids[0]
        for group_id in unique_ids[1:]:
            union_find.union(first, group_id)


def _average_hash(path: Path, *, size: int) -> int:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on optional runtime package.
        raise RuntimeError("--dedup-ahash-threshold requires Pillow to be installed.") from exc

    if size <= 0:
        raise ValueError("--dedup-ahash-size must be positive")
    image = Image.open(path).convert("L").resize((size, size), Image.Resampling.BILINEAR)
    pixel_reader = getattr(image, "get_flattened_data", image.getdata)
    pixels = list(pixel_reader())
    threshold = sum(float(value) for value in pixels) / float(len(pixels))
    out = 0
    for idx, value in enumerate(pixels):
        if float(value) >= threshold:
            out |= 1 << idx
    return out


def _group_ahash_similar_cscans(
    *,
    dataset_root: Path,
    cscan_group_ids: list[str],
    groups: dict[str, dict[str, object]],
    union_find: _UnionFind,
    threshold: int,
    size: int,
) -> None:
    if threshold < 0:
        raise ValueError("--dedup-ahash-threshold must be non-negative")
    hashes: list[tuple[str, int]] = []
    for group_id in cscan_group_ids:
        image_relpaths = groups[group_id].get("image_relpaths", [])
        assert isinstance(image_relpaths, list)
        if not image_relpaths:
            continue
        path = dataset_root / str(image_relpaths[0])
        if path.exists():
            hashes.append((group_id, _average_hash(path, size=size)))

    if threshold == 0:
        by_hash: dict[int, list[str]] = defaultdict(list)
        for group_id, digest in hashes:
            by_hash[digest].append(group_id)
        for group_ids in by_hash.values():
            if len(group_ids) < 2:
                continue
            first = group_ids[0]
            for group_id in group_ids[1:]:
                union_find.union(first, group_id)
        return

    for idx, (left_id, left_hash) in enumerate(hashes):
        for right_id, right_hash in hashes[idx + 1 :]:
            if (left_hash ^ right_hash).bit_count() <= threshold:
                union_find.union(left_id, right_id)


def _build_cscan_recording_map(
    *,
    dataset_root: Path,
    groups: dict[str, dict[str, object]],
    session_gap_sec: float,
    exact_dedup: bool,
    ahash_threshold: int | None,
    ahash_size: int,
) -> dict[str, str]:
    cscan_items = [
        (group_id, parsed_time)
        for group_id in groups
        if (parsed_time := _parse_cscan_time(group_id)) is not None
    ]
    cscan_items.sort(key=lambda item: item[1])
    cscan_group_ids = [group_id for group_id, _dt in cscan_items]
    union_find = _UnionFind(cscan_group_ids)

    previous_group_id: str | None = None
    previous_time: datetime | None = None
    for group_id, parsed_time in cscan_items:
        if (
            previous_group_id is not None
            and previous_time is not None
            and (parsed_time - previous_time).total_seconds() <= float(session_gap_sec)
        ):
            union_find.union(previous_group_id, group_id)
        previous_group_id = group_id
        previous_time = parsed_time

    if exact_dedup:
        _group_exact_duplicate_cscans(
            dataset_root=dataset_root,
            cscan_group_ids=cscan_group_ids,
            groups=groups,
            union_find=union_find,
        )

    if ahash_threshold is not None:
        _group_ahash_similar_cscans(
            dataset_root=dataset_root,
            cscan_group_ids=cscan_group_ids,
            groups=groups,
            union_find=union_find,
            threshold=int(ahash_threshold),
            size=int(ahash_size),
        )

    components: dict[str, list[str]] = defaultdict(list)
    for group_id in cscan_group_ids:
        components[union_find.find(group_id)].append(group_id)

    out: dict[str, str] = {}
    for group_ids in components.values():
        recording_id = _recording_id_for_cscan_component(sorted(group_ids))
        for group_id in group_ids:
            out[group_id] = recording_id
    return out


def _resolve_recordings(
    groups: dict[str, dict[str, object]],
    *,
    dataset_root: Path,
    session_gap_sec: float,
    exact_dedup: bool,
    ahash_threshold: int | None,
    ahash_size: int,
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
    cscan_recording_map = _build_cscan_recording_map(
        dataset_root=dataset_root,
        groups=groups,
        session_gap_sec=session_gap_sec,
        exact_dedup=exact_dedup,
        ahash_threshold=ahash_threshold,
        ahash_size=ahash_size,
    )

    for group_id, stats in groups.items():
        recording_id = cscan_recording_map.get(group_id, group_id)
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


def _validate_rows(rows: list[dict[str, str]]) -> None:
    """Fail loudly if a recording/session unit leaks across splits."""
    splits_by_recording: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        recording_id = row["recording_id"]
        split_name = row["split"]
        if split_name not in SPLIT_NAMES:
            raise ValueError(f"Unexpected split value for {recording_id}: {split_name!r}")
        splits_by_recording[recording_id].add(split_name)

    leaking = {
        recording_id: sorted(split_names)
        for recording_id, split_names in splits_by_recording.items()
        if len(split_names) > 1
    }
    if leaking:
        examples = list(leaking.items())[:10]
        raise RuntimeError(f"Found recording_ids assigned to multiple splits: {examples}")


def _print_summary(rows: list[dict[str, str]], output_path: Path) -> None:
    print(f"[splits] wrote {len(rows)} group assignments to {output_path}")
    split_totals = defaultdict(int)
    labeled_totals = defaultdict(int)
    recording_totals = defaultdict(set)
    for row in rows:
        split_name = row["split"]
        split_totals[split_name] += 1
        labeled_totals[split_name] += int(row["num_labeled_raw"])
        recording_totals[split_name].add(row["recording_id"])
    for split_name in SPLIT_NAMES:
        print(
            f"[splits] {split_name}: groups={split_totals[split_name]} "
            f"recordings={len(recording_totals[split_name])} "
            f"labeled_raw={labeled_totals[split_name]}"
        )
    print("[splits] recording_ids spanning multiple splits: 0")


def main() -> None:
    args = parse_args()
    dataset_root = args.dir.expanduser()
    manifest_path = args.manifest.expanduser() if args.manifest else dataset_root / "extra" / "manifest.csv"
    output_path = args.output.expanduser() if args.output else dataset_root / "extra" / "splits.csv"

    fractions = _validate_fracs(args.train_frac, args.val_frac)
    groups = _collect_group_stats(manifest_path)
    recordings, group_map = _resolve_recordings(
        groups,
        dataset_root=dataset_root,
        session_gap_sec=float(args.cscan_session_gap_sec),
        exact_dedup=not bool(args.no_exact_dedup),
        ahash_threshold=None if bool(args.no_ahash_dedup) else args.dedup_ahash_threshold,
        ahash_size=int(args.dedup_ahash_size),
    )
    assignments = _assign_recordings(recordings, fractions, seed=args.seed)
    rows = _build_rows(groups, group_map, assignments)
    _validate_rows(rows)
    write_splits_csv(output_path, rows)
    _print_summary(rows, output_path)


if __name__ == "__main__":
    main()
