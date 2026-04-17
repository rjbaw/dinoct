from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MANIFEST_FIELDS = (
    "sample_id",
    "stem",
    "kind",
    "image_relpath",
    "label_relpath",
    "paired_background_relpath",
    "background_match",
    "code",
    "group_id",
    "family_id",
    "variant",
    "modality",
    "frame_id",
    "capture_ts",
)
SPLIT_FIELDS = (
    "group_id",
    "recording_id",
    "split",
    "acquisition_mode",
    "num_labeled_raw",
    "num_raw",
    "num_background",
    "num_paired_labeled_raw",
    "modalities",
)
GROUP_INDEX_FIELDS = (
    "group_id",
    "recording_id",
    "acquisition_mode",
    "num_labeled_raw",
    "num_raw",
    "num_background",
    "num_paired_labeled_raw",
    "modalities",
    "example_stems",
)

_TIMESTAMP_CAPTURE_RE = re.compile(r"^raw_image(?P<frame>\d+)_(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})$")
_LEGACY_TIMESTAMP_CAPTURE_RE = re.compile(r"^(?P<frame>\d+)_(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})$")
_RAW_BURST_RE = re.compile(r"^raw_image(?P<frame>\d+)_(?P<variant>test\d+|high_reflect)$")
_FAMILY_VARIANT_RE = re.compile(r"^(?P<family>[^_]+)_(?P<variant>.+)$")

_CONTINUOUS_VARIANTS = {
    "3d_aligned",
    "3d_aligned_test",
    "3d_not_aligned",
    "3d_not_aligned_test",
    "current",
    "new",
    "new_capture",
    "no_background",
    "real",
    "skin",
    "skin_oct",
    "skin_oct_motion",
    "skin_octa",
    "skin_octa_static",
    "test4",
    "test5",
}


@dataclass(frozen=True, slots=True)
class OCTSampleMeta:
    sample_id: str
    stem: str
    kind: str
    image_relpath: str
    label_relpath: str
    paired_background_relpath: str
    background_match: str
    code: int
    group_id: str
    family_id: str
    variant: str
    modality: str
    frame_id: str
    capture_ts: str

    def to_row(self) -> dict[str, str]:
        return {
            "sample_id": self.sample_id,
            "stem": self.stem,
            "kind": self.kind,
            "image_relpath": self.image_relpath,
            "label_relpath": self.label_relpath,
            "paired_background_relpath": self.paired_background_relpath,
            "background_match": self.background_match,
            "code": str(int(self.code)),
            "group_id": self.group_id,
            "family_id": self.family_id,
            "variant": self.variant,
            "modality": self.modality,
            "frame_id": self.frame_id,
            "capture_ts": self.capture_ts,
        }


def _iter_image_paths(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    items = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    items.sort(key=lambda p: p.name.lower())
    return items


def _normalize_modality(variant: str, *, fallback: str = "single") -> str:
    value = variant.strip().lower()
    return value or fallback


def infer_stem_metadata(stem: str) -> dict[str, str]:
    match = _TIMESTAMP_CAPTURE_RE.match(stem)
    if match:
        frame_idx = match.group("frame")
        capture_ts = match.group("ts")
        frame_id = f"image{frame_idx}"
        return {
            "group_id": f"cscan:{capture_ts}",
            "family_id": capture_ts,
            "variant": frame_id,
            "modality": "raw_capture",
            "frame_id": frame_id,
            "capture_ts": capture_ts,
        }

    match = _LEGACY_TIMESTAMP_CAPTURE_RE.match(stem)
    if match:
        frame_idx = match.group("frame")
        capture_ts = match.group("ts")
        return {
            "group_id": f"cscan:{capture_ts}",
            "family_id": capture_ts,
            "variant": "legacy_capture",
            "modality": "legacy_cscan",
            "frame_id": match.group("frame"),
            "capture_ts": capture_ts,
        }

    match = _RAW_BURST_RE.match(stem)
    if match:
        frame_idx = match.group("frame")
        variant = match.group("variant")
        return {
            "group_id": f"continuous:{variant}",
            "family_id": frame_idx,
            "variant": variant,
            "modality": variant,
            "frame_id": frame_idx,
            "capture_ts": "",
        }

    match = _FAMILY_VARIANT_RE.match(stem)
    if match:
        family_id = match.group("family")
        variant = match.group("variant")
        if family_id.isdigit():
            if variant in _CONTINUOUS_VARIANTS:
                return {
                    "group_id": f"continuous:{variant}",
                    "family_id": family_id,
                    "variant": variant,
                    "modality": variant,
                    "frame_id": family_id,
                    "capture_ts": "",
                }
            return {
                "group_id": f"continuous:{variant}",
                "family_id": family_id,
                "variant": variant,
                "modality": _normalize_modality(variant),
                "frame_id": family_id,
                "capture_ts": "",
            }

        group_prefix = "group"
        return {
            "group_id": f"{group_prefix}:{family_id}",
            "family_id": family_id,
            "variant": variant,
            "modality": _normalize_modality(variant),
            "frame_id": "",
            "capture_ts": "",
        }

    return {
        "group_id": f"sample:{stem}",
        "family_id": stem,
        "variant": "",
        "modality": "single",
        "frame_id": "",
        "capture_ts": "",
    }


def background_pairing_key(stem: str) -> tuple[str, str] | None:
    if _TIMESTAMP_CAPTURE_RE.match(stem):
        return None
    match = _FAMILY_VARIANT_RE.match(stem)
    if match is None:
        return None
    return match.group("family"), match.group("variant")


def build_manifest_rows(root: Path) -> list[OCTSampleMeta]:
    root = root.expanduser()
    raw_dir = root / "raw"
    background_dir = root / "background"
    labeled_dir = root / "labeled"

    raw_paths = _iter_image_paths(raw_dir)
    background_paths = _iter_image_paths(background_dir)
    label_paths = sorted(p for p in labeled_dir.glob("*.txt") if p.is_file()) if labeled_dir.is_dir() else []

    labels_by_stem = {path.stem: str(path.relative_to(root)) for path in label_paths}
    bg_by_stem = {path.stem: str(path.relative_to(root)) for path in background_paths}
    bg_by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    for path in background_paths:
        key = background_pairing_key(path.stem)
        if key is not None:
            bg_by_key[key].append(str(path.relative_to(root)))
    for values in bg_by_key.values():
        values.sort()

    rows: list[OCTSampleMeta] = []

    for path in raw_paths:
        stem = path.stem
        label_relpath = labels_by_stem.get(stem, "")
        paired_background_relpath = ""
        background_match = ""
        if stem in bg_by_stem:
            paired_background_relpath = bg_by_stem[stem]
            background_match = "exact"
        else:
            key = background_pairing_key(stem)
            if key is not None:
                candidates = bg_by_key.get(key, [])
                if candidates:
                    paired_background_relpath = candidates[0]
                    background_match = "family_variant" if len(candidates) == 1 else "family_variant_ambiguous"

        meta = infer_stem_metadata(stem)
        rows.append(
            OCTSampleMeta(
                sample_id=f"raw:{stem}",
                stem=stem,
                kind="raw",
                image_relpath=str(path.relative_to(root)),
                label_relpath=label_relpath,
                paired_background_relpath=paired_background_relpath,
                background_match=background_match,
                code=1 if label_relpath else 0,
                group_id=meta["group_id"],
                family_id=meta["family_id"],
                variant=meta["variant"],
                modality=meta["modality"],
                frame_id=meta["frame_id"],
                capture_ts=meta["capture_ts"],
            )
        )

    for path in background_paths:
        stem = path.stem
        meta = infer_stem_metadata(stem)
        rows.append(
            OCTSampleMeta(
                sample_id=f"background:{stem}",
                stem=stem,
                kind="background",
                image_relpath=str(path.relative_to(root)),
                label_relpath="",
                paired_background_relpath="",
                background_match="",
                code=2,
                group_id=f"background:{stem}",
                family_id=stem,
                variant=meta["variant"],
                modality=meta["modality"],
                frame_id=meta["frame_id"],
                capture_ts=meta["capture_ts"],
            )
        )

    rows.sort(key=lambda row: (row.kind != "raw", row.image_relpath))
    return rows


def write_manifest_csv(path: Path, rows: list[OCTSampleMeta]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(MANIFEST_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_row())


def read_manifest_csv(path: Path) -> list[OCTSampleMeta]:
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        rows: list[OCTSampleMeta] = []
        for raw in reader:
            rows.append(
                OCTSampleMeta(
                    sample_id=str(raw.get("sample_id", "") or ""),
                    stem=str(raw.get("stem", "") or ""),
                    kind=str(raw.get("kind", "") or ""),
                    image_relpath=str(raw.get("image_relpath", "") or ""),
                    label_relpath=str(raw.get("label_relpath", "") or ""),
                    paired_background_relpath=str(raw.get("paired_background_relpath", "") or ""),
                    background_match=str(raw.get("background_match", "") or ""),
                    code=int(raw.get("code", "0") or 0),
                    group_id=str(raw.get("group_id", "") or ""),
                    family_id=str(raw.get("family_id", "") or ""),
                    variant=str(raw.get("variant", "") or ""),
                    modality=str(raw.get("modality", "") or ""),
                    frame_id=str(raw.get("frame_id", "") or ""),
                    capture_ts=str(raw.get("capture_ts", "") or ""),
                )
            )
    return rows


def write_splits_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SPLIT_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: str(row.get(field, "") or "") for field in SPLIT_FIELDS})


def read_splits_csv(path: Path) -> dict[str, str]:
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        out: dict[str, str] = {}
        for row in reader:
            group_id = str(row.get("group_id", "") or "").strip()
            split = str(row.get("split", "") or "").strip()
            if group_id and split:
                out[group_id] = split
    return out


def default_acquisition_mode(group_id: str) -> str:
    if group_id.startswith("background:"):
        return "background"
    if group_id.startswith("cscan:"):
        return "c_scan"
    if group_id.startswith("continuous:"):
        return "continuous_sequence"
    if group_id.startswith("series:"):
        return "recording_family"
    return "other"


def build_group_index_rows(rows: list[OCTSampleMeta]) -> list[dict[str, str]]:
    groups: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "num_labeled_raw": 0,
            "num_raw": 0,
            "num_background": 0,
            "num_paired_labeled_raw": 0,
            "modalities": set(),
            "example_stems": [],
        }
    )

    for row in rows:
        group = groups[row.group_id]
        modalities = group["modalities"]
        assert isinstance(modalities, set)
        if row.modality:
            modalities.add(row.modality)

        stems = group["example_stems"]
        assert isinstance(stems, list)
        if len(stems) < 5 and row.stem not in stems:
            stems.append(row.stem)

        if row.kind == "raw":
            group["num_raw"] = int(group["num_raw"]) + 1
            if row.code == 1:
                group["num_labeled_raw"] = int(group["num_labeled_raw"]) + 1
                if row.paired_background_relpath:
                    group["num_paired_labeled_raw"] = int(group["num_paired_labeled_raw"]) + 1
        elif row.kind == "background":
            group["num_background"] = int(group["num_background"]) + 1

    out: list[dict[str, str]] = []
    for group_id in sorted(groups):
        group = groups[group_id]
        modalities = group["modalities"]
        stems = group["example_stems"]
        assert isinstance(modalities, set)
        assert isinstance(stems, list)
        if int(group["num_raw"]) == 0:
            continue
        out.append(
            {
                "group_id": group_id,
                "recording_id": group_id,
                "acquisition_mode": default_acquisition_mode(group_id),
                "num_labeled_raw": str(int(group["num_labeled_raw"])),
                "num_raw": str(int(group["num_raw"])),
                "num_background": str(int(group["num_background"])),
                "num_paired_labeled_raw": str(int(group["num_paired_labeled_raw"])),
                "modalities": "|".join(sorted(str(item) for item in modalities)),
                "example_stems": "|".join(str(item) for item in stems),
            }
        )
    return out


def write_group_index_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(GROUP_INDEX_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: str(row.get(field, "") or "") for field in GROUP_INDEX_FIELDS})


def read_group_overrides_csv(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}

    out: dict[str, dict[str, str]] = {}
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            group_id = str(row.get("group_id", "") or "").strip()
            if not group_id:
                continue
            out[group_id] = {
                "recording_id": str(row.get("recording_id", "") or "").strip(),
                "acquisition_mode": str(row.get("acquisition_mode", "") or "").strip(),
            }
    return out


def summarize_manifest(rows: list[OCTSampleMeta]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        counts[f"kind:{row.kind}"] += 1
        counts[f"code:{row.code}"] += 1
        if row.paired_background_relpath:
            counts["raw_with_background_pair"] += 1
        if row.capture_ts:
            counts["capture_group_rows"] += 1
    counts["groups"] = len({row.group_id for row in rows if row.kind == "raw"})
    counts["background_groups"] = len({row.group_id for row in rows if row.kind == "background"})
    return dict(sorted(counts.items()))
