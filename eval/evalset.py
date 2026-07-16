from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VECTOR_LENGTH = 500


@dataclass(frozen=True, slots=True)
class DirectoryEvalConfig:
    eval_dir: Path
    split_name: str = "real_hard"
    acquisition_mode: str = "real_hard"
    recording_id: str | None = None


class DirectoryCurveEvalDataset:
    def __init__(self, config: DirectoryEvalConfig) -> None:
        self.eval_dir = config.eval_dir.expanduser().resolve()
        if not self.eval_dir.is_dir():
            raise FileNotFoundError(self.eval_dir)
        self.root = str(self.eval_dir)
        self._split = str(config.split_name)
        self._acquisition_mode = str(config.acquisition_mode)
        self._dataset_id = str(config.recording_id or self.eval_dir.name or "real_hard")
        self._entries = self._build_entries()

    def _build_entries(self) -> np.ndarray:
        image_paths = sorted([p for p in self.eval_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS], key=lambda p: p.name)
        rows: list[tuple[str, int, str, str, str, str, str, str, str, str, str]] = []
        for image_path in image_paths:
            label_path = image_path.with_suffix(".txt")
            if not label_path.exists():
                continue
            filename = image_path.name
            label_rel = label_path.name
            stem = image_path.stem
            sample_id = f"{self._split}:{stem}"
            group_id = f"{self._acquisition_mode}:{self._dataset_id}:{stem}"
            rows.append(
                (
                    filename,
                    1,
                    label_rel,
                    "",
                    group_id,
                    f"{self._acquisition_mode}:{self._dataset_id}",
                    stem,
                    self._acquisition_mode,
                    sample_id,
                    "eval",
                    self._split,
                )
            )

        if not rows:
            raise FileNotFoundError(f"No paired image/txt samples found in {self.eval_dir}")

        dtype = np.dtype(
            [
                ("filename", "U512"),
                ("code", "<u1"),
                ("label_relpath", "U512"),
                ("background_relpath", "U512"),
                ("group_id", "U128"),
                ("family_id", "U128"),
                ("variant", "U128"),
                ("modality", "U128"),
                ("sample_id", "U512"),
                ("kind", "U32"),
                ("split", "U16"),
            ]
        )
        out = np.empty(len(rows), dtype=dtype)
        for idx, row in enumerate(rows):
            out[idx] = row
        return out

    def _get_entries(self) -> np.ndarray:
        return self._entries

    def get_target(self, index: int) -> np.ndarray:
        entry = self._entries[int(index)]
        return load_label_vector(Path(self.root) / str(entry["label_relpath"]))

    def __len__(self) -> int:
        return int(self._entries.shape[0])


def load_label_vector(txt_path: Path) -> np.ndarray:
    arr = np.loadtxt(txt_path)
    if arr.ndim == 2:
        if arr.shape[1] == 2:
            vec = arr[:, 1]
        elif arr.shape[1] == 1:
            vec = arr[:, 0]
        elif arr.shape[0] == 1:
            vec = arr[0]
        else:
            raise ValueError(f"{txt_path} has unexpected shape {arr.shape}; expected (500,) or (500,2)")
    else:
        vec = arr
    vec = np.asarray(vec).reshape(-1)
    if vec.shape[0] != VECTOR_LENGTH:
        raise ValueError(f"{txt_path} must contain {VECTOR_LENGTH} values; got shape {vec.shape}")
    return vec.astype(np.float32)


def split_rows_for_directory_dataset(dataset: DirectoryCurveEvalDataset) -> dict[str, dict[str, str]]:
    entries = dataset._get_entries()
    out: dict[str, dict[str, str]] = {}
    for entry in entries:
        group_id = str(entry["group_id"])
        out[group_id] = {
            "recording_id": group_id,
            "split": str(entry["split"]),
            "acquisition_mode": str(entry["modality"]),
        }
    return out


__all__ = [
    "DirectoryCurveEvalDataset",
    "DirectoryEvalConfig",
    "load_label_vector",
    "split_rows_for_directory_dataset",
]
