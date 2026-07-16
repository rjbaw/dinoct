import logging
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .extended import ExtendedVisionDataset
from ...oct_metadata import read_manifest_csv, read_splits_csv


logger = logging.getLogger("dinoct")
VECTOR_LENGTH = 500


class OCT(ExtendedVisionDataset):
    Target = np.ndarray | None

    def __init__(
        self,
        *,
        root: str,
        extra: str,
        split: str | None = None,
        transforms: Callable | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        self._split = split.strip().lower() if split else None

        self._entries = None

        entries_path = self._get_extra_full_path(self._entries_path)
        if os.path.exists(entries_path):
            logger.info("Refreshing metadata cache: %s", entries_path)
        else:
            logger.info("Metadata cache not found – generating: %s", entries_path)
        self._dump_entries()

    def _get_extra_full_path(self, extra_path: str) -> str:
        return os.path.join(self._extra_root, extra_path)

    def _load_extra(self, extra_path: str) -> np.ndarray:
        extra_full_path = self._get_extra_full_path(extra_path)
        return np.load(extra_full_path, mmap_mode="r")

    def _save_extra(self, extra_array: np.ndarray, extra_path: str) -> None:
        extra_full_path = self._get_extra_full_path(extra_path)
        os.makedirs(self._extra_root, exist_ok=True)
        np.save(extra_full_path, extra_array)

    @property
    def _entries_path(self) -> str:
        if self._split:
            return f"entries_{self._split}.npy"
        return "entries.npy"

    def _get_entries(self) -> np.ndarray:
        if self._entries is None:
            self._entries = self._load_extra(self._entries_path)
        assert self._entries is not None
        return self._entries

    def get_image_data(self, index: int) -> bytes:
        img_relpath = self._get_entries()[index]["filename"]
        with open(os.path.join(self.root, img_relpath), mode="rb") as f:
            return f.read()

    def get_code(self, index: int) -> int:
        return int(self._get_entries()[index]["code"])

    def get_target(self, index: int) -> Target | None:
        entry = self._get_entries()[index]
        code = int(entry["code"])

        if code == 2:
            # Background: return a zero vector (float32 for downstream torch conversion)
            return np.zeros(VECTOR_LENGTH, dtype=np.float32)

        if code == 1:
            label_relpath = ""
            if "label_relpath" in entry.dtype.names:
                label_relpath = str(entry["label_relpath"])
            if label_relpath:
                txt_path = os.path.join(self.root, label_relpath)
            else:
                base_name, _ = os.path.splitext(os.path.basename(entry["filename"]))
                txt_path = os.path.join(self.root, "labeled", base_name + ".txt")
            return self._load_label_vector(txt_path)

        return None

    def get_targets(self) -> np.ndarray:
        return self._get_entries()["code"]

    def __len__(self) -> int:
        entries = self._get_entries()
        return len(entries)

    def _dump_entries(self) -> None:
        manifest_path = Path(self._get_extra_full_path("manifest.csv"))
        if manifest_path.exists():
            self._dump_entries_from_manifest(manifest_path)
            return

        if self._split:
            raise FileNotFoundError(
                f"split={self._split!r} was requested but there is no manifest at "
                f"{manifest_path}. Split filtering needs extra/manifest.csv and "
                "extra/splits.csv; generate them with tools/data/build_oct_manifest.py "
                "and tools/data/build_oct_splits.py, or copy the released split "
                f"metadata into {self._extra_root!r}."
            )

        raw_dir = os.path.join(self.root, "raw")
        labeled_dir = os.path.join(self.root, "labeled")
        background_dir = os.path.join(self.root, "background")

        def collect_imgs(root: str) -> list[str]:
            if not os.path.isdir(root):
                return []
            files = [f for f in os.listdir(root) if f.lower().endswith(".jpg")]
            files.sort()
            return [os.path.join(root, f) for f in files]

        raw_imgs = collect_imgs(raw_dir)
        background_imgs = collect_imgs(background_dir)

        imgs = raw_imgs + background_imgs
        if not imgs:
            raise FileNotFoundError(
                "OCT dataset not found. Expected images under "
                f"{self.root!r} in at least one of: raw/ or background/ "
                "(and optional labels under labeled/)."
            )

        dtype = np.dtype(
            [
                ("filename", "U256"),
                ("code", "<u1"),
            ]
        )

        entries_array = np.empty(len(imgs), dtype=dtype)

        for idx, img_path in enumerate(imgs):
            rel_path = os.path.relpath(img_path, self.root)
            base_name, _ = os.path.splitext(os.path.basename(img_path))

            if img_path.startswith(background_dir):
                code = 2
            else:
                txt_path = os.path.join(labeled_dir, base_name + ".txt")
                code = 1 if os.path.exists(txt_path) else 0

            entries_array[idx] = (rel_path, code)

        logger.info(f'saving entries to "{self._entries_path}"')
        self._save_extra(entries_array, self._entries_path)
    def _dump_entries_from_manifest(self, manifest_path: Path) -> None:
        rows = read_manifest_csv(manifest_path)
        split_map: dict[str, str] = {}
        splits_path = Path(self._get_extra_full_path("splits.csv"))
        if splits_path.exists():
            split_map = {key: value.lower() for key, value in read_splits_csv(splits_path).items()}

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

        filtered: list[tuple[str, int, str, str, str, str, str, str, str, str, str]] = []
        for row in rows:
            row_split = split_map.get(row.group_id, "")
            if self._split and row_split != self._split:
                continue
            filtered.append(
                (
                    row.image_relpath,
                    int(row.code),
                    row.label_relpath,
                    row.paired_background_relpath,
                    row.group_id,
                    row.family_id,
                    row.variant,
                    row.modality,
                    row.sample_id,
                    row.kind,
                    row_split,
                )
            )

        if self._split and not filtered:
            if not splits_path.exists():
                raise FileNotFoundError(
                    f"split={self._split!r} was requested and a manifest exists, but there is "
                    f"no splits.csv at {splits_path}. Generate it with "
                    "tools/data/build_oct_splits.py or copy the released splits.csv there."
                )
            raise FileNotFoundError(
                f"No OCT samples matched split={self._split!r} using manifest {manifest_path}; "
                f"check that {splits_path} covers these group_ids."
            )

        entries_array = np.empty(len(filtered), dtype=dtype)
        for idx, item in enumerate(filtered):
            entries_array[idx] = item

        logger.info(f'saving entries to "{self._entries_path}"')
        self._save_extra(entries_array, self._entries_path)

    def _load_label_vector(self, txt_path: str) -> np.ndarray:
        """
        Load a per-column z vector (axial image row per column) from .txt. Accepts either:
          - 500 floats (one per column), or
          - a 500x2 table (x, z), from which the second column is used.
        Returns float32 array of shape (500,).
        """
        arr = np.loadtxt(txt_path)
        # Handle possible (N,2) where first col is x and second is z
        if arr.ndim == 2:
            if arr.shape[1] == 2:
                vec = arr[:, 1]
            elif arr.shape[1] == 1:
                vec = arr[:, 0]
            elif arr.shape[0] == 1:
                # A single row of values (e.g., written as "1×N") -> treat as vector.
                vec = arr[0]
            else:
                raise ValueError(f"{txt_path} has unexpected shape {arr.shape}; expected (500,) or (500,2)")
        else:
            vec = arr
        vec = np.asarray(vec).reshape(-1)
        if vec.shape[0] != VECTOR_LENGTH:
            raise ValueError(f"{txt_path} must contain {VECTOR_LENGTH} values; got shape {vec.shape}")
        return vec.astype(np.float32)

    def dump_extra(self) -> None:
        self._dump_entries()


__all__ = ["OCT"]
