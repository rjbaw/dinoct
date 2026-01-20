import logging
import os
from collections.abc import Callable

import numpy as np

from .extended import ExtendedVisionDataset


logger = logging.getLogger("dinoct")
VECTOR_LENGTH = 500


class OCT(ExtendedVisionDataset):
    Target = np.ndarray | None

    def __init__(
        self,
        *,
        # split: "ImageNet.Split",
        # split: "",
        root: str,
        extra: str,
        transforms: Callable | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        # self._split = split

        # self._entries = Optional[np.ndarray] = None
        self._entries = None
        # self._class_ids = None
        # self._class_names = None

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
        return "entries.npy"

    def _get_entries(self) -> np.ndarray:
        if self._entries is None:
            self._entries = self._load_extra(self._entries_path)
        assert self._entries is not None
        return self._entries

    # def _get_class_ids(self) -> np.ndarray:
    #     if self._split == _Split.TEST:
    #         assert False, "Class IDs are not available in TEST split"
    #     if self._class_ids is None:
    #         self._class_ids = self._load_extra(self._class_ids_path)
    #     assert self._class_ids is not None
    #     return self._class_ids

    # def _get_class_names(self) -> np.ndarray:
    #     if self._split == _Split.TEST:
    #         assert False, "Class names are not available in TEST split"
    #     if self._class_names is None:
    #         self._class_names = self._load_extra(self._class_names_path)
    #     assert self._class_names is not None
    #     return self._class_names

    # def find_class_id(self, class_index: int) -> str:
    #     class_ids = self._get_class_ids()
    #     return str(class_ids[class_index])

    # def find_class_name(self, class_index: int) -> str:
    #     class_names = self._get_class_names()
    #     return str(class_names[class_index])

    def get_image_data(self, index: int) -> bytes:
        img_relpath = self._get_entries()[index]["filename"]
        with open(os.path.join(self.root, img_relpath), mode="rb") as f:
            return f.read()

    def get_target(self, index: int) -> Target | None:
        entry = self._get_entries()[index]
        code = entry["code"]

        if code == 2:
            # Background: return a zero vector (float32 for downstream torch conversion)
            return np.zeros(VECTOR_LENGTH, dtype=np.float32)

        if code == 1:
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

    def _load_label_vector(self, txt_path: str) -> np.ndarray:
        """
        Load a per-column y vector from .txt. Accepts either:
          - 500 floats (one per column), or
          - a 500x2 table (x, y), from which the second column is used.
        Returns float32 array of shape (500,).
        """
        arr = np.loadtxt(txt_path)
        # Handle possible (N,2) where first col is x and second is y
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

    # def _dump_class_ids_and_names(self) -> None:
    #     split = self.split
    #     if split == ImageNet.Split.TEST:
    #         return

    #     entries_array = self._load_extra(self._entries_path)

    #     max_class_id_length, max_class_name_length, max_class_index = -1, -1, -1
    #     for entry in entries_array:
    #         class_index, class_id, class_name = (
    #             entry["class_index"],
    #             entry["class_id"],
    #             entry["class_name"],
    #         )
    #         max_class_index = max(int(class_index), max_class_index)
    #         max_class_id_length = max(len(str(class_id)), max_class_id_length)
    #         max_class_name_length = max(len(str(class_name)), max_class_name_length)

    #     class_count = max_class_index + 1
    #     class_ids_array = np.empty(class_count, dtype=f"U{max_class_id_length}")
    #     class_names_array = np.empty(class_count, dtype=f"U{max_class_name_length}")
    #     for entry in entries_array:
    #         class_index, class_id, class_name = (
    #             entry["class_index"],
    #             entry["class_id"],
    #             entry["class_name"],
    def dump_extra(self) -> None:
        self._dump_entries()


__all__ = ["OCT"]
