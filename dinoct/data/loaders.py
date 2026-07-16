# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
from pathlib import Path
import shutil
import tarfile
from typing import Any, Callable, Optional, TypeVar

from huggingface_hub import hf_hub_download
import torch

from ..utils import seed_worker
from .datasets import OCT


logger = logging.getLogger("dinoct")

_DINOCT_CACHE_DIR = Path(os.environ.get("DINOCT_CACHE_DIR", Path.home() / ".cache" / "dinoct"))
_HF_OCT_ARCHIVE_NAME = "oct.tar.gz"

def _parse_dataset_str(dataset_str: str):
    tokens = dataset_str.split(":")

    name = tokens[0].strip().upper()
    kwargs: dict[str, str] = {}

    for token in tokens[1:]:
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f'Invalid dataset token "{token}". Expected "key=value".')
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key not in ("root", "extra", "split", "hub", "revision", "cache_dir"):
            raise ValueError(f'Unsupported dataset option "{key}" in "{dataset_str}"')
        kwargs[key] = value

    if name == "OCT":
        kwargs = _resolve_oct_dataset_kwargs(kwargs)
        class_ = OCT
    else:
        raise ValueError(f'Unsupported dataset "{name}"')

    return class_, kwargs


def _resolve_oct_dataset_kwargs(kwargs: dict[str, str]) -> dict[str, str]:
    """
    Normalizes OCT dataset kwargs.

    Supports either:
      - Local paths: OCT:root=/path/to/oct[:extra=/path/to/extra]
      - Hugging Face Hub dataset repo: OCT:hub=<user/dataset>[:revision=...][:cache_dir=...][:extra=...]

    A local root is preferred over "hub=" only when it actually holds the image tree
    (raw/ or background/). A root that exists but only carries metadata (e.g. the
    documented data/oct/extra/ split-metadata location) does not shadow the hub
    download, and its extra/ dir is still honored for split metadata.
    """
    root = kwargs.get("root")
    prefer_local = root is not None and _local_root_has_images(Path(root))

    if "hub" in kwargs and not prefer_local:
        repo_id = kwargs.pop("hub")
        local_root = kwargs.pop("root", None)
        if local_root is not None and Path(local_root).exists():
            logger.info(
                'local root "%s" has no raw/ or background/ images: downloading from hub "%s"',
                local_root,
                repo_id,
            )

        revision = kwargs.pop("revision", None) or None
        cache_dir = kwargs.pop("cache_dir", None) or None

        archive_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=_HF_OCT_ARCHIVE_NAME,
                repo_type="dataset",
                revision=revision,
                cache_dir=cache_dir,
            )
        )
        base = _hf_archive_cache_base(repo_id=repo_id, revision=revision or "main")
        extracted_root = _extract_hf_archive(archive_path=archive_path, base=base)
        kwargs["root"] = str(extracted_root)
        if "extra" not in kwargs:
            kwargs["extra"] = str(_resolve_hub_extra_dir(local_root=local_root, extracted_root=extracted_root))
        logger.info(
            "hub dataset %s: root=%s extra=%s",
            repo_id,
            extracted_root,
            kwargs["extra"],
        )
    else:
        if "hub" in kwargs:
            # Prefer the local root when it holds the image tree.
            kwargs.pop("hub", None)
            kwargs.pop("revision", None)
            kwargs.pop("cache_dir", None)
        else:
            for key in ("revision", "cache_dir"):
                if key in kwargs:
                    raise ValueError(f'OCT dataset option "{key}=" is only valid when using "hub=".')

    if "root" not in kwargs:
        raise ValueError('OCT dataset requires "root=<path>" or "hub=<user/dataset>".')

    # Make "extra" optional for local datasets.
    if "extra" not in kwargs:
        kwargs["extra"] = str(Path(kwargs["root"]) / "extra")

    _warn_on_partial_split_metadata(Path(kwargs["extra"]))

    extra_keys = set(kwargs) - {"root", "extra", "split"}
    if extra_keys:
        raise ValueError(f"OCT dataset has unsupported options: {sorted(extra_keys)}")

    return kwargs


def _local_root_has_images(root: Path) -> bool:
    """True when a local OCT root holds the image tree (raw/ or background/)."""
    return (root / "raw").is_dir() or (root / "background").is_dir()


def _resolve_hub_extra_dir(*, local_root: str | None, extracted_root: Path) -> Path:
    """Pick the split-metadata dir for a hub-downloaded dataset.

    The hub archive ships images only; the released/generated split metadata
    (extra/manifest.csv + extra/splits.csv) is documented to live under the local
    root (e.g. data/oct/extra/). Honor that location when it holds metadata so the
    documented reproduction flow works even when images come from the hub cache;
    otherwise fall back to the extracted archive's own extra/ dir.
    """
    if local_root:
        local_extra = Path(local_root) / "extra"
        if (local_extra / "manifest.csv").is_file() or (local_extra / "splits.csv").is_file():
            return local_extra
    return extracted_root / "extra"


def _warn_on_partial_split_metadata(extra_dir: Path) -> None:
    """The explicit split assignment needs BOTH manifest.csv and splits.csv; warn otherwise."""
    if (extra_dir / "splits.csv").is_file() and not (extra_dir / "manifest.csv").is_file():
        logger.warning(
            'found "%s" but no manifest.csv next to it: splits.csv assigns splits to '
            "manifest group_ids, so it cannot take effect alone and training would fall "
            "back to a seeded random split. Generate the manifest with "
            '"python tools/data/build_oct_manifest.py --dir <image_root> --output %s" '
            "(point --dir at the dataset images, e.g. the extracted hub archive) and re-run.",
            extra_dir / "splits.csv",
            extra_dir / "manifest.csv",
        )


def _safe_cache_name(name: str) -> str:
    return name.replace("/", "__").replace("\\", "__").replace(":", "__")

def _hf_archive_cache_base(*, repo_id: str, revision: str) -> Path:
    safe_repo = _safe_cache_name(repo_id)
    safe_rev = _safe_cache_name(revision)
    return _DINOCT_CACHE_DIR / "datasets" / "hf_archive" / safe_repo / safe_rev


def _extract_hf_archive(*, archive_path: Path, base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    extracted_dir = base / "extracted"
    if extracted_dir.is_dir():
        return extracted_dir

    tmp_dir = base / "extracted.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, mode="r:*") as tf:
        tf.extractall(path=tmp_dir, filter=tarfile.data_filter)

    tmp_dir.rename(extracted_dir)
    return extracted_dir


def make_dataset(
    *,
    dataset_str: str,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
):
    """
    Creates a dataset with the specified parameters.

    Args:
        dataset_str: A dataset string description (e.g. OCT:root=data/oct:extra=data/oct/extra).
        transform: A transform to apply to images.
        target_transform: A transform to apply to targets.

    Returns:
        The created dataset.
    """
    logger.info(f'using dataset: "{dataset_str}"')

    class_, kwargs = _parse_dataset_str(dataset_str)
    dataset = class_(transform=transform, target_transform=target_transform, **kwargs)

    logger.info(f"# of dataset samples: {len(dataset):,d}")

    # Aggregated datasets do not expose (yet) these attributes, so add them.
    if not hasattr(dataset, "transform"):
        setattr(dataset, "transform", transform)
    if not hasattr(dataset, "target_transform"):
        setattr(dataset, "target_transform", target_transform)

    return dataset


T = TypeVar("T")


def make_data_loader(
    *,
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    seed: int = 0,
    drop_last: bool = True,
    persistent_workers: bool = False,
    collate_fn: Optional[Callable[[list[T]], Any]] = None,
):
    """
    Creates a data loader with the specified parameters.

    Args:
        dataset: A dataset instance (OCT only in this project).
        batch_size: The size of batches to generate.
        num_workers: The number of workers to use.
        shuffle: Whether to shuffle samples.
        seed: The random seed to use.
        drop_last: Whether the last non-full batch of data should be dropped.
        persistent_workers: maintain the workers Dataset instances alive after a dataset has been consumed once.
        collate_fn: Function that performs batch collation
    """

    logger.info("using PyTorch data loader")
    # Safer pin_memory: default to GPU-only, allow override via env PIN_MEMORY={0,1}
    _pin_env = os.environ.get("PIN_MEMORY", "auto").strip().lower()
    if _pin_env in ("0", "false", "no"):
        use_pin = False
    elif _pin_env in ("1", "true", "yes"):
        use_pin = True
    else:
        use_pin = torch.cuda.is_available()
    if drop_last:
        try:
            dataset_len: int | None = len(dataset)
        except (TypeError, NotImplementedError):
            dataset_len = None
        if dataset_len is not None and dataset_len < int(batch_size):
            raise ValueError(
                f"drop_last=True with a dataset smaller than one batch "
                f"({dataset_len} samples < batch_size {batch_size}): every epoch yields "
                "zero batches, so step-driven training loops would spin forever. "
                "Reduce batch_size or pass drop_last=False."
            )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    data_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=bool(shuffle),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=use_pin,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
        generator=generator,
        worker_init_fn=seed_worker,
    )

    try:
        logger.info(f"# of batches: {len(data_loader):,d}")
    except TypeError:  # pragma: no cover
        logger.info("# of batches: <unknown>")
    return data_loader
