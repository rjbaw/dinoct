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
      - Hugging Face Hub dataset repo: OCT:hub=<user/dataset>[:revision=...][:subdir=...][:cache_dir=...][:extra=...]
    """
    if "split" in kwargs:
        raise ValueError('OCT dataset does not support "split=". Remove it from the dataset string.')

    root = kwargs.get("root")
    prefer_local = root is not None and Path(root).exists()

    if "hub" in kwargs and not prefer_local:
        repo_id = kwargs.pop("hub")
        kwargs.pop("root", None)

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
            kwargs["extra"] = str(base / "extra")
    else:
        if "hub" in kwargs:
            # Prefer local root if it exists.
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

    extra_keys = set(kwargs) - {"root", "extra"}
    if extra_keys:
        raise ValueError(f"OCT dataset has unsupported options: {sorted(extra_keys)}")

    return kwargs


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
    )

    try:
        logger.info(f"# of batches: {len(data_loader):,d}")
    except TypeError:  # pragma: no cover
        logger.info("# of batches: <unknown>")
    return data_loader
