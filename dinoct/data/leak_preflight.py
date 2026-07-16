"""Fail-closed leak preflight for SSL pretraining.

Aborts the run BEFORE training if the resolved SSL pool would leak the eval set.

Two independent guards (either firing aborts the run):
  1. ssl_split ALLOWLIST. real_hard's source recordings (group_ids
     `continuous:new`, `continuous:new_capture`) live in the `exclude` split.
     ssl_split='train' filters them out, but 'all'/'none'/'' silently re-admit
     them. So when splits.csv exists we REQUIRE ssl_split to be in the
     allowlist (default ('train',)).
  2. BYTE-IDENTICAL md5 screen. Every eval image is md5'd; only SSL-pool files
     whose file SIZE matches an eval image are md5'd (size pre-filter -> fast),
     and any md5 collision aborts. This catches byte-duplicated eval images
     regardless of split bookkeeping.

CPU-only, one-time at startup; the size pre-filter keeps it to well under a
second on the OCT pool (only a handful of SSL files share an eval size).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterable

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _gather_images(dirs: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        out += [p for p in sorted(d.rglob("*")) if p.suffix.lower() in _IMG_EXTS]
    return out


def assert_no_ssl_leak(
    *,
    ssl_files: list[Path],
    eval_dirs: Iterable[Path],
    ssl_split: str,
    splits_exist: bool,
    allow_splits: tuple[str, ...] = ("train",),
    deep_md5: bool = True,
    logger: logging.Logger | None = None,
) -> None:
    """Raise RuntimeError if the SSL pool leaks the eval set. Fail-closed."""
    log = logger or logging.getLogger(__name__)
    ssl_split = str(ssl_split).strip().lower()
    allow_splits = tuple(str(s).strip().lower() for s in allow_splits)

    # --- Guard 0: the allowlist itself must not name a leaky/held-out split ----
    _unsafe = {"", "all", "none", "false", "off", "exclude"}
    _bad = sorted(s for s in allow_splits if s in _unsafe)
    if _bad:
        raise RuntimeError(
            f"LEAK PREFLIGHT FAILED: train.leak_allow_splits contains unsafe split(s) {_bad} that "
            f"would admit held-out/leaked recordings. Use a real-split allowlist such as ('train',)."
        )
    if not allow_splits:
        raise RuntimeError("LEAK PREFLIGHT FAILED: train.leak_allow_splits is empty.")

    # --- Guard 1: ssl_split allowlist -----------------------------------------
    if splits_exist:
        if ssl_split not in allow_splits:
            raise RuntimeError(
                f"LEAK PREFLIGHT FAILED: ssl_split={ssl_split!r} is not in the allowed set "
                f"{tuple(allow_splits)} while the data carries splits and {len(ssl_files)} SSL files "
                f"resolved. An unfiltered/'all'/'exclude' pool re-admits the held-out and leaked "
                f"(continuous:new / continuous:new_capture) recordings. "
                f"Set train.ssl_split='train' (or add the split to train.leak_allow_splits)."
            )
    else:
        log.warning(
            "LEAK PREFLIGHT: data carries no split metadata -> cannot verify split filtering; "
            "relying on the md5 byte-screen ONLY."
        )

    # --- Guard 1b: SSL file-list coverage (catch wrong/extracted-root silent degradation) ---
    n_total = len(ssl_files) if ssl_files is not None else 0
    if n_total == 0:
        raise RuntimeError(
            "LEAK PREFLIGHT FAILED: resolved SSL file list is EMPTY -> nothing to screen "
            "(likely a root-resolution bug). Refusing to train."
        )
    n_exist = sum(1 for f in ssl_files if Path(f).exists())
    if n_exist < int(0.95 * n_total):
        raise RuntimeError(
            f"LEAK PREFLIGHT FAILED: only {n_exist}/{n_total} resolved SSL files exist on disk "
            f"(<95%). The md5 screen would cover almost nothing -> the SSL pool root is probably "
            f"wrong (config-string vs hub-extracted). Point the preflight at the dataset's REAL root."
        )

    # --- Guard 2: byte-identical md5 screen (size pre-filtered) ---------------
    eval_imgs = _gather_images(eval_dirs)
    if not eval_imgs:
        if deep_md5:
            raise RuntimeError(
                f"LEAK PREFLIGHT FAILED: the md5 byte-screen is ON but NO eval images were found "
                f"under {[str(d) for d in eval_dirs]} -> cannot verify non-leakage. Set "
                f"train.leak_eval_dirs to the eval set, or disable train.leak_preflight_md5 explicitly."
            )
        log.warning(
            "LEAK PREFLIGHT (split-only mode): no eval images under %s; split-allowlist guard "
            "enforced, md5 screen not run.",
            [str(d) for d in eval_dirs],
        )
        return
    if not deep_md5:
        log.info(
            "LEAK PREFLIGHT PASS (split-only): ssl_split=%r in %s; %d eval images NOT md5-screened.",
            ssl_split, tuple(allow_splits), len(eval_imgs),
        )
        return

    # eval index: size -> {md5 -> name}
    eval_by_size: dict[int, dict[str, str]] = {}
    for p in eval_imgs:
        try:
            sz = p.stat().st_size
        except OSError:
            continue
        eval_by_size.setdefault(sz, {})[_md5(p)] = p.name
    eval_sizes = set(eval_by_size)

    hits: list[tuple[str, str]] = []
    n_hashed = 0
    for sp in ssl_files:
        sp = Path(sp)
        try:
            sz = sp.stat().st_size
        except OSError:
            continue
        if sz not in eval_sizes:  # size pre-filter: byte-identical => identical size
            continue
        n_hashed += 1
        m = _md5(sp)
        if m in eval_by_size[sz]:
            hits.append((eval_by_size[sz][m], str(sp)))
            if len(hits) >= 50:
                break

    if hits:
        ex = "; ".join(f"{e} == {s}" for e, s in hits[:5])
        raise RuntimeError(
            f"LEAK PREFLIGHT FAILED: >={len(hits)} eval image(s) are BYTE-IDENTICAL to images in "
            f"the resolved SSL pool (e.g. {ex}). The SSL backbone would be pretrained on the eval "
            f"set. Fix train.ssl_split / the dataset before training."
        )

    log.info(
        "LEAK PREFLIGHT PASS: 0/%d eval images byte-identical to the %d-file SSL pool "
        "(%d size-matched files md5'd); ssl_split=%r in %s.",
        len(eval_imgs), len(ssl_files), n_hashed, ssl_split, tuple(allow_splits),
    )
