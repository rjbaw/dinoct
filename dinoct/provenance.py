"""Lightweight provenance stamping for checkpoints and evaluation summaries.

Everything here is best-effort: a missing git binary, a non-repo checkout, or an
unreadable file must never break training or evaluation, so helpers return
None/{} instead of raising.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path


def file_md5(path: str | Path | None) -> str | None:
    """MD5 hex digest of a file, or None if the file is missing/unreadable."""
    if path is None:
        return None
    try:
        try:
            digest = hashlib.md5(usedforsecurity=False)
        except TypeError:  # backend without the flag
            digest = hashlib.md5()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except (OSError, ValueError):
        # ValueError covers FIPS builds where md5 is unavailable entirely.
        return None


def git_state(repo_root: str | Path | None = None) -> dict[str, object]:
    """{"commit": <sha>, "dirty": <bool>} for the checkout AT repo_root, or {}.

    Returns {} when repo_root is not itself the top level of a git checkout — e.g. a
    pip-installed package whose venv happens to live inside some unrelated repo —
    so provenance is absent rather than misattributed.
    """
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if top.returncode != 0:
            return {}
        if Path(top.stdout.strip()).resolve() != root.resolve():
            return {}
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if commit.returncode != 0:
            return {}
        out: dict[str, object] = {"commit": commit.stdout.strip()}
    except (OSError, subprocess.SubprocessError):
        return {}
    # Dirty check is separate so a slow `git status` (huge untracked trees) degrades
    # to commit-only provenance instead of discarding the captured commit.
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if dirty.returncode == 0:
            out["dirty"] = bool(dirty.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def runtime_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for mod in ("torch", "torchvision", "numpy", "PIL"):
        try:
            versions[mod] = str(__import__(mod).__version__)
        except Exception:
            continue
    return versions


__all__ = ["file_md5", "git_state", "runtime_versions"]
