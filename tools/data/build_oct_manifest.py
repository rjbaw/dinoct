#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
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
    build_group_index_rows,
    build_manifest_rows,
    summarize_manifest,
    write_group_index_csv,
    write_manifest_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build experiment metadata for the OCT dataset.")
    parser.add_argument("--dir", type=Path, default=Path("data/oct"), help="Dataset root containing raw/, background/, labeled/")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Manifest CSV path. Defaults to <dir>/extra/manifest.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional summary JSON path. Defaults to <dir>/extra/manifest_summary.json",
    )
    parser.add_argument(
        "--groups",
        type=Path,
        default=None,
        help="Optional auto-generated group index CSV. Defaults to <dir>/extra/groups_auto.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dir.expanduser()
    manifest_path = args.output.expanduser() if args.output else dataset_root / "extra" / "manifest.csv"
    summary_path = args.summary.expanduser() if args.summary else dataset_root / "extra" / "manifest_summary.json"
    groups_path = args.groups.expanduser() if args.groups else dataset_root / "extra" / "groups_auto.csv"

    rows = build_manifest_rows(dataset_root)
    write_manifest_csv(manifest_path, rows)
    write_group_index_csv(groups_path, build_group_index_rows(rows))

    summary = summarize_manifest(rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"[manifest] wrote {len(rows)} rows to {manifest_path}")
    print(f"[manifest] auto group index saved to {groups_path}")
    print(f"[manifest] summary saved to {summary_path}")
    for key, value in summary.items():
        print(f"[manifest] {key}={value}")


if __name__ == "__main__":
    main()
