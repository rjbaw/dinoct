#!/usr/bin/env python3

import argparse
import os
import shutil
from pathlib import Path


def renamed_with_dir(dest: Path, dir_name: str) -> Path:
    return dest.with_name(f"{dest.stem}_{dir_name}{dest.suffix}")


def flatten(root: Path) -> None:
    root = root.resolve()

    for current_dir, _subdirs, files in os.walk(root, topdown=False):
        current_path = Path(current_dir)
        if current_path == root:
            continue

        for fname in files:
            src = current_path / fname
            dest = root / fname
            dest = renamed_with_dir(dest, current_path.name)
            shutil.move(src, dest)
            print(f"{dest.name}")

        current_path.rmdir()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory",
        default=".",
    )
    args = parser.parse_args()
    flatten(Path(args.directory))
