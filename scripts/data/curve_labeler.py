#!/usr/bin/env python3
"""
Interactive curve label editor for OCT images.

Features
- Loads curve labels from `<root>/labeled/*.txt` (by default `data/oct/labeled`).
- Finds corresponding images under `<root>/raw/` and `<root>/background/` (also tries `<root>/labeled/`).
- Supports label files formatted as either:
  - `N` floats (one per column), or
  - `N×2` table `(x, y)` (the second column is used; `x` is preserved on save).
- Mouse editing:
  - Left click/drag: set `y` at the nearest `x` sample.
  - Right click: set interpolation anchor (same as pressing `i` once).
- Keyboard controls to move along x and adjust y, interpolate, smooth, save, undo/redo, and navigate samples.

Assumptions
- Labels represent pixel coordinates in the image coordinate system (origin top-left; y increases downward).
- The number of y-values may differ from image width; labels are displayed over the image by resampling x positions.
- Image coordinate system: origin at top-left, y increases downward (matches PIL/imshow upper origin).

Usage
  python scripts/data/curve_labeler.py --dir data/oct
  python scripts/data/curve_labeler.py --dir data/oct --pattern "*_skin.txt"  # optional subset

Keys
- left/right: move selected x index by 1 (no Shift/Ctrl acceleration).
- PageUp/PageDown: jump ±50 columns; Home/End: go to start/end.
- up/down: adjust y at selected x by 1 (Shift=±5, Ctrl=±10).
- i: set first anchor; press i again to set second anchor and linearly interpolate between anchors.
- m: smooth local window around current x (moving average, width=9 by default).
- [: decrease edit step size (min 0.5); ]: increase step size.
- n/p: next/previous sample.
- N/P: next/previous sample, discarding unsaved changes.
- s: save labels to file.
- r: revert to last saved on disk.
- h: toggle help overlay.
- q or escape: quit.
- u: undo   |   U: redo
"""

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import matplotlib
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover
    matplotlib = None
    plt = None


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
UNDO_LIMIT = 200


@dataclass(frozen=True, slots=True)
class Sample:
    stem: str
    img_path: Path
    lbl_path: Path
    out_path: Path  # identical to lbl_path (in-place save)


@dataclass(slots=True)
class LabelData:
    y: np.ndarray
    x: np.ndarray | None = None


def _unique_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        out.append(item)
        seen.add(item)
    return out


def _candidate_stems(label_stem: str) -> list[str]:
    candidates = [label_stem]
    if "_" in label_stem:
        base, _suffix = label_stem.rsplit("_", 1)
        candidates.extend([base, f"{base}_current"])
    return _unique_keep_order([c for c in candidates if c])


def _resolve_dirs(dir_arg: Path) -> tuple[Path, Path, list[Path]]:
    """
    Accept either:
      - dataset root containing labeled/raw/background, or
      - labeled directory directly.
    """
    dir_arg = dir_arg.expanduser().resolve()
    if (dir_arg / "labeled").is_dir():
        dataset_root = dir_arg
        labels_dir = dataset_root / "labeled"
    else:
        labels_dir = dir_arg
        dataset_root = labels_dir.parent if labels_dir.name == "labeled" else labels_dir

    image_dirs = [dataset_root / "raw", dataset_root / "background", labels_dir]
    image_dirs = [p for p in image_dirs if p.is_dir()]
    return dataset_root, labels_dir, image_dirs


def _iter_label_files(labels_dir: Path, pattern: str | None) -> list[Path]:
    if pattern:
        paths = [Path(p) for p in glob.glob(str(labels_dir / pattern), recursive=True)]
    else:
        paths = list(labels_dir.glob("*.txt"))
    return sorted([p for p in paths if p.is_file() and p.suffix.lower() == ".txt"])


def _find_image_for_label(stem: str, image_dirs: list[Path]) -> Path | None:
    for cand in _candidate_stems(stem):
        for img_dir in image_dirs:
            for ext in IMAGE_EXTS:
                p = img_dir / f"{cand}{ext}"
                if p.exists():
                    return p
    return None


def find_samples(dir_arg: Path, pattern: str | None) -> list[Sample]:
    dataset_root, labels_dir, image_dirs = _resolve_dirs(dir_arg)
    txt_paths = _iter_label_files(labels_dir, pattern)

    samples: list[Sample] = []
    skipped: list[Path] = []
    for tpath in txt_paths:
        img_path = _find_image_for_label(tpath.stem, image_dirs)
        if img_path is None:
            skipped.append(tpath)
            continue

        samples.append(Sample(stem=tpath.stem, img_path=img_path, lbl_path=tpath, out_path=tpath))

    if not samples:
        msg = f"No matching (label,image) pairs found under {dataset_root} (labels in {labels_dir})."
        if skipped:
            msg += f" Missing images for {len(skipped)} label file(s)."
        raise SystemExit(msg)
    if skipped:
        print(f"[curve_labeler] warning: skipped {len(skipped)} label file(s) with no matching image.")
    return samples


def read_labels(path: Path) -> LabelData:
    arr = np.loadtxt(path, ndmin=2)
    if arr.size == 0:
        raise ValueError(f"{path} is empty; expected at least one value.")
    x: np.ndarray | None = None
    if arr.shape[1] == 2:
        x = arr[:, 0].astype(np.float64, copy=False)
        y = arr[:, 1].astype(np.float64, copy=False)
    elif arr.shape[1] == 1:
        y = arr[:, 0].astype(np.float64, copy=False)
    elif arr.shape[0] == 1:
        # A single row of values (e.g., written as "1×N") -> treat as vector.
        y = arr[0].astype(np.float64, copy=False)
    else:
        raise ValueError(f"{path} has unexpected shape {arr.shape}; expected (N,), (N,1), or (N,2)")
    return LabelData(y=y, x=x)


def write_labels(path: Path, labels: LabelData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if labels.x is None:
        np.savetxt(path, labels.y, fmt="%.6f")
    else:
        xy = np.stack([labels.x, labels.y], axis=1)
        np.savetxt(path, xy, fmt="%.6f")


class CurveEditor:
    def __init__(
        self,
        samples: list[Sample],
        *,
        start: int = 0,
        step: float = 1.0,
        smooth_w: int = 9,
        autosave: bool = False,
    ):
        if matplotlib is None or plt is None:
            raise SystemExit(
                "matplotlib is required for the interactive curve labeler.\nInstall: uv sync --extra label\n"
            )
        if not samples:
            raise SystemExit("No samples found. Check --dir and optional --pattern.")
        self.samples = samples
        self.index = max(0, min(start, len(samples) - 1))
        self.step = step
        self.smooth_w = smooth_w if smooth_w % 2 == 1 else smooth_w + 1
        self.autosave = bool(autosave)

        # State for current sample
        self.image: np.ndarray | None = None
        self.labels: LabelData | None = None
        self.labels_saved: LabelData | None = None
        self.x_positions: np.ndarray | None = None
        self.height: int | None = None
        self.width: int | None = None
        self.cur_xi = 0  # index into y
        self.anchor: int | None = None
        self.undo_stack: list[np.ndarray] = []
        self.redo_stack: list[np.ndarray] = []
        self.help_on = True
        self.toast: str = ""
        self._quit_armed = False

        # Matplotlib fig
        self.fig, self.ax = plt.subplots(figsize=(9, 8))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("button_press_event", self.on_mouse_down)
        self.fig.canvas.mpl_connect("button_release_event", self.on_mouse_up)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

        self.im_artist = self.ax.imshow(np.zeros((16, 16, 3), dtype=np.uint8), origin="upper")
        (self.line_artist,) = self.ax.plot([], [], "-", color="lime", lw=1.5, alpha=0.9)
        (self.marker_artist,) = self.ax.plot([], [], "o", color="red", ms=6)
        (self.anchor_artist,) = self.ax.plot([], [], "o", color="cyan", ms=6, alpha=0.9)
        self.text_artist = self.ax.text(
            0.01,
            0.01,
            "",
            transform=self.ax.transAxes,
            va="bottom",
            ha="left",
            color="yellow",
            fontsize=9,
            bbox={"facecolor": "black", "alpha": 0.4, "pad": 4},
        )

        self._dragging = False
        self._drag_last_idx: int | None = None
        self._drag_pushed_undo = False

        self.load_current()
        self.redraw()

    def _compute_x_positions(self) -> np.ndarray:
        assert self.labels is not None
        assert self.width is not None
        n = int(self.labels.y.shape[0])
        if n <= 0:
            return np.array([], dtype=np.float64)

        if self.labels.x is not None and self.labels.x.shape[0] == n:
            x = self.labels.x.astype(np.float64, copy=False)
            x_min = float(np.nanmin(x))
            x_max = float(np.nanmax(x))
            if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
                return (x - x_min) / (x_max - x_min) * float(max(self.width - 1, 1))

        return np.linspace(0, max(self.width - 1, 1), n, dtype=np.float64)

    def load_current(self):
        sample = self.samples[self.index]
        # Load image
        im = Image.open(sample.img_path).convert("RGB")
        self.width, self.height = im.size
        self.image = np.array(im)
        # Load labels
        self.labels = read_labels(sample.lbl_path)
        self.labels_saved = LabelData(
            y=self.labels.y.copy(), x=(self.labels.x.copy() if self.labels.x is not None else None)
        )
        n = int(self.labels.y.shape[0])

        # X positions used only for display/mouse mapping (labels.x is preserved for saving).
        self.x_positions = self._compute_x_positions()

        # Clamp y to image bounds
        self.labels.y = np.clip(self.labels.y, 0, self.height - 1)
        self.cur_xi = int(np.clip(self.cur_xi, 0, n - 1)) if n > 0 else 0
        self.anchor = None
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.toast = ""
        self._quit_armed = False
        self._dragging = False
        self._drag_last_idx = None
        self._drag_pushed_undo = False

    def is_dirty(self) -> bool:
        if self.labels is None or self.labels_saved is None:
            return False
        return not np.allclose(self.labels.y, self.labels_saved.y)

    def push_undo(self):
        assert self.labels is not None
        self.undo_stack.append(self.labels.y.copy())
        if len(self.undo_stack) > UNDO_LIMIT:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def save(self):
        assert self.labels is not None
        sample = self.samples[self.index]
        write_labels(sample.out_path, self.labels)
        self.labels_saved = LabelData(
            y=self.labels.y.copy(), x=(self.labels.x.copy() if self.labels.x is not None else None)
        )
        self.toast = "saved"
        self._quit_armed = False

    def revert(self):
        sample = self.samples[self.index]
        path = sample.out_path if sample.out_path.exists() else sample.lbl_path
        self.labels = read_labels(path)
        assert self.height is not None
        self.labels.y = np.clip(self.labels.y, 0, self.height - 1)
        self.labels_saved = LabelData(
            y=self.labels.y.copy(), x=(self.labels.x.copy() if self.labels.x is not None else None)
        )
        n = int(self.labels.y.shape[0])
        self.x_positions = self._compute_x_positions()
        self.cur_xi = int(np.clip(self.cur_xi, 0, n - 1)) if n > 0 else 0
        self.anchor = None
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.toast = "reverted"
        self._quit_armed = False

    def interpolate_between(self, i0: int, i1: int):
        assert self.labels is not None
        if i0 == i1:
            return
        if i0 > i1:
            i0, i1 = i1, i0
        self.push_undo()
        y0, y1 = float(self.labels.y[i0]), float(self.labels.y[i1])
        k = i1 - i0
        if k <= 1:
            return
        self.labels.y[i0 : i1 + 1] = np.linspace(y0, y1, k + 1)

    def smooth_at(self, center_idx: int):
        assert self.labels is not None
        w = self.smooth_w
        half = w // 2
        i0 = max(0, center_idx - half)
        i1 = min(len(self.labels.y) - 1, center_idx + half)
        seg = self.labels.y[i0 : i1 + 1]
        k = int(min(len(seg), w))
        if k <= 1:
            return
        kernel = np.ones(k, dtype=np.float64) / float(k)
        pad = k // 2
        seg_pad = np.pad(seg, (pad, pad), mode="edge")
        smoothed = np.convolve(seg_pad, kernel, mode="valid")
        self.push_undo()
        self.labels.y[i0 : i1 + 1] = smoothed[: len(seg)]

    def move_index(self, delta: int):
        assert self.labels is not None
        n = int(self.labels.y.shape[0])
        self.cur_xi = int(np.clip(self.cur_xi + delta, 0, n - 1)) if n > 0 else 0

    def adjust_y(self, delta: float):
        assert self.labels is not None
        assert self.height is not None
        self.push_undo()
        self.labels.y[self.cur_xi] = float(np.clip(self.labels.y[self.cur_xi] + delta, 0, self.height - 1))

    def compute_step(self, key: str) -> float:
        # key may contain modifiers like 'shift+up'
        base = self.step
        mult = 1.0
        if "shift+" in key:
            mult *= 5.0
        if "ctrl+" in key or "control+" in key:
            mult *= 10.0
        return base * mult

    def status(self) -> str:
        s = self.samples[self.index]
        assert self.labels is not None
        assert self.x_positions is not None
        changed = "*" if self.is_dirty() else ""
        autosave = "on" if self.autosave else "off"
        msg = (
            f"[{self.index + 1}/{len(self.samples)}] {s.img_path.name} | {s.lbl_path.name}{changed}\n"
            f"x[{self.cur_xi}]={self.x_positions[self.cur_xi]:.1f}  y={self.labels.y[self.cur_xi]:.1f}  step={self.step:.2f}  smooth_w={self.smooth_w}  autosave={autosave}"
        )
        if self.anchor is not None:
            msg += f"\nanchor={self.anchor}"
        if self.toast:
            msg += f"\n{self.toast}"
        if self.help_on:
            msg += (
                "\nleft/right: move (no Shift/Ctrl accel)  |  up/down: adjust (Shift×5, Ctrl×10)\n"
                "PageUp/PageDown: ±50 cols   Home/End: start/end\n"
                "[ / ]: step –/+    m: smooth window    i/right click: interp anchors\n"
                "n/p: next/prev     N/P: discard+move   s: save   r: revert\n"
                "u: undo   U: redo  h: toggle help      q/ESC: quit"
            )
        return msg

    def redraw(self):
        assert self.image is not None
        assert self.width is not None and self.height is not None
        assert self.labels is not None
        assert self.x_positions is not None

        self.im_artist.set_data(self.image)
        self.im_artist.set_extent([-0.5, self.width - 0.5, self.height - 0.5, -0.5])
        self.ax.set_title(f"Curve Label Editor — {self.samples[self.index].img_path.name}")
        self.ax.set_xlim([-0.5, self.width - 0.5])
        self.ax.set_ylim([self.height - 0.5, -0.5])

        self.line_artist.set_data(self.x_positions, self.labels.y)
        self.marker_artist.set_data([self.x_positions[self.cur_xi]], [self.labels.y[self.cur_xi]])
        if self.anchor is None:
            self.anchor_artist.set_data([], [])
        else:
            self.anchor_artist.set_data([self.x_positions[self.anchor]], [self.labels.y[self.anchor]])

        self.text_artist.set_text(self.status())
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        key = event.key or ""
        self.toast = ""

        if key in ("q", "escape"):
            if self.is_dirty() and not self.autosave:
                if self._quit_armed:
                    plt.close(self.fig)
                    return
                self._quit_armed = True
                self.toast = "unsaved changes: press q/ESC again to quit without saving"
                self.redraw()
                return
            if self.autosave and self.is_dirty():
                self.save()
            plt.close(self.fig)
            return

        # Navigation
        if key.endswith("left"):
            self.move_index(-1)
            self.redraw()
            return
        if key.endswith("right"):
            self.move_index(+1)
            self.redraw()
            return
        if key == "pageup":
            self.move_index(+50)
            self.redraw()
            return
        if key == "pagedown":
            self.move_index(-50)
            self.redraw()
            return
        if key == "home":
            self.cur_xi = 0
            self.redraw()
            return
        if key == "end":
            assert self.labels is not None
            self.cur_xi = len(self.labels.y) - 1
            self.redraw()
            return
        if key.endswith("up"):
            self.adjust_y(-self.compute_step(key))
            self.redraw()
            return
        if key.endswith("down"):
            self.adjust_y(+self.compute_step(key))
            self.redraw()
            return

        if key in {"n", "p", "N", "P"}:
            delta = 1 if key.lower() == "n" else -1
            force_discard = key.isupper()
            if self.is_dirty() and not self.autosave and not force_discard:
                self.toast = "unsaved changes: press s to save, r to revert, or N/P to discard+move"
                self.redraw()
                return
            if self.is_dirty() and self.autosave:
                self.save()
            new_index = int(np.clip(self.index + delta, 0, len(self.samples) - 1))
            if new_index != self.index:
                self.index = new_index
                self.load_current()
            self.redraw()
            return

        if key == "s":
            self.save()
            self.redraw()
            return
        if key == "r":
            self.revert()
            self.redraw()
            return
        if key == "h":
            self.help_on = not self.help_on
            self.redraw()
            return
        if key == "[":
            self.step = max(0.5, self.step / 2.0)
            self.redraw()
            return
        if key == "]":
            self.step = min(50.0, self.step * 2.0)
            self.redraw()
            return
        if key == "i":
            if self.anchor is None:
                self.anchor = self.cur_xi
            else:
                self.interpolate_between(self.anchor, self.cur_xi)
                self.anchor = None
                self.redraw()
            return
        if key == "m":
            self.smooth_at(self.cur_xi)
            self.redraw()
            return
        if key == "u":
            if self.undo_stack:
                assert self.labels is not None
                self.redo_stack.append(self.labels.y.copy())
                self.labels.y = self.undo_stack.pop()
                self.toast = "undo"
                self._quit_armed = False
                self.redraw()
            return
        if key == "U":
            if self.redo_stack:
                assert self.labels is not None
                self.undo_stack.append(self.labels.y.copy())
                self.labels.y = self.redo_stack.pop()
                self.toast = "redo"
                self._quit_armed = False
                self.redraw()
            return

    def on_mouse_down(self, event):
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self.labels is None or self.x_positions is None or self.width is None or self.height is None:
            return

        if event.button == 3:  # right click: set anchor
            idx = self._nearest_index(float(event.xdata))
            self.anchor = idx
            self.toast = f"anchor set: {idx}"
            self._quit_armed = False
            self.redraw()
            return

        if event.button != 1:  # left click only
            return

        self._dragging = True
        self._drag_last_idx = None
        self._drag_pushed_undo = False
        self._apply_mouse_edit(float(event.xdata), float(event.ydata))

    def on_mouse_move(self, event):
        if not self._dragging:
            return
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._apply_mouse_edit(float(event.xdata), float(event.ydata))

    def on_mouse_up(self, event):
        if event.button != 1:
            return
        self._dragging = False
        self._drag_last_idx = None
        self._drag_pushed_undo = False

    def _nearest_index(self, x: float) -> int:
        assert self.labels is not None
        assert self.x_positions is not None
        n = int(self.labels.y.shape[0])
        if n <= 1:
            return 0
        # Use x_positions if it isn't strictly increasing; fall back to linear mapping.
        try:
            idx = int(np.argmin(np.abs(self.x_positions - x)))
        except Exception:
            idx = int(round(x / float(max(self.width - 1, 1)) * float(n - 1)))
        return int(np.clip(idx, 0, n - 1))

    def _apply_mouse_edit(self, x: float, y: float) -> None:
        assert self.labels is not None
        assert self.width is not None and self.height is not None
        idx = self._nearest_index(x)
        if self._drag_last_idx == idx and self._drag_pushed_undo:
            # Avoid spamming updates when staying on the same sample index.
            return
        if not self._drag_pushed_undo:
            self.push_undo()
            self._drag_pushed_undo = True
        self._drag_last_idx = idx
        self.cur_xi = idx
        self.labels.y[idx] = float(np.clip(y, 0, self.height - 1))
        self._quit_armed = False
        self.redraw()

    def run(self):
        # Use interactive backend if possible
        if matplotlib.get_backend().lower().startswith("agg"):
            print("Warning: non-interactive matplotlib backend detected. Set MPLBACKEND or run locally with a GUI.")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Interactive curve label editor")
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("data/oct"),
        help="Dataset root (containing labeled/raw/background) or the labeled/ directory itself",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help='Optional glob (relative to labeled/) to select subset of label files (e.g., "*_skin.txt")',
    )
    parser.add_argument("--start", type=int, default=0, help="Start index in the sample list")
    parser.add_argument("--step", type=float, default=1.0, help="Base y-step in pixels for up/down")
    parser.add_argument("--smooth-w", type=int, default=9, help="Smoothing window size (odd) for m key")
    parser.add_argument("--autosave", action="store_true", help="Auto-save on sample change and on quit")
    args = parser.parse_args()

    if matplotlib is None or plt is None:
        raise SystemExit(
            "matplotlib is required for the interactive curve label editor.\nInstall: uv sync --extra label\n"
        )

    samples = find_samples(args.dir, args.pattern)
    editor = CurveEditor(samples, start=args.start, step=args.step, smooth_w=args.smooth_w, autosave=args.autosave)
    editor.run()


if __name__ == "__main__":
    main()
