# dinoct

DINO-style self-supervised pretraining for OCT B-scan images, plus curve head (LoRA) post-train stage.

## Repo layout

- `dinoct/`: Python package (models, data, training)
- `configs/`: YAML configs (merged: `configs/ssl_default_config.yaml` + `configs/train/oct.yaml`)
- `scripts/`: entrypoints and utilities

## Quick Start

- Python: `>=3.12`
- CUDA: single GPU

This repo uses `uv`:

```bash
uv sync
uv run python -m dinoct --help
```

## Data

Default expected layout under `data/oct/`:

- `data/oct/raw/*.jpg`
- `data/oct/background/*.jpg`
- `data/oct/labeled/<image_stem>.txt` (optional; marks an image as labeled)
- `data/oct/extra/entries.npy` (metadata cache; regenerated each run)

Each label file should contain either:
- 500 floats (one per column), or
- a 500×2 table `(x, y)` (the second column is used).

You can change the dataset paths via the dataset string: `oct:root=<root>:extra=<extra>` (see `configs/train/oct.yaml`).
In that string, `root`/`extra` refer to dataset directories (not the repo root).
The dataset name token is case-insensitive.

## Labeling (curve editor)

The interactive curve label editor requires `matplotlib`:

```bash
uv sync --extra label
uv run python scripts/data/curve_labeler.py --dir data/oct
```

## Pretrain (SSL)

```bash
uv run python -m dinoct \
  --config configs/train/oct.yaml \
  --output-dir outputs/run1 \
  --steps 10000 \
  --post-train-steps 0
```

Outputs:

- `outputs/run1/pretrain/dinov3_pretrain.pth`
- `outputs/run1/pretrain/train.log`, `metrics.csv`, `config_used.yaml`

## Post-train (curve head)

```bash
uv run python -m dinoct \
  --config configs/train/oct.yaml \
  --output-dir outputs/run1 \
  --steps 10000 \
  --post-train-steps 1000
```

To run post-train only (with an existing pretrain checkpoint):

```bash
uv run python -m dinoct \
  --config configs/train/oct.yaml \
  --output-dir outputs/run1 \
  --post-train-only \
  --pretrained-backbone outputs/run1/pretrain/dinov3_pretrain.pth \
  --post-train-steps 1000
```

Outputs:

- `outputs/run1/post_train/fused_curve.pth`
- `outputs/run1/post_train/fused_curve_best.pth`
- `outputs/run1/post_train/val_summary.json` (validation metrics for final + best checkpoint)

## Visualizations

```bash
uv run python scripts/visualize.py \
  --mode curve \
  --curve-ckpt outputs/run1/post_train/fused_curve.pth \
  --input path/to/image_or_dir \
  --outdir outputs/viz
```

## Export (TorchScript/ONNX)

```bash
uv run python scripts/export_model.py --model outputs/run1/post_train/fused_curve.pth --outdir exports
```

## License
Apache-2.0.  

This project includes alot of code derived from Meta Platforms, Inc. and affiliates' DINOv2 and DINOv3 repositories, licensed under the Apache License, Version 2.0.
