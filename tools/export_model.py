#!/usr/bin/env python3
"""Export the DINOCT CurveModel (LoRA-fused) to TorchScript/ONNX.

Contract of both exports: image [batch, 3, 512, 500] float32
    -> presence_logits [batch] (image-level curve-presence logit),
       z_vec [batch, 500]     (per-column curve row, soft-argmax, in the 512-row frame).

The graphs contain NO resize/normalization. Callers must replicate the training
preprocessing exactly (see "Inference preprocessing" in README.md): RGB decode
(grayscale sources get replicated channels) -> bicubic resize to 512x500 ->
float [0,1] -> per-image z-score (population std, eps=1e-6). Reference:
_make_oct_transform in dinoct/train/post_train.py.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").exists() and (candidate / "dinoct").is_dir():
            return candidate
    raise RuntimeError("Could not locate repo root from script path.")


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dinoct.models import build_backbone, native_patch_size_for_backbone  # noqa: E402
from dinoct.train.post_train import (  # noqa: E402
    CurveModel,
    LoRALinear,
    ORIG_H,
    ORIG_W,
    remap_legacy_curve_head_keys,
)

log = logging.getLogger("export")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def pick_device(device_arg: str = "auto") -> torch.device:
    want = str(device_arg).strip().lower()
    if want == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # pragma: no cover
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(want)


@torch.no_grad()
def fuse_all_lora_(module: nn.Module) -> int:
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            delta = child.lora_B @ child.lora_A
            child.base.weight += child.scaling * delta
            child.lora_A.zero_()
            child.lora_B.zero_()
            setattr(module, name, child.base)
            count += 1
        else:
            count += fuse_all_lora_(child)
    return count


def soft_argmax_height_jit_safe(logits_hw: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits_hw, dim=1)
    grid = torch.cumsum(torch.ones_like(logits_hw[:, :, :1], dtype=logits_hw.dtype), dim=1) - 1.0
    return (p * grid).sum(dim=1)


class ExportWrapper(nn.Module):
    def __init__(self, model: CurveModel):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        presence_logits, curve_logits = self.model(x)
        z_logits = curve_logits[:, :-1, :]
        z_vec = soft_argmax_height_jit_safe(z_logits)
        return presence_logits, z_vec


def infer_backbone_from_curve_state_dict(state_dict: dict) -> str:
    keys = list(state_dict.keys())
    if any(k.startswith("backbone.stages.") or k.startswith("backbone.downsample_layers.") for k in keys):
        stage2_re = re.compile(r"^backbone\.stages\.2\.(\d+)\.")
        stage2_idx: list[int] = []
        for k in keys:
            m = stage2_re.match(k)
            if m:
                stage2_idx.append(int(m.group(1)))
        if stage2_idx:
            depth2 = max(stage2_idx) + 1
            return "convnext_small" if depth2 >= 27 else "convnext_tiny"
        return "convnext_tiny"

    embed_dim = None
    w = state_dict.get("pres_head.net.0.weight", None)
    if isinstance(w, torch.Tensor) and w.ndim == 2:
        embed_dim = int(w.shape[1])
    w = state_dict.get("backbone.norm.weight", None)
    if embed_dim is None and isinstance(w, torch.Tensor) and w.ndim == 1:
        embed_dim = int(w.shape[0])

    if embed_dim == 384:
        return "small"
    raise ValueError(
        f"Could not infer a supported backbone from checkpoint (embed_dim={embed_dim}). "
        "Supported: small, convnext_tiny, convnext_small."
    )


_LORA_BLOCK_RE = re.compile(r"^(backbone\.(?:blocks\.\d+|stages\.\d+\.\d+))\.")


def infer_lora_from_state_dict(state_dict: dict) -> tuple[int, int]:
    """Return (num_lora_blocks, lora_r) from checkpoint keys; (0, 8) if already fused."""
    blocks: set[str] = set()
    r = 8
    for k, v in state_dict.items():
        if k.endswith(".lora_A"):
            m = _LORA_BLOCK_RE.match(k)
            if m:
                blocks.add(m.group(1))
            if isinstance(v, torch.Tensor) and v.ndim == 2:
                r = int(v.shape[0])
    return len(blocks), r


def build_model(
    backbone_name: str, model_path: str, device: torch.device, patch_size: int = 0
) -> CurveModel:
    if not (model_path and os.path.exists(model_path)):
        raise FileNotFoundError(f"Fused model checkpoint not found: {model_path}")

    log.info(f"Loading fused DINOCT curve model: {model_path}")
    sd = torch.load(model_path, map_location="cpu")
    sd = sd.get("model", sd)

    if backbone_name == "auto":
        backbone_name = infer_backbone_from_curve_state_dict(sd)
        log.info(f"Auto-detected backbone: {backbone_name}")

    lora_blocks, lora_r = infer_lora_from_state_dict(sd)
    if lora_blocks > 0:
        log.warning(
            f"Checkpoint has un-fused LoRA on {lora_blocks} blocks (r={lora_r}); "
            "assuming alpha=16 for fusing — verify this matches the training config."
        )
    else:
        log.info("Checkpoint is already fused (no LoRA keys); building plain model.")

    arch = backbone_name.replace("vit_", "") if backbone_name.startswith("vit_") else backbone_name
    # patch_size sets inference geometry only (pad-to-multiple + token grid) for ConvNeXt,
    # so it cannot be inferred from checkpoint weights and MUST match the training config's
    # student.patch_size — a mismatch loads cleanly but computes garbage.
    if patch_size <= 0:
        if arch.startswith("convnext"):
            patch_size = 14
            log.warning(
                "No --patch-size given; assuming patch_size=14 (dinoct training convention). "
                "Verify this matches the checkpoint's training config (student.patch_size)."
            )
        else:
            patch_size = native_patch_size_for_backbone(arch)
    log.info(f"Building with patch_size={patch_size}")
    backbone = build_backbone(arch, patch_size=patch_size)
    model = CurveModel(
        backbone=backbone,
        patch_size=patch_size,
        lora_cfg={"blocks": lora_blocks, "r": lora_r, "alpha": 16, "dropout": 0.0, "use_mlp": False},
    )
    model.eval()

    sd = remap_legacy_curve_head_keys(sd)
    result = model.load_state_dict(sd, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint/model key mismatch for backbone={backbone_name!r} "
            f"(lora_blocks={lora_blocks}): missing={result.missing_keys[:8]} "
            f"unexpected={result.unexpected_keys[:8]} — refusing to export a "
            "partially initialized model."
        )

    model = model.to(device=device, dtype=torch.float32)
    if hasattr(model.backbone, "rope_embed"):
        try:
            model.backbone.rope_embed.dtype = torch.float32
        except Exception:
            pass
    return model


def verify(eager, traced, x, tag="TS vs Eager"):
    with torch.inference_mode():
        a = eager(x)
        b = traced(x)

    def to_f(t):
        if isinstance(t, (tuple, list)):
            return [to_f(u) for u in t]
        return t.detach().cpu().float()

    def cmp(a, b, p=""):
        if isinstance(a, list):
            for i, (aa, bb) in enumerate(zip(a, b)):
                cmp(aa, bb, p + f"[{i}]")
        else:
            md = float((a - b).abs().mean())
            xd = float((a - b).abs().max())
            log.info(f"{tag}{p}: mean|diff|={md:.6g}  max|diff|={xd:.6g}")

    cmp(to_f(a), to_f(b))


def main():
    ap = argparse.ArgumentParser("Export DINOCT CurveModel (LoRA-fused) to TorchScript/ONNX")
    ap.add_argument("--backbone", choices=["auto", "small", "convnext_tiny", "convnext_small"], default="auto")
    ap.add_argument(
        "--model",
        default="outputs/post_train/fused_curve_best.pth",
        help="Path to fused_curve.pth (preferred single checkpoint)",
    )
    ap.add_argument("--outdir", default="exports")
    ap.add_argument(
        "--patch-size",
        type=int,
        default=0,
        help="Inference patch size the checkpoint was trained with (student.patch_size). "
        "0 = auto: ViT native; ConvNeXt assumes 14 (dinoct convention).",
    )
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--static", action="store_true", help="Export static batch=1 (no dynamic axes)")
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = pick_device(args.device)
    log.info(f"Using device: {device}")

    model = build_model(args.backbone, args.model, device, patch_size=args.patch_size)
    fused = fuse_all_lora_(model)
    log.info(f"Fused {fused} LoRA layers.")

    wrapped = ExportWrapper(model).to(device).eval()

    batch = 1 if args.static else 2
    example = torch.randn(batch, 3, ORIG_H, ORIG_W, device=device, dtype=torch.float32)

    ts_path = Path(args.outdir) / "curve_model.ts"
    log.info(f"Tracing TorchScript -> {ts_path}")
    with torch.inference_mode():
        traced = torch.jit.trace(wrapped, example, strict=False)
        torch.jit.save(traced, ts_path)

    if args.verify:
        verify(wrapped, traced, example)

    onnx_path = Path(args.outdir) / "curve_model.onnx"
    log.info(f"Exporting ONNX opset={args.opset} -> {onnx_path}")
    dyn = {"image": {0: "batch"}, "presence_logits": {0: "batch"}, "z_vec": {0: "batch"}}
    out_names = ["presence_logits", "z_vec"]

    with torch.inference_mode():
        try:
            torch.onnx.export(
                wrapped,
                example,
                onnx_path,
                export_params=True,
                opset_version=args.opset,
                do_constant_folding=True,
                input_names=["image"],
                output_names=out_names,
                dynamic_axes=dyn if not args.static else None,
                keep_initializers_as_inputs=False,
            )
        except ModuleNotFoundError as exc:
            if "onnxscript" not in str(exc):
                raise
            log.warning("onnxscript missing; falling back to legacy exporter.")
            from torch.onnx import OperatorExportTypes, TrainingMode
            from torch.onnx import utils as onnx_utils

            onnx_utils._export(
                wrapped,
                example,
                f=onnx_path,
                export_params=True,
                verbose=False,
                training=TrainingMode.EVAL,
                input_names=["image"],
                output_names=out_names,
                operator_export_type=OperatorExportTypes.ONNX,
                dynamic_axes=dyn if not args.static else None,
                keep_initializers_as_inputs=False,
            )

    data_path = onnx_path.with_suffix(onnx_path.suffix + ".data")
    if data_path.exists():
        try:
            import onnx

            model = onnx.load(onnx_path, load_external_data=True)
            try:
                onnx.save_model(model, onnx_path, save_as_external_data=False)
            except TypeError:
                onnx.save_model(model, onnx_path)
            data_path.unlink(missing_ok=True)
            log.info("Merged external data into single ONNX file.")
        except Exception as exc:  # pragma: no cover
            log.warning("Could not merge external ONNX data: %s", exc)

    log.info(f"Done.\n  TorchScript: {ts_path}\n  ONNX:       {onnx_path}")


if __name__ == "__main__":
    main()
