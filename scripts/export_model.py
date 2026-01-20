#!/usr/bin/env python3
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dinoct.models import build_backbone  # noqa: E402
from dinoct.train.post_train import CurveModel, LoRALinear, ORIG_H, ORIG_W  # noqa: E402

log = logging.getLogger("export")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # pragma: no cover
        return torch.device("mps")
    return torch.device("cpu")


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
        logits_pres, logits_curve = self.model(x)
        p_curve = torch.sigmoid(logits_pres)
        y_vec = soft_argmax_height_jit_safe(logits_curve)
        return p_curve, y_vec, logits_curve


def infer_backbone_from_curve_state_dict(state_dict: dict) -> str:
    keys = list(state_dict.keys())
    if any(k.startswith("backbone.stages.") or k.startswith("backbone.downsample_layers.") for k in keys):
        stage2_re = re.compile(r"^backbone\\.stages\\.2\\.(\\d+)\\.")
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


def build_model(backbone_name: str, model_path: str, device: torch.device) -> CurveModel:
    if not (model_path and os.path.exists(model_path)):
        raise FileNotFoundError(f"Fused model checkpoint not found: {model_path}")

    log.info(f"Loading fused curve model: {model_path}")
    sd = torch.load(model_path, map_location="cpu")
    sd = sd.get("model", sd)

    if backbone_name == "auto":
        backbone_name = infer_backbone_from_curve_state_dict(sd)
        log.info(f"Auto-detected backbone: {backbone_name}")

    arch = backbone_name.replace("vit_", "") if backbone_name.startswith("vit_") else backbone_name
    backbone = build_backbone(arch, patch_size=14)
    model = CurveModel(
        backbone=backbone, patch_size=14, lora_cfg={"blocks": 3, "r": 8, "alpha": 16, "dropout": 0.05, "use_mlp": False}
    )
    model.eval()

    try:
        model.load_state_dict(sd, strict=False)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load checkpoint into backbone={backbone_name!r}. "
            "Try `--backbone auto` or set the correct `--backbone` explicitly."
        ) from exc

    model = model.to(device=device, dtype=torch.float32)
    # Force rope to fp32 to avoid bfloat16 in ONNX
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
    ap = argparse.ArgumentParser("Export CurveModel (LoRA-fused) to TorchScript/ONNX")
    ap.add_argument("--backbone", choices=["auto", "small", "convnext_tiny", "convnext_small"], default="auto")
    ap.add_argument(
        "--model",
        default="outputs/post_train/fused_curve.pth",
        help="Path to fused_curve.pth (preferred single checkpoint)",
    )
    ap.add_argument("--outdir", default="exports")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--static", action="store_true", help="Export static batch=1 (no dynamic axes)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = pick_device()
    log.info(f"Using device: {device}")

    model = build_model(args.backbone, args.model, device)
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
    dyn = {"image": {0: "batch"}, "p_curve": {0: "batch"}, "y_vec": {0: "batch"}, "curve_logits": {0: "batch"}}
    out_names = ["p_curve", "y_vec", "curve_logits"]

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
            from torch.onnx import utils as onnx_utils
            from torch.onnx import OperatorExportTypes, TrainingMode

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

    # If exporter wrote external data, merge into single file for easier distribution
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
