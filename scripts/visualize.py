#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dinoct.data.transforms import Ensure3CH, MaybeToTensor, PerImageZScore  # noqa: E402
from dinoct.models import build_backbone  # noqa: E402
from dinoct.train.post_train import (  # noqa: E402
    CurveModel,
    ORIG_H,
    ORIG_W,
    soft_argmax_height,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _sanitize_filename_component(text: str) -> str:
    text = str(text).strip()
    if not text:
        return "image"
    out: list[str] = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out).strip("_.")
    return cleaned or "image"


@dataclass(frozen=True)
class OutputWriter:
    outdir: Path
    file_prefix: str
    sep: str = "_"

    def path(self, name: str) -> Path:
        name = Path(name).name  # ensure it's a basename
        prefix = self.file_prefix
        p = self.outdir / f"{prefix}{self.sep}{name}"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # pragma: no cover
        return torch.device("mps")
    return torch.device("cpu")


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _resolve_arch(arch_raw: str) -> str:
    arch = arch_raw.replace("vit_", "") if arch_raw.startswith("vit_") else arch_raw
    return arch


def _resolve_ffn_layer(name: str) -> str:
    if name.lower() in {"swiglufused", "swiglu_fused"}:
        return "swiglu"
    return name


def load_backbone_from_pretrain(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, int]:
    ckpt = _torch_load(ckpt_path)
    cfg: dict[str, Any] = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    student_cfg: dict[str, Any] = cfg.get("student", {}) if isinstance(cfg, dict) else {}

    arch = _resolve_arch(str(student_cfg.get("arch", "vit_small")))
    patch_size = int(student_cfg.get("patch_size", 14))

    backbone = build_backbone(
        arch,
        patch_size=patch_size,
        drop_path_rate=float(student_cfg.get("drop_path_rate", 0.0)),
        layerscale_init=student_cfg.get("layerscale", None),
        ffn_layer=_resolve_ffn_layer(str(student_cfg.get("ffn_layer", "mlp"))),
        qkv_bias=bool(student_cfg.get("qkv_bias", True)),
        proj_bias=bool(student_cfg.get("proj_bias", True)),
        ffn_bias=bool(student_cfg.get("ffn_bias", True)),
        n_storage_tokens=int(student_cfg.get("num_register_tokens", 0)),
        mask_k_bias=bool(student_cfg.get("mask_k_bias", False)),
        device=device,
    ).to(device)

    state = ckpt.get("student", ckpt) if isinstance(ckpt, dict) else ckpt
    if isinstance(state, dict):
        backbone.load_state_dict(state, strict=False)
    backbone.eval()
    return backbone, patch_size


def load_curve_model(
    *,
    ckpt_path: Path,
    state: dict[str, Any] | None,
    device: torch.device,
    backbone_name: str,
    patch_size: int,
    lora_blocks: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_use_mlp: bool,
) -> CurveModel:
    backbone = build_backbone(_resolve_arch(backbone_name), patch_size=patch_size, device=device)
    model = CurveModel(
        backbone=backbone,
        patch_size=patch_size,
        lora_cfg={
            "blocks": lora_blocks,
            "r": lora_r,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "use_mlp": lora_use_mlp,
        },
    ).to(device)
    if state is None:
        ckpt = _torch_load(ckpt_path)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    if isinstance(state, dict):
        model.load_state_dict(state, strict=False)
    model.eval()
    return model


def make_feature_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            MaybeToTensor(),
            Ensure3CH(),
            PerImageZScore(eps=1e-6),
        ]
    )


def make_curve_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((ORIG_H, ORIG_W), interpolation=InterpolationMode.BICUBIC),
            MaybeToTensor(),
            Ensure3CH(),
            PerImageZScore(eps=1e-6),
        ]
    )


def _normalize01(x: np.ndarray, *, clip_percentiles: tuple[float, float] | None = (1.0, 99.0)) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)

    if clip_percentiles is not None:
        lo, hi = clip_percentiles
        mn = float(np.percentile(x[finite], lo))
        mx = float(np.percentile(x[finite], hi))
    else:
        mn = float(x[finite].min())
        mx = float(x[finite].max())

    if not math.isfinite(mn) or not math.isfinite(mx) or mx <= mn:
        return np.zeros_like(x, dtype=np.float32)
    x = np.clip(x, mn, mx)
    return (x - mn) / (mx - mn)


_INFERNO_LUT_HEX = """\
00000300000400000601000701010901010b02010e020210
03021204031404031605041806041b07051d08061f090621
0a07230b07260d08280e082a0f092d10092f120a32130a34
140b36160b39170b3b190b3e1a0b401c0c431d0c451f0c47
200c4a220b4c240b4e260b50270b52290b542b0a562d0a58
2e0a5a300a5c32095d34095f3509603709613909623b0964
3c09653e0966400966410967430a68450a69460a69480b6a
4a0b6a4b0c6b4d0c6b4f0d6c500d6c520e6c530e6d550f6d
570f6d58106d5a116d5b116e5d126e5f126e60136e62146e
63146e65156e66156e68166e6a176e6b176e6d186e6e186e
70196e72196d731a6d751b6d761b6d781c6d7a1c6d7b1d6c
7d1d6c7e1e6c801f6b811f6b83206b85206a86216a88216a
8922698b22698d23698e2468902468912567932567952666
9626669827659928649b28649c29639e2963a02a62a12b61
a32b61a42c60a62c5fa72d5fa92e5eab2e5dac2f5cae305b
af315bb1315ab23259b43358b53357b73456b83556ba3655
bb3754bd3753be3852bf3951c13a50c23b4fc43c4ec53d4d
c73e4cc83e4bc93f4acb4049cc4148cd4247cf4446d04544
d14643d24742d44841d54940d64a3fd74b3ed94d3dda4e3b
db4f3adc5039dd5238de5337df5436e05634e25733e35832
e45a31e55b30e65c2ee65e2de75f2ce8612be9622aea6428
eb6527ec6726ed6825ed6a23ee6c22ef6d21f06f1ff0701e
f1721df2741cf2751af37719f37918f47a16f57c15f57e14
f68012f68111f78310f7850ef8870df8880cf88a0bf98c09
f98e08f99008fa9107fa9306fa9506fa9706fb9906fb9b06
fb9d06fb9e07fba007fba208fba40afba60bfba80dfbaa0e
fbac10fbae12fbb014fbb116fbb318fbb51afbb71cfbb91e
fabb21fabd23fabf25fac128f9c32af9c52cf9c72ff8c931
f8cb34f8cd37f7cf3af7d13cf6d33ff6d542f5d745f5d948
f4db4bf4dc4ff3de52f3e056f3e259f2e45df2e660f1e864
f1e968f1eb6cf1ed70f1ee74f1f079f1f27df2f381f2f485
f3f689f4f78df5f891f6fa95f7fb99f9fc9dfafda0fcfea4
"""
_INFERNO_LUT = np.frombuffer(bytes.fromhex(_INFERNO_LUT_HEX), dtype=np.uint8).reshape(256, 3)


def _apply_colormap(heat01: np.ndarray, cmap: str = "inferno") -> np.ndarray:
    heat01 = np.clip(heat01, 0.0, 1.0).astype(np.float32, copy=False)
    cmap_name = str(cmap).strip().lower()
    try:
        import matplotlib

        cm = matplotlib.colormaps.get_cmap(cmap_name)
        rgba = cm(heat01, bytes=True)
        return np.asarray(rgba[..., :3], dtype=np.uint8)
    except Exception:
        if cmap_name == "inferno":
            idx = (heat01 * 256.0).astype(np.int32)
            idx = np.clip(idx, 0, 255)
            return _INFERNO_LUT[idx]
        v = (heat01 * 255.0).round().clip(0, 255).astype(np.uint8)
        return np.stack([v, v, v], axis=-1)


def _blend(base_rgb: np.ndarray, overlay_rgb: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(alpha)
    alpha = 0.0 if alpha < 0.0 else (1.0 if alpha > 1.0 else alpha)
    if alpha == 0.0:
        return base_rgb
    if alpha == 1.0:
        return overlay_rgb
    base = base_rgb.astype(np.float32)
    over = overlay_rgb.astype(np.float32)
    out = base * (1.0 - alpha) + over * alpha
    return out.round().clip(0, 255).astype(np.uint8)


def _to_grayscale_rgb(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected (H,W,3) rgb, got {rgb.shape}")
    x = rgb.astype(np.float32)
    y = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    g = y.round().clip(0, 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def _overlay_heat(
    *,
    base_rgb: np.ndarray,
    heat_rgb: np.ndarray,
    heat01: np.ndarray,
    alpha: float,
    mode: str,
    base_dim: float,
    base_gray: bool,
    heat_gamma: float,
) -> np.ndarray:
    base = _to_grayscale_rgb(base_rgb) if base_gray else base_rgb
    base_dim = float(base_dim)
    base_dim = 0.0 if base_dim < 0.0 else (1.0 if base_dim > 1.0 else base_dim)
    if base_dim != 1.0:
        base = (base.astype(np.float32) * base_dim).round().clip(0, 255).astype(np.uint8)

    heat_gamma = float(heat_gamma)
    heat_gamma = 1.0 if not math.isfinite(heat_gamma) or heat_gamma <= 0 else heat_gamma
    h = np.clip(heat01, 0.0, 1.0) ** heat_gamma

    if mode == "constant":
        return _blend(base, heat_rgb, alpha)
    if mode != "heat":
        raise ValueError(f"Unknown overlay mode: {mode}")

    alpha = float(alpha)
    alpha = 0.0 if alpha < 0.0 else (1.0 if alpha > 1.0 else alpha)
    a = (alpha * h).astype(np.float32)[..., None]  # (H,W,1)
    out = base.astype(np.float32) * (1.0 - a) + heat_rgb.astype(np.float32) * a
    return out.round().clip(0, 255).astype(np.uint8)


def _attention_head_entropy(attn: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """
    attn: (heads, N_patches) non-negative, typically sums to ~1 per head.
    Returns: (heads,) entropy (lower => more concentrated).
    """
    a = attn.clamp_min(0)
    a = a / (a.sum(dim=-1, keepdim=True) + eps)
    return -(a * (a + eps).log()).sum(dim=-1)


def _upsample_patch_mask(
    mask_hw: np.ndarray,
    *,
    patch_size: int,
    out_hw: tuple[int, int],
) -> np.ndarray:
    if mask_hw.ndim != 2:
        raise ValueError(f"Expected (H,W) mask, got {mask_hw.shape}")
    H_tokens, W_tokens = mask_hw.shape
    up_h = H_tokens * patch_size
    up_w = W_tokens * patch_size
    t = torch.from_numpy(mask_hw.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(up_h, up_w), mode="nearest")
    img = t[0, 0].numpy()
    H_out, W_out = out_hw
    return (img[:H_out, :W_out] > 0.5).astype(np.uint8)


def _overlay_mask(
    base_rgb: np.ndarray, mask01: np.ndarray, *, color: tuple[int, int, int] = (0, 255, 255), alpha: float = 0.6
) -> np.ndarray:
    if mask01.ndim != 2:
        raise ValueError(f"Expected (H,W) mask, got {mask01.shape}")
    alpha = float(alpha)
    alpha = 0.0 if alpha < 0.0 else (1.0 if alpha > 1.0 else alpha)
    if alpha == 0.0:
        return base_rgb
    out = base_rgb.astype(np.float32).copy()
    m = mask01.astype(bool)
    c = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    out[m] = out[m] * (1.0 - alpha) + c * alpha
    return out.round().clip(0, 255).astype(np.uint8)


def _pca_rgb_from_patch_tokens(patch_tokens: torch.Tensor, *, H_tokens: int, W_tokens: int) -> np.ndarray:
    x = patch_tokens.detach().float().cpu()
    if x.ndim != 3 or x.shape[0] != 1:
        raise ValueError(f"Expected patch tokens with shape (1,N,C), got {tuple(x.shape)}")
    x = x.squeeze(0)  # (N,C)
    x = F.normalize(x, p=2, dim=-1)
    x = x - x.mean(dim=0, keepdim=True)

    # PCA via SVD: components are the first right-singular vectors.
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    comps = vh[:3].T  # (C,3)
    x_proj = x @ comps  # (N,3)

    # Stabilize signs for nicer visuals.
    max_abs_idx = torch.argmax(torch.abs(comps), dim=0)  # (3,)
    signs = torch.sign(comps[max_abs_idx, torch.arange(3)])
    x_proj = x_proj * signs

    rgb = x_proj.reshape(H_tokens, W_tokens, 3).numpy()
    rgb -= rgb.min(axis=(0, 1), keepdims=True)
    rgb /= np.ptp(rgb, axis=(0, 1), keepdims=True) + 1e-8
    return (rgb * 255.0).round().clip(0, 255).astype(np.uint8)


def _upsample_patch_rgb(
    rgb_hw3: np.ndarray,
    *,
    patch_size: int,
    out_hw: tuple[int, int],
) -> np.ndarray:
    if rgb_hw3.ndim != 3 or rgb_hw3.shape[-1] != 3:
        raise ValueError(f"Expected (H,W,3) rgb, got {rgb_hw3.shape}")
    H_tokens, W_tokens = rgb_hw3.shape[0], rgb_hw3.shape[1]
    up_h = H_tokens * patch_size
    up_w = W_tokens * patch_size

    t = torch.from_numpy(rgb_hw3).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t = F.interpolate(t, size=(up_h, up_w), mode="nearest")
    img = (t[0].permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    H_out, W_out = out_hw
    return img[:H_out, :W_out]


def _upsample_patch_heat(
    heat_hw: np.ndarray,
    *,
    patch_size: int,
    out_hw: tuple[int, int],
) -> np.ndarray:
    if heat_hw.ndim != 2:
        raise ValueError(f"Expected (H,W) heat, got {heat_hw.shape}")
    H_tokens, W_tokens = heat_hw.shape
    up_h = H_tokens * patch_size
    up_w = W_tokens * patch_size
    t = torch.from_numpy(heat_hw).unsqueeze(0).unsqueeze(0).float()
    t = F.interpolate(t, size=(up_h, up_w), mode="bilinear", align_corners=False)
    img = t[0, 0].numpy()
    H_out, W_out = out_hw
    return img[:H_out, :W_out]


def _draw_curve_rgb(
    base_rgb: np.ndarray, y_vec: np.ndarray, *, color: tuple[int, int, int] = (255, 0, 0), radius: int = 1
) -> np.ndarray:
    out = base_rgb.copy()
    H, W = out.shape[0], out.shape[1]
    if y_vec.shape != (W,):
        raise ValueError(f"Expected y_vec shape ({W},), got {y_vec.shape}")
    xs = np.arange(W, dtype=np.int64)
    ys = np.clip(np.round(y_vec).astype(np.int64), 0, H - 1)
    for off in range(-radius, radius + 1):
        yy = ys + off
        m = (yy >= 0) & (yy < H)
        out[yy[m], xs[m]] = np.array(color, dtype=np.uint8)
    return out


def infer_backbone_spec_from_state_dict(state: dict[str, Any]) -> tuple[str | None, int | None]:
    keys = list(state.keys())
    if any(k.startswith("backbone.stages.") or k.startswith("backbone.downsample_layers.") for k in keys):
        stage2_re = re.compile(r"^backbone\\.stages\\.2\\.(\\d+)\\.")
        stage2_idx: list[int] = []
        for k in keys:
            m = stage2_re.match(k)
            if m:
                stage2_idx.append(int(m.group(1)))
        if stage2_idx:
            depth2 = max(stage2_idx) + 1
            return ("convnext_small" if depth2 >= 27 else "convnext_tiny"), None
        return "convnext_tiny", None

    embed_to_name = {384: "small"}

    def _try_weight(value: Any) -> tuple[str | None, int | None]:
        if not isinstance(value, torch.Tensor) or value.ndim != 4:
            return None, None
        patch = int(value.shape[-1])
        embed_dim = int(value.shape[0])
        return embed_to_name.get(embed_dim), patch

    for key in ("backbone.patch_embed.proj.weight", "patch_embed.proj.weight"):
        name, patch = _try_weight(state.get(key))
        if patch is not None:
            return name, patch

    for key, value in state.items():
        if key.endswith("patch_embed.proj.weight"):
            name, patch = _try_weight(value)
            if patch is not None:
                return name, patch

    return None, None


def prepare_curve_model(args: argparse.Namespace, *, device: torch.device) -> tuple[CurveModel, transforms.Compose]:
    ckpt_path = Path(args.curve_ckpt) if args.curve_ckpt else None
    if ckpt_path is None:
        best = REPO_ROOT / "outputs" / "post_train" / "fused_curve_best.pth"
        final = REPO_ROOT / "outputs" / "post_train" / "fused_curve.pth"
        ckpt_path = best if best.exists() else final
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(
            f"Curve checkpoint not found: {ckpt_path or '<missing>'}. "
            "Pass `--curve-ckpt outputs/.../post_train/fused_curve_best.pth`."
        )

    ckpt = _torch_load(ckpt_path)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported curve checkpoint format: {type(state)}")

    inferred_name, inferred_patch = infer_backbone_spec_from_state_dict(state)
    backbone_name = inferred_name or (None if args.backbone == "auto" else args.backbone)
    if backbone_name is None:
        raise ValueError("Could not infer backbone from curve checkpoint; pass --backbone explicitly.")
    patch_size = int(inferred_patch or args.patch_size)

    model = load_curve_model(
        ckpt_path=ckpt_path,
        state=state,
        device=device,
        backbone_name=backbone_name,
        patch_size=patch_size,
        lora_blocks=args.lora_blocks,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_use_mlp=args.lora_use_mlp,
    )
    return model, make_curve_transform()


def run_curve_image(
    *,
    model: CurveModel,
    transform: transforms.Compose,
    image_path: Path,
    device: torch.device,
    writer: OutputWriter,
    colormap: str,
    overlay_alpha: float,
    overlay_mode: str,
    overlay_base_dim: float,
    overlay_base_gray: bool,
    heat_gamma: float,
    curve_uncertainty: bool,
    curve_uncertainty_images: bool,
) -> None:
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((ORIG_W, ORIG_H), resample=Image.BICUBIC)
    x = transform(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        presence_logits, curve_logits = model(x, orig_hw=(ORIG_H, ORIG_W))
        p_curve = float(torch.sigmoid(presence_logits).detach().cpu().float().reshape(-1)[0].item())
        y_vec = soft_argmax_height(curve_logits[:, :-1, :]).detach().cpu().float().numpy().reshape(-1)
        prob_hw = torch.softmax(curve_logits, dim=1)[0].detach().cpu().float().numpy()

    writer.path("curve_presence.txt").write_text(f"{p_curve:.6f}\n")
    np.savetxt(writer.path("curve_y.csv"), y_vec[None, :], delimiter=",", fmt="%.6f")

    base_rgb = np.array(img_resized, dtype=np.uint8)
    overlay = _draw_curve_rgb(base_rgb, y_vec, radius=2)
    Image.fromarray(overlay).save(writer.path("curve_overlay.jpg"))

    heat01 = _normalize01(np.log(prob_hw + 1e-9), clip_percentiles=(1.0, 99.0))
    Image.fromarray(_apply_colormap(heat01, cmap=colormap)).save(writer.path("curve_heatmap.jpg"))

    if curve_uncertainty:
        ent_w = -(prob_hw * np.log(prob_hw + 1e-9)).sum(axis=0)  # (W,)
        maxp_w = prob_hw.max(axis=0)  # (W,)
        np.savetxt(writer.path("curve_entropy.csv"), ent_w[None, :], delimiter=",", fmt="%.6f")
        np.savetxt(writer.path("curve_maxprob.csv"), maxp_w[None, :], delimiter=",", fmt="%.6f")

        mean_ent = float(np.mean(ent_w))
        mean_maxp = float(np.mean(maxp_w))
        writer.path("curve_uncertainty.txt").write_text(f"mean_entropy={mean_ent:.6f}\nmean_maxprob={mean_maxp:.6f}\n")

        if curve_uncertainty_images:
            ent_hw = np.broadcast_to(ent_w[None, :], (ORIG_H, ORIG_W))
            ent01 = _normalize01(ent_hw, clip_percentiles=(1.0, 99.0))
            ent_rgb = _apply_colormap(ent01, cmap=colormap)
            Image.fromarray(ent_rgb).save(writer.path("curve_entropy.jpg"))
            Image.fromarray(
                _overlay_heat(
                    base_rgb=base_rgb,
                    heat_rgb=ent_rgb,
                    heat01=ent01,
                    alpha=overlay_alpha,
                    mode=overlay_mode,
                    base_dim=overlay_base_dim,
                    base_gray=overlay_base_gray,
                    heat_gamma=heat_gamma,
                )
            ).save(writer.path("curve_entropy_overlay.jpg"))

            maxp_hw = np.broadcast_to(maxp_w[None, :], (ORIG_H, ORIG_W))
            maxp01 = _normalize01(maxp_hw, clip_percentiles=(1.0, 99.0))
            maxp_rgb = _apply_colormap(maxp01, cmap=colormap)
            Image.fromarray(maxp_rgb).save(writer.path("curve_maxprob.jpg"))
            Image.fromarray(
                _overlay_heat(
                    base_rgb=base_rgb,
                    heat_rgb=maxp_rgb,
                    heat01=maxp01,
                    alpha=overlay_alpha,
                    mode=overlay_mode,
                    base_dim=overlay_base_dim,
                    base_gray=overlay_base_gray,
                    heat_gamma=heat_gamma,
                )
            ).save(writer.path("curve_maxprob_overlay.jpg"))


def prepare_backbone(args: argparse.Namespace, *, device: torch.device) -> tuple[torch.nn.Module, int]:
    ckpt_path = Path(args.backbone_ckpt) if args.backbone_ckpt else None
    backbone: torch.nn.Module
    patch_size: int

    if ckpt_path is None:
        default = REPO_ROOT / "outputs" / "pretrain" / "dinov3_pretrain.pth"
        ckpt_path = default if default.exists() else None

    if ckpt_path is not None and ckpt_path.exists():
        backbone, patch_size = load_backbone_from_pretrain(ckpt_path, device=device)
    else:
        arch = "small" if args.backbone == "auto" else _resolve_arch(args.backbone)
        backbone = build_backbone(arch, patch_size=args.patch_size, device=device).to(device).eval()
        patch_size = int(args.patch_size)
    return backbone, patch_size


def run_features_image(
    *,
    backbone: torch.nn.Module,
    patch_size: int,
    image_path: Path,
    device: torch.device,
    writer: OutputWriter,
    image_size: int,
    colormap: str,
    overlay_alpha: float,
    overlay_mode: str,
    overlay_base_dim: float,
    overlay_base_gray: bool,
    heat_gamma: float,
    attn: str,
    seed: str,
    save_attn_heads: bool,
) -> None:
    img = Image.open(image_path).convert("RGB")
    x = make_feature_transform(image_size)(img).unsqueeze(0).to(device)
    H_in, W_in = int(x.shape[-2]), int(x.shape[-1])

    vis_pil = transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC)(img)
    vis_pil = transforms.CenterCrop(image_size)(vis_pil)
    vis_rgb = np.array(vis_pil, dtype=np.uint8)

    pad_h = (int(patch_size) - (H_in % int(patch_size))) % int(patch_size)
    pad_w = (int(patch_size) - (W_in % int(patch_size))) % int(patch_size)
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

    # Capture last-block attention weights (ViT-only).
    supports_attn = hasattr(backbone, "blocks") and hasattr(backbone, "patch_embed")
    attn_buf: list[torch.Tensor] = []
    hook = None

    def _capture_attn(
        module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], kwargs: dict[str, Any], _output: torch.Tensor
    ):
        x_in = inputs[0]  # (B,N,C) after norm1
        qkv = module.qkv(x_in)  # type: ignore[attr-defined]
        B, N, _ = qkv.shape
        C = module.qkv.in_features  # type: ignore[attr-defined]
        qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads)  # type: ignore[attr-defined]
        q, k, _v = qkv.permute(2, 0, 3, 1, 4)
        rope = kwargs.get("rope")
        if rope is not None:
            q, k = module.apply_rope(q, k, rope)  # type: ignore[attr-defined]
        attn = (q @ k.transpose(-2, -1)) * module.scale  # type: ignore[attr-defined]
        attn_buf.append(attn.softmax(dim=-1).detach().cpu())

    with torch.inference_mode():
        if supports_attn:
            last_attn = backbone.blocks[-1].attn  # type: ignore[attr-defined]
            hook = last_attn.register_forward_hook(_capture_attn, with_kwargs=True)
            # Run patch embed once to get token grid size (handles padding-to-multiple).
            patch_grid = backbone.patch_embed(x)  # type: ignore[attr-defined]
            H_tokens, W_tokens = int(patch_grid.shape[1]), int(patch_grid.shape[2])
        else:
            if attn != "none" or save_attn_heads:
                raise ValueError("Attention visualizations require a ViT backbone; use `--attn none` for ConvNeXt.")
            H_tokens, W_tokens = int(x.shape[-2] // patch_size), int(x.shape[-1] // patch_size)

        out = backbone.forward_features(x)[0]  # type: ignore[attr-defined]
    if hook is not None:
        hook.remove()

    cls = out["x_norm_clstoken"]
    patch_tokens = out["x_norm_patchtokens"]
    pca_rgb = _pca_rgb_from_patch_tokens(patch_tokens, H_tokens=H_tokens, W_tokens=W_tokens)
    pca_up = _upsample_patch_rgb(pca_rgb, patch_size=patch_size, out_hw=(H_in, W_in))
    Image.fromarray(pca_up).save(writer.path("pca_map.jpg"))

    patch_l2 = F.normalize(patch_tokens, p=2, dim=-1)
    cls_l2 = F.normalize(cls, p=2, dim=-1)

    sim = F.cosine_similarity(patch_tokens, cls.unsqueeze(1), dim=-1)  # (B,N)
    sim_hw = sim.reshape(1, H_tokens, W_tokens).squeeze(0).detach().cpu().numpy()
    sim_up = _upsample_patch_heat(sim_hw, patch_size=patch_size, out_hw=(H_in, W_in))
    sim01 = _normalize01(sim_up, clip_percentiles=(1.0, 99.0))
    sim_rgb = _apply_colormap(sim01, cmap=colormap)
    Image.fromarray(sim_rgb).save(writer.path("similarity_map.jpg"))
    Image.fromarray(
        _overlay_heat(
            base_rgb=vis_rgb,
            heat_rgb=sim_rgb,
            heat01=sim01,
            alpha=overlay_alpha,
            mode=overlay_mode,
            base_dim=overlay_base_dim,
            base_gray=overlay_base_gray,
            heat_gamma=heat_gamma,
        )
    ).save(writer.path("similarity_overlay.jpg"))

    # Feature "edge" map: cosine distance to nearest neighbors in patch grid.
    # Often highlights boundaries more clearly than CLS-based similarity.
    p_hw = patch_l2[0].reshape(H_tokens, W_tokens, -1)
    dist_r = 1.0 - (p_hw[:, :-1, :] * p_hw[:, 1:, :]).sum(dim=-1)  # (H,W-1)
    dist_d = 1.0 - (p_hw[:-1, :, :] * p_hw[1:, :, :]).sum(dim=-1)  # (H-1,W)
    edge_hw = torch.zeros((H_tokens, W_tokens), device=p_hw.device, dtype=dist_r.dtype)
    edge_hw[:, :-1] = torch.maximum(edge_hw[:, :-1], dist_r)
    edge_hw[:, 1:] = torch.maximum(edge_hw[:, 1:], dist_r)
    edge_hw[:-1, :] = torch.maximum(edge_hw[:-1, :], dist_d)
    edge_hw[1:, :] = torch.maximum(edge_hw[1:, :], dist_d)
    edge = edge_hw.detach().cpu().float().numpy()
    edge_up = _upsample_patch_heat(edge, patch_size=patch_size, out_hw=(H_in, W_in))
    edge01 = _normalize01(edge_up, clip_percentiles=(1.0, 99.0))
    edge_rgb = _apply_colormap(edge01, cmap=colormap)
    Image.fromarray(edge_rgb).save(writer.path("feature_edge_map.jpg"))
    Image.fromarray(
        _overlay_heat(
            base_rgb=vis_rgb,
            heat_rgb=edge_rgb,
            heat01=edge01,
            alpha=overlay_alpha,
            mode=overlay_mode,
            base_dim=overlay_base_dim,
            base_gray=overlay_base_gray,
            heat_gamma=heat_gamma,
        )
    ).save(writer.path("feature_edge_overlay.jpg"))

    # Seed-patch similarity (often more "segment-like" than CLS attention).
    seed_idx: int
    seed_desc: str
    seed_s = str(seed).strip().lower()
    if seed_s in {"auto", "cls"}:
        seed_sim = (patch_l2 * cls_l2.unsqueeze(1)).sum(dim=-1)  # (B,N)
        seed_idx = int(seed_sim[0].argmax().item())
        seed_desc = "auto=argmax(cos(patch,cls))"
    elif seed_s in {"center", "centre"}:
        seed_x = W_tokens // 2
        seed_y = H_tokens // 2
        seed_idx = int(seed_y * W_tokens + seed_x)
        seed_desc = "center"
    else:
        try:
            sx, sy = seed_s.split(",", 1)
            x_px = int(sx)
            y_px = int(sy)
        except Exception as exc:
            raise ValueError(f"Invalid --seed '{seed}'. Use 'auto', 'center', or 'x,y' pixels.") from exc
        seed_x = int(max(0, min(W_in - 1, x_px)) // patch_size)
        seed_y = int(max(0, min(H_in - 1, y_px)) // patch_size)
        seed_idx = int(seed_y * W_tokens + seed_x)
        seed_desc = f"pixel={x_px},{y_px} -> patch={seed_x},{seed_y}"

    seed_vec = patch_l2[:, seed_idx, :].unsqueeze(1)  # (B,1,C)
    seed_map = (patch_l2 * seed_vec).sum(dim=-1)  # (B,N)
    seed_hw = seed_map.reshape(1, H_tokens, W_tokens).squeeze(0).detach().cpu().numpy()
    seed_up = _upsample_patch_heat(seed_hw, patch_size=patch_size, out_hw=(H_in, W_in))
    seed01 = _normalize01(seed_up, clip_percentiles=(1.0, 99.0))
    seed_rgb = _apply_colormap(seed01, cmap=colormap)
    Image.fromarray(seed_rgb).save(writer.path("seed_similarity_map.jpg"))
    Image.fromarray(
        _overlay_heat(
            base_rgb=vis_rgb,
            heat_rgb=seed_rgb,
            heat01=seed01,
            alpha=overlay_alpha,
            mode=overlay_mode,
            base_dim=overlay_base_dim,
            base_gray=overlay_base_gray,
            heat_gamma=heat_gamma,
        )
    ).save(writer.path("seed_similarity_overlay.jpg"))
    writer.path("seed_patch.txt").write_text(f"seed_idx={seed_idx}\nseed={seed_desc}\n")

    if attn != "none" and attn_buf:
        attn_last = attn_buf[-1][0]  # (heads, N, N)
        n_extra = 1 + int(getattr(backbone, "n_storage_tokens", 0))
        cls2patch_heads = attn_last[:, 0, n_extra:]  # (heads, N_patches)

        # Choose which head/aggregation to visualize.
        head_idx = None
        if attn == "mean":
            cls2patch = cls2patch_heads.mean(dim=0)
        elif attn == "best":
            ent = _attention_head_entropy(cls2patch_heads)
            head_idx = int(ent.argmin().item())
            cls2patch = cls2patch_heads[head_idx]
        else:
            raise ValueError(f"Unknown --attn mode: {attn}")

        if head_idx is not None:
            writer.path("attention_head.txt").write_text(f"{head_idx}\n")
        ent = _attention_head_entropy(cls2patch_heads).detach().cpu().float().numpy()
        writer.path("attention_head_entropy.csv").write_text(",".join(f"{v:.6f}" for v in ent) + "\n")

        heat = cls2patch.reshape(H_tokens, W_tokens).detach().cpu().float().numpy()
        heat_up = _upsample_patch_heat(heat, patch_size=patch_size, out_hw=(H_in, W_in))
        heat01 = _normalize01(heat_up, clip_percentiles=(1.0, 99.0))
        heat_rgb = _apply_colormap(heat01, cmap=colormap)
        Image.fromarray(heat_rgb).save(writer.path("attention_map.jpg"))
        Image.fromarray(
            _overlay_heat(
                base_rgb=vis_rgb,
                heat_rgb=heat_rgb,
                heat01=heat01,
                alpha=overlay_alpha,
                mode=overlay_mode,
                base_dim=overlay_base_dim,
                base_gray=overlay_base_gray,
                heat_gamma=heat_gamma,
            )
        ).save(writer.path("attention_overlay.jpg"))

        if save_attn_heads:
            heads = int(cls2patch_heads.shape[0])
            for h in range(heads):
                hh = cls2patch_heads[h].reshape(H_tokens, W_tokens).detach().cpu().float().numpy()
                hh_up = _upsample_patch_heat(hh, patch_size=patch_size, out_hw=(H_in, W_in))
                hh01 = _normalize01(hh_up, clip_percentiles=(1.0, 99.0))
                hh_rgb = _apply_colormap(hh01, cmap=colormap)
                Image.fromarray(hh_rgb).save(writer.path(f"attention_head_{h}.jpg"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("dinoct visualizations (curve + backbone features)")
    parser.add_argument("--mode", choices=["curve", "features", "all"], default="features")
    parser.add_argument(
        "--input", "--image", dest="input_path", required=True, help="Path to an image file or a directory"
    )
    parser.add_argument(
        "--recursive", action="store_true", help="If --input is a directory, recurse into subdirectories"
    )
    parser.add_argument(
        "--outdir", default=str(REPO_ROOT / "outputs" / "vis"), help="Directory to write output images/files"
    )
    parser.add_argument(
        "--colormap",
        default="inferno",
        help="Matplotlib colormap name if matplotlib is installed (fallback is grayscale)",
    )
    parser.add_argument("--overlay-alpha", type=float, default=0.8, help="Overlay strength (0..1)")
    parser.add_argument(
        "--overlay-mode",
        choices=["constant", "heat"],
        default="heat",
        help="constant=constant alpha, heat=alpha scaled by heat intensity",
    )
    parser.add_argument("--overlay-base-dim", type=float, default=0.45, help="Dim the base image before overlay (0..1)")
    parser.add_argument("--overlay-base-gray", action="store_true", help="Convert base image to grayscale for overlays")
    parser.add_argument(
        "--heat-gamma", type=float, default=1.0, help="Gamma on heat (higher => more contrast on hotspots)"
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="Stop on the first failing image instead of continuing"
    )

    # Backbone / features
    parser.add_argument(
        "--backbone-ckpt", default=None, help="Path to `pretrain/dinov3_pretrain.pth` (loads `student` backbone)"
    )
    parser.add_argument("--image-size", type=int, default=224, help="Feature visualization crop size")
    parser.add_argument(
        "--attn",
        choices=["none", "mean", "best"],
        default="none",
        help="Attention map: mean=head average, best=lowest-entropy head",
    )
    parser.add_argument(
        "--seed",
        default="auto",
        help="Seed patch for seed_similarity_*: 'auto'|'center'|'x,y' in pixels of the eval crop",
    )
    parser.add_argument(
        "--save-attn-heads",
        action="store_true",
        help="Also write attention_head_{i}.jpg for every head (can be slow for many images)",
    )

    # Curve model
    parser.add_argument("--curve-ckpt", default=str(REPO_ROOT / "outputs" / "post_train" / "fused_curve_best.pth"))
    parser.add_argument(
        "--curve-uncertainty", action="store_true", help="Write curve uncertainty CSV + summary txt (entropy/maxprob)"
    )
    parser.add_argument(
        "--curve-uncertainty-images",
        action="store_true",
        help="Also write uncertainty images/overlays (often redundant with curve_heatmap.jpg)",
    )
    parser.add_argument("--backbone", choices=["auto", "small", "convnext_tiny", "convnext_small"], default="auto")
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--lora-blocks", type=int, default=3)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-use-mlp", action="store_true")

    return parser.parse_args()


def collect_images(input_path: Path, *, recursive: bool) -> tuple[list[Path], Path | None]:
    if input_path.is_file():
        return [input_path], None
    if input_path.is_dir():
        it = input_path.rglob("*") if recursive else input_path.glob("*")
        images = [p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        images.sort()
        return images, input_path
    raise FileNotFoundError(input_path)


def main() -> None:
    args = parse_args()
    device = pick_device()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    images, root = collect_images(Path(args.input_path), recursive=bool(args.recursive))
    if len(images) == 0:
        raise FileNotFoundError(f"No images found under: {args.input_path}")

    curve_model: CurveModel | None = None
    curve_transform: transforms.Compose | None = None
    if args.mode in {"curve", "all"}:
        curve_model, curve_transform = prepare_curve_model(args, device=device)

    backbone: torch.nn.Module | None = None
    feature_patch_size: int | None = None
    if args.mode in {"features", "all"}:
        backbone, feature_patch_size = prepare_backbone(args, device=device)

    failures: list[str] = []
    for idx, image_path in enumerate(images, start=1):
        if root is not None:
            rel = image_path.relative_to(root)
            file_prefix_raw = rel.with_suffix("").as_posix().replace("/", "__")
        else:
            file_prefix_raw = image_path.stem

        writer = OutputWriter(
            outdir=outdir,
            file_prefix=_sanitize_filename_component(file_prefix_raw),
        )

        print(f"[{idx}/{len(images)}] {image_path}")
        try:
            if curve_model is not None and curve_transform is not None:
                run_curve_image(
                    model=curve_model,
                    transform=curve_transform,
                    image_path=image_path,
                    device=device,
                    writer=writer,
                    colormap=str(args.colormap),
                    overlay_alpha=float(args.overlay_alpha),
                    overlay_mode=str(args.overlay_mode),
                    overlay_base_dim=float(args.overlay_base_dim),
                    overlay_base_gray=bool(args.overlay_base_gray),
                    heat_gamma=float(args.heat_gamma),
                    curve_uncertainty=bool(args.curve_uncertainty),
                    curve_uncertainty_images=bool(args.curve_uncertainty_images),
                )
            if backbone is not None and feature_patch_size is not None:
                run_features_image(
                    backbone=backbone,
                    patch_size=feature_patch_size,
                    image_path=image_path,
                    device=device,
                    writer=writer,
                    image_size=int(args.image_size),
                    colormap=str(args.colormap),
                    overlay_alpha=float(args.overlay_alpha),
                    overlay_mode=str(args.overlay_mode),
                    overlay_base_dim=float(args.overlay_base_dim),
                    overlay_base_gray=bool(args.overlay_base_gray),
                    heat_gamma=float(args.heat_gamma),
                    attn=str(args.attn),
                    seed=str(args.seed),
                    save_attn_heads=bool(args.save_attn_heads),
                )
        except Exception as exc:
            msg = f"{image_path}: {type(exc).__name__}: {exc}"
            failures.append(msg)
            print(f"[{idx}/{len(images)}] ERROR: {msg}", file=sys.stderr)
            if args.fail_fast:
                raise

    if failures:
        (outdir / "errors.txt").write_text("\n".join(failures) + "\n")
        print(f"Wrote {len(failures)} failures to {outdir / 'errors.txt'}", file=sys.stderr)


if __name__ == "__main__":
    main()
