from __future__ import annotations

import csv
import copy
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from ..data import make_dataset
from ..data.datasets import OCT
from ..data.transforms import Ensure3CH, MaybeToTensor, PerImageZScore
from ..eval import DEFAULT_ACC_TOLERANCES, curve_metrics_batch, metric_name_for_tolerance
from ..models.convnext import LayerNorm as ConvNeXtLayerNorm
from ..provenance import file_md5, git_state, runtime_versions
from ..utils import fix_random_seeds, seed_worker

ORIG_H, ORIG_W = 512, 500
logger = logging.getLogger("dinoct")


def pad_to_multiple_hw_center(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    _, _, H, W = x.shape
    pad_h = (multiple - (H % multiple)) % multiple
    pad_w = (multiple - (W % multiple)) % multiple

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if pad_h or pad_w:
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)
    return x, (pad_top, pad_bottom, pad_left, pad_right)


def soft_argmax_height(logits_hw: torch.Tensor) -> torch.Tensor:
    """Column-wise softmax over H then soft-argmax → (B, W). Differentiable."""
    _, H, _ = logits_hw.shape
    p = F.softmax(logits_hw, dim=1)
    grid = torch.arange(H, device=logits_hw.device, dtype=logits_hw.dtype).view(1, H, 1)
    return (p * grid).sum(dim=1)


def gaussian_targets_from_z(z: torch.Tensor, H: int, sigma: float = 1.5) -> torch.Tensor:
    """z: (B, W) pixel coords -> (B, H, W) Gaussian targets per column."""
    B, W = z.shape
    grid = torch.arange(H, device=z.device, dtype=z.dtype).view(1, H, 1)
    g = torch.exp(-0.5 * ((grid - z.unsqueeze(1)) / sigma) ** 2)
    g = g / (g.sum(dim=1, keepdim=True) + 1e-8)
    return g


def remap_legacy_curve_head_keys(state_dict: dict) -> dict:
    """Checkpoints saved before the axial coordinate was renamed y->z store the curve
    head's output conv under `out_y`; the attribute is now `out_z`. Remap so old
    checkpoints keep loading."""
    return {k.replace("out_y.", "out_z."): v for k, v in state_dict.items()}


def column_ce_loss(logits_hw: torch.Tensor, targets_hw: torch.Tensor, non_bg_mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over columns, averaged over non-bg samples."""
    logp = F.log_softmax(logits_hw, dim=1)
    ce_per_sample = -(targets_hw * logp).sum(dim=1).mean(dim=1)
    m = non_bg_mask.float()
    if m.sum() == 0:
        return logits_hw.new_zeros(())
    return (ce_per_sample * m).sum() / (m.sum() + 1e-8)


def curvature_loss_from_logits(logits_hw: torch.Tensor, non_bg_mask: torch.Tensor) -> torch.Tensor:
    """|y_{x+1} - 2*y_x + y_{x-1}| averaged."""
    z_hat = soft_argmax_height(logits_hw)
    d2 = z_hat[:, 2:] - 2 * z_hat[:, 1:-1] + z_hat[:, :-2]
    curv_per_sample = d2.abs().mean(dim=1)
    m = non_bg_mask.float()
    if m.sum() == 0:
        return logits_hw.new_zeros(())
    return (curv_per_sample * m).sum() / (m.sum() + 1e-8)


def confidence_weight(curve_logits_z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return (B,W) normalized confidence weights in [0,1]."""
    p = F.softmax(curve_logits_z.float(), dim=1).clamp_min(float(eps))
    entropy = -(p * p.log()).sum(dim=1)  # (B,W)
    entropy = entropy / max(math.log(p.shape[1]), 1e-8)
    return (1.0 - entropy).clamp(0.0, 1.0)


def robust_curv_loss(curve_logits_z: torch.Tensor, non_bg_mask: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Confidence-weighted (via entropy) robust smoothness over W using a Huber penalty on d2."""
    z_hat = soft_argmax_height(curve_logits_z.float())  # (B,W)
    if z_hat.shape[1] < 3:
        return curve_logits_z.new_zeros(())
    d2 = z_hat[:, 2:] - 2 * z_hat[:, 1:-1] + z_hat[:, :-2]  # (B,W-2)

    confidence = confidence_weight(curve_logits_z)[:, 1:-1]  # align to W-2

    absd = d2.abs()
    delta_f = max(float(delta), 1e-6)
    huber = torch.where(absd < delta_f, 0.5 * (d2**2) / delta_f, absd - 0.5 * delta_f)
    smooth_loss = confidence * huber

    m = non_bg_mask.float()
    if m.sum() == 0:
        return curve_logits_z.new_zeros(())
    smooth_loss = smooth_loss * m.unsqueeze(1)
    return smooth_loss.sum() / (smooth_loss.shape[1] * m.sum() + 1e-8)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = float(decay)
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for ema_p, p in zip(self.ema.parameters(), model.parameters()):
            ema_p.mul_(d).add_(p.detach(), alpha=1.0 - d)
        for ema_b, b in zip(self.ema.buffers(), model.buffers()):
            ema_b.copy_(b)




class LoRALinear(nn.Module):
    """W x + (alpha/r) * B(Ax). Base weight frozen; only A,B train."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.r > 0:
            out = out + F.linear(F.linear(self.drop(x), self.lora_A), self.lora_B) * self.scaling
        return out


def _flat_vit_blocks(vit: nn.Module) -> list[nn.Module]:
    if hasattr(vit, "_iter_blocks"):
        return list(vit._iter_blocks())
    if not hasattr(vit, "blocks"):
        raise TypeError("Expected ViT-like module with `.blocks`")
    blocks: list[nn.Module] = []
    for item in vit.blocks:
        if isinstance(item, nn.ModuleList):
            blocks.extend(list(item))
        else:
            blocks.append(item)
    return blocks


def apply_lora_to_vit(
    vit: nn.Module,
    *,
    num_blocks: int,
    r: int,
    alpha: int,
    dropout: float,
    use_mlp: bool,
) -> None:
    """Patch the last `num_blocks` with LoRA on qkv/proj (+mlp if requested)."""
    blocks = _flat_vit_blocks(vit)
    if not blocks:
        raise ValueError("ViT model has no blocks under `.blocks`")
    blocks = blocks[-num_blocks:] if num_blocks > 0 else []
    for blk in blocks:
        blk.attn.qkv = LoRALinear(blk.attn.qkv, r=r, alpha=alpha, dropout=dropout)
        blk.attn.proj = LoRALinear(blk.attn.proj, r=r, alpha=alpha, dropout=dropout)
        if use_mlp:
            blk.mlp.fc1 = LoRALinear(blk.mlp.fc1, r=r, alpha=alpha, dropout=dropout)
            blk.mlp.fc2 = LoRALinear(blk.mlp.fc2, r=r, alpha=alpha, dropout=dropout)


def apply_lora_to_convnext(
    convnext: nn.Module,
    *,
    num_blocks: int,
    r: int,
    alpha: int,
    dropout: float,
) -> None:
    """Patch the last `num_blocks` ConvNeXt blocks with LoRA on pwconv1/pwconv2."""
    if not hasattr(convnext, "stages"):
        raise TypeError("Expected ConvNeXt-like module with `.stages`")
    blocks: list[nn.Module] = []
    for stage in list(getattr(convnext, "stages")):
        blocks.extend(list(stage))
    if not blocks:
        raise ValueError("ConvNeXt model has no blocks under `.stages`")
    blocks = blocks[-num_blocks:] if num_blocks > 0 else []
    for blk in blocks:
        if hasattr(blk, "pwconv1") and isinstance(getattr(blk, "pwconv1"), nn.Linear):
            blk.pwconv1 = LoRALinear(blk.pwconv1, r=r, alpha=alpha, dropout=dropout)
        if hasattr(blk, "pwconv2") and isinstance(getattr(blk, "pwconv2"), nn.Linear):
            blk.pwconv2 = LoRALinear(blk.pwconv2, r=r, alpha=alpha, dropout=dropout)


def apply_lora_to_backbone(
    backbone: nn.Module,
    *,
    num_blocks: int,
    r: int,
    alpha: int,
    dropout: float,
    use_mlp: bool,
) -> None:
    """Apply LoRA to the last blocks of a supported backbone (ViT or ConvNeXt)."""
    if hasattr(backbone, "blocks"):
        apply_lora_to_vit(backbone, num_blocks=num_blocks, r=r, alpha=alpha, dropout=dropout, use_mlp=use_mlp)
        return
    if hasattr(backbone, "stages"):
        apply_lora_to_convnext(backbone, num_blocks=num_blocks, r=r, alpha=alpha, dropout=dropout)
        return
    raise TypeError(
        "Unsupported backbone type for LoRA injection (expected ViT-like `.blocks` or ConvNeXt-like `.stages`)."
    )


class CurveHead(nn.Module):
    """Light conv decoder producing per-column (H+1)-class logits:
    classes 0..H-1 = curve at row z, class H = no-curve.
    """

    def __init__(self, in_channels: int, mid: int = 128):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, mid, kernel_size=1, bias=True)
        self.vert1 = nn.Conv2d(mid, mid, kernel_size=(5, 1), padding=(2, 0), groups=mid)
        self.act1 = nn.GELU()
        self.vert2 = nn.Conv2d(mid, mid, kernel_size=(5, 1), padding=(2, 0), groups=mid)
        self.act2 = nn.GELU()
        self.h_norm = nn.GroupNorm(1, mid)  # stable for small batch
        self.h_act = nn.GELU()
        self.horiz1 = nn.Conv2d(
            mid,
            mid,
            kernel_size=(1, 9),
            padding=(0, 4),
            groups=mid,
            padding_mode="replicate",
            bias=False,
        )
        nn.init.zeros_(self.horiz1.weight)  # start as no-op
        self.h_gamma = nn.Parameter(1e-3 * torch.ones(mid, 1, 1))  # per-channel layer scale
        self.h_drop = nn.Dropout(p=0.1)  # optional; tune
        self.out_z = nn.Conv2d(mid, 1, kernel_size=1, bias=True)
        self.out_none = nn.Conv2d(mid, 1, kernel_size=1, bias=True)

    def forward(self, tokens_hw: torch.Tensor, out_size_hw: tuple[int, int]) -> torch.Tensor:
        x = self.proj(tokens_hw)
        x = self.act1(self.vert1(x))
        x = self.act2(self.vert2(x))
        h = self.horiz1(self.h_act(self.h_norm(x)))
        gamma = self.h_gamma.to(dtype=h.dtype)
        x = x + self.h_drop(gamma * h)

        H_out, W_out = out_size_hw

        # z logits map -> upsample to (H_out, W_out)
        z_logits = self.out_z(x)
        z_logits = F.interpolate(z_logits, size=(H_out, W_out), mode="bilinear", align_corners=False)
        z_logits = z_logits.squeeze(1)

        # no-curve logits per column
        col_feat = x.mean(dim=2, keepdim=True)
        none_logits = self.out_none(col_feat)
        none_logits = F.interpolate(none_logits, size=(1, W_out), mode="bilinear", align_corners=False)
        none_logits = none_logits.squeeze(1).squeeze(1)

        return torch.cat([z_logits, none_logits.unsqueeze(1)], dim=1)


class InputAntiAlias(nn.Module):
    def __init__(self, strength: float = 0.0):
        super().__init__()
        kernel_1d = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=torch.float32)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d).view(1, 1, 5, 5)
        self.register_buffer("kernel_2d", kernel_2d)
        self.register_buffer("aa_strength", torch.tensor(float(strength), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        strength = self.aa_strength.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)
        kernel = self.kernel_2d.to(device=x.device, dtype=x.dtype).expand(x.shape[1], 1, 5, 5)
        blurred = F.conv2d(x, kernel, padding=2, groups=x.shape[1]).to(dtype=x.dtype)
        return torch.lerp(x, blurred, strength)


@dataclass
class LossCfg:
    sigma: float = 1.5
    lambda_curve: float = 1.0
    lambda_curv: float = 0.05
    bg_weight: float = 5.0
    eps_none: float = 0.02
    curv_delta: float = 1.0


@dataclass(frozen=True)
class PostTrainAugmentCfg:
    p: float = 0.0
    types: tuple[str, ...] = ("stripe", "ghost", "dropout", "combined")
    severity: str = "medium"


def _normalize_aug_types(types: Sequence[str] | str | None) -> tuple[str, ...]:
    if types is None:
        return ("stripe", "ghost", "dropout", "combined")
    if isinstance(types, str):
        raw = [part.strip().lower() for part in types.split(",")]
    else:
        raw = [str(part).strip().lower() for part in types]
    valid = {"stripe", "ghost", "dropout", "combined", "photometric"}
    out = tuple(part for part in raw if part in valid)
    return out or ("stripe", "ghost", "dropout", "combined")


def _raised_cosine(length: int, center: float, half_width: float) -> torch.Tensor:
    coords = torch.arange(length, dtype=torch.float32)
    dist = (coords - float(center)).abs() / max(float(half_width), 1.0)
    window = torch.zeros((length,), dtype=torch.float32)
    inside = dist < 1.0
    window[inside] = 0.5 * (1.0 + torch.cos(math.pi * dist[inside]))
    return window


def _stripe_aug(image: torch.Tensor, rng: np.random.Generator, severity: str) -> torch.Tensor:
    out = image.clone()
    _, height, _ = out.shape
    level = {"mild": 0.75, "medium": 1.0, "severe": 1.35}.get(severity, 1.0)
    count = max(1, int(round(level)))
    thickness = max(2, int(round(6 * level)))
    opacity = min(0.88, 0.55 + 0.18 * level)
    target = out.mean() + (1.8 + 0.4 * level) * out.std().clamp_min(1e-4)
    top_band = min(max(int(200 * level), 32), max(height - 2, 1))
    for _ in range(count):
        center = int(rng.integers(4, top_band + 1))
        y0 = max(0, center - thickness)
        y1 = min(height, center + thickness + 1)
        out[:, y0:y1, :] = out[:, y0:y1, :] * (1.0 - opacity) + target * opacity
    return out


def _ghost_aug(image: torch.Tensor, severity: str) -> torch.Tensor:
    out = image.clone()
    _, height, _ = out.shape
    level = {"mild": 0.75, "medium": 1.0, "severe": 1.35}.get(severity, 1.0)
    shift = min(max(int(round(24 * level)), 1), max(height - 1, 1))
    opacity = min(0.42, 0.18 + 0.08 * level)
    ghost = torch.zeros_like(out)
    ghost[:, shift:, :] = out[:, :-shift, :]
    return out + opacity * ghost


def _dropout_aug(image: torch.Tensor, rng: np.random.Generator, severity: str) -> torch.Tensor:
    out = image.clone()
    channels, height, width = out.shape
    level = {"mild": 0.75, "medium": 1.0, "severe": 1.35}.get(severity, 1.0)
    regions = max(1, int(round(1.5 * level)))
    contrast = max(0.30, 0.62 - 0.16 * level)
    scale = max(0.45, 0.78 - 0.12 * level)
    base_mean = out.mean(dim=(1, 2), keepdim=True)
    low_contrast = (base_mean + contrast * (out - base_mean)) * scale
    for _ in range(regions):
        half_w = max(12, int(round(float(rng.uniform(48, 96)) * level)))
        half_h = max(20, int(round(float(rng.uniform(72, 140)) * level)))
        cx = int(rng.integers(half_w, max(width - half_w, half_w) + 1))
        cy_low = max(half_h, 48)
        cy_high = min(max(height - half_h - 1, cy_low), 320)
        cy = int(rng.integers(cy_low, cy_high + 1))
        xw = _raised_cosine(width, cx, half_w)
        yw = _raised_cosine(height, cy, half_h)
        mask = torch.outer(yw, xw).to(dtype=out.dtype, device=out.device).expand(channels, -1, -1)
        out = out * (1.0 - mask) + low_contrast * mask
    return out


def _photometric_aug(image: torch.Tensor, rng: np.random.Generator, severity: str) -> torch.Tensor:
    """Intensity-only photometric perturbation on a per-image z-scored tensor.

    A composition of brightness/contrast jitter + gaussian blur + gaussian noise, each gated by
    its own inner probability so a single draw yields a real perturbation. Same operator family as
    the SSL backbone ``aggressive_aug`` (ColorJitter b=0.6/c=0.6, GaussianBlur, GaussianNoise
    std~0.1), scaled for z-scored space (image std ~= 1). All ops are symmetric in intensity and
    leave surface geometry unchanged, so the curve label stays valid; unlike stripe/ghost/dropout
    they do not overlap the eval corruption operators.
    """
    out = image.clone()
    _, height, width = out.shape
    level = {"mild": 0.75, "medium": 1.0, "severe": 1.35}.get(severity, 1.0)
    applied = False

    # brightness: additive DC shift in std units (scalar keeps the 3 channels identical)
    if rng.random() < 0.8:
        out = out + float(rng.uniform(-0.5, 0.5)) * level
        applied = True

    # contrast: blend toward the per-image mean (factor in [1-c, 1+c])
    if rng.random() < 0.8:
        contrast = 1.0 + float(rng.uniform(-0.5, 0.5)) * level
        base_mean = out.mean()
        out = base_mean + contrast * (out - base_mean)
        applied = True

    # gaussian blur: symmetric kernel -> preserves the surface location (label stays valid)
    if rng.random() < 0.5:
        radius_max = max(0.5, 3.0 * level)
        out = transforms.GaussianBlur(kernel_size=9, sigma=(0.1, radius_max))(out)
        applied = True

    # gaussian noise: additive in std units; force at least one op so a draw is never a no-op
    if rng.random() < 0.5 or not applied:
        noise_std = 0.1 * level
        noise = torch.from_numpy((rng.standard_normal(size=(1, height, width)) * noise_std).astype(np.float32))
        out = out + noise.to(dtype=out.dtype, device=out.device)
    return out


def _apply_post_train_aug(image: torch.Tensor, cfg: PostTrainAugmentCfg | None) -> torch.Tensor:
    if cfg is None or float(cfg.p) <= 0.0 or np.random.random() >= float(cfg.p):
        return image
    rng = np.random.default_rng(np.random.randint(0, 2**32 - 1, dtype=np.uint32).item())
    aug_type = str(rng.choice(cfg.types))
    severity = str(cfg.severity).strip().lower()
    if aug_type == "stripe":
        return _stripe_aug(image, rng, severity)
    if aug_type == "ghost":
        return _ghost_aug(image, severity)
    if aug_type == "dropout":
        return _dropout_aug(image, rng, severity)
    if aug_type == "combined":
        return _ghost_aug(_stripe_aug(image, rng, severity), severity)
    if aug_type == "photometric":
        return _photometric_aug(image, rng, severity)
    return image


def column_ce_loss_h1w(
    logits_h1w: torch.Tensor,
    targets_h1w: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy over columns for (H+1)-class logits, optionally sample-weighted."""
    logp = F.log_softmax(logits_h1w, dim=1)  # (B,H+1,W)
    ce_per_col = -(targets_h1w * logp).sum(dim=1)  # (B,W)
    ce_per_sample = ce_per_col.mean(dim=1)  # (B,)
    if sample_weight is None:
        return ce_per_sample.mean()
    w = sample_weight.float()
    return (ce_per_sample * w).sum() / (w.sum() + 1e-8)


class CurveLoss(nn.Module):
    def __init__(self, cfg: LossCfg):
        super().__init__()
        self.cfg = cfg

    def forward(
        self, curve_logits: torch.Tensor, z_curve: torch.Tensor, is_bg: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self.cfg
        B, H1, W = curve_logits.shape
        H = H1 - 1
        non_bg = (1 - is_bg).float()

        with torch.no_grad():
            g = gaussian_targets_from_z(z_curve, H=H, sigma=cfg.sigma)
            eps_none = float(cfg.eps_none)
            targets_none = (is_bg.float().view(B, 1, 1) * 1.0) + ((1 - is_bg).float().view(B, 1, 1) * eps_none)
            targets_none = targets_none.expand(B, 1, W)
            targets_z = g * (1.0 - targets_none)  # ensures sum across (H+1) is 1
            targets = torch.cat([targets_z, targets_none], dim=1)

        w = torch.ones((B,), device=curve_logits.device, dtype=curve_logits.dtype)
        w = torch.where(is_bg.bool(), w * float(cfg.bg_weight), w)

        loss_curve = column_ce_loss_h1w(curve_logits, targets, sample_weight=w)
        loss_curv = robust_curv_loss(curve_logits[:, :H, :], non_bg_mask=non_bg, delta=float(cfg.curv_delta))
        total = cfg.lambda_curve * loss_curve + cfg.lambda_curv * loss_curv
        return total, {
            "loss_col_ce": loss_curve.detach(),
            "loss_smooth": loss_curv.detach(),
        }


NORM_MODES = ("all_norms", "final_only", "none")


def freeze_backbone_except_lora_and_norms(
    backbone: nn.Module,
    unfreeze_dwconv: bool = False,
    norm_mode: str = "all_norms",
):
    """Freeze backbone weights except LoRA adapters and selected normalization layers.

    ``norm_mode`` controls which backbone norm layers remain trainable:
      - ``"all_norms"`` (default): every LayerNorm/GroupNorm/BatchNorm2d/ConvNeXtLayerNorm
        is trainable (the historical behavior).
      - ``"final_only"``: only the final backbone norm (top-level ``backbone.norm``,
        i.e. ``getattr(backbone, "norm", None)``) is trainable, which for the main
        ConvNeXt/ViT backbones is the final ``self.norm`` LayerNorm applied before the head.
      - ``"none"``: no norm layers are trainable (LoRA-only + curve head).

    When ``unfreeze_dwconv`` is True, the ConvNeXt depthwise (7x7 spatial) convs are
    also made trainable. They are selected robustly as grouped convs whose groups equal
    their input channels (>1), which matches ConvNeXt block ``dwconv`` layers and excludes
    the stem and inter-stage downsample convs (both ``groups==1``). This is orthogonal to
    ``norm_mode``.
    """
    norm_mode_l = str(norm_mode).strip().lower()
    if norm_mode_l not in NORM_MODES:
        raise ValueError(f"Unknown norm_mode {norm_mode!r}; expected one of {NORM_MODES}.")

    for p in backbone.parameters():
        p.requires_grad = False
    for m in backbone.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad = True
            m.lora_B.requires_grad = True

    trainable_norm_tensors = 0
    if norm_mode_l == "all_norms":
        for m in backbone.modules():
            if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d, ConvNeXtLayerNorm)):
                for p in m.parameters():
                    p.requires_grad = True
                    trainable_norm_tensors += 1
    elif norm_mode_l == "final_only":
        final_norm = getattr(backbone, "norm", None)
        if final_norm is None:
            raise ValueError("Backbone has no `.norm` module; cannot use norm_mode='final_only'.")
        for p in final_norm.parameters():
            p.requires_grad = True
            trainable_norm_tensors += 1
    # norm_mode_l == "none": leave every norm layer frozen.

    logger.info(
        "post-train backbone norm mode: %s (trainable norm tensors: %d)",
        norm_mode_l,
        trainable_norm_tensors,
    )

    if unfreeze_dwconv:
        for m in backbone.modules():
            if isinstance(m, nn.Conv2d) and m.groups == m.in_channels and m.in_channels > 1:
                for p in m.parameters():
                    p.requires_grad = True


def _load_shape_compatible_state(
    model: nn.Module,
    state: dict[str, torch.Tensor],
) -> tuple[list[str], list[str], list[str]]:
    current = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in state.items():
        if key in current and torch.is_tensor(value) and tuple(current[key].shape) == tuple(value.shape):
            compatible[key] = value
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return list(missing), list(unexpected), skipped


class CurveModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        patch_size: int,
        lora_cfg: dict[str, int | float | bool],
        curve_head_mid: int = 128,
        feature_layers: int = 1,
        input_aa_strength: float = 0.0,
        unfreeze_dwconv: bool = False,
        norm_mode: str = "all_norms",
    ):
        super().__init__()
        self.backbone = backbone
        self.input_aa = InputAntiAlias(strength=float(input_aa_strength)) if float(input_aa_strength) > 0.0 else None
        if hasattr(backbone, "gradient_checkpointing"):
            setattr(backbone, "gradient_checkpointing", True)
        apply_lora_to_backbone(
            backbone,
            num_blocks=int(lora_cfg.get("blocks", 3)),
            r=int(lora_cfg.get("r", 8)),
            alpha=int(lora_cfg.get("alpha", 16)),
            dropout=float(lora_cfg.get("dropout", 0.05)),
            use_mlp=bool(lora_cfg.get("use_mlp", False)),
        )
        freeze_backbone_except_lora_and_norms(
            backbone, unfreeze_dwconv=bool(unfreeze_dwconv), norm_mode=str(norm_mode)
        )
        self.feature_layers = max(1, int(feature_layers))
        if self.feature_layers > 1 and not hasattr(backbone, "get_intermediate_layers"):
            raise ValueError("feature_layers > 1 requires a ViT backbone with get_intermediate_layers().")
        base_C = getattr(backbone, "embed_dim", backbone.num_features if hasattr(backbone, "num_features") else 768)
        C = int(base_C) * self.feature_layers
        self.curve_head = CurveHead(C, mid=int(curve_head_mid))
        self.patch_size = patch_size

    def forward(
        self, images_3chw: torch.Tensor, *, orig_hw: tuple[int, int] = (ORIG_H, ORIG_W)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.input_aa is not None:
            images_3chw = self.input_aa(images_3chw)
        x, pads = pad_to_multiple_hw_center(images_3chw, self.patch_size)
        pt, _, pl, _ = pads
        H_pad, W_pad = x.shape[-2], x.shape[-1]
        if self.feature_layers > 1:
            layer_maps = self.backbone.get_intermediate_layers(
                x,
                n=self.feature_layers,
                reshape=True,
                return_class_token=False,
                norm=True,
            )
            tokens_hw = torch.cat([layer_map for layer_map in layer_maps], dim=1).contiguous()
        else:
            outputs = self.backbone.forward_features(x)
            # cls = outputs["x_norm_clstoken"]
            patch_tokens = outputs["x_norm_patchtokens"]
            H_tokens = H_pad // self.patch_size
            W_tokens = W_pad // self.patch_size
            tokens_hw = patch_tokens.reshape(x.shape[0], H_tokens, W_tokens, -1).permute(0, 3, 1, 2).contiguous()

        logits_pad = self.curve_head(tokens_hw, (H_pad, W_pad))
        H0, W0 = orig_hw
        z_logits = logits_pad[:, pt : pt + H0, pl : pl + W0]
        none_logits = logits_pad[:, -1, pl : pl + W0]
        curve_logits = torch.cat([z_logits, none_logits.unsqueeze(1)], dim=1)
        p = F.softmax(curve_logits.float(), dim=1)
        p_none = p[:, -1, :].mean(dim=1).clamp(1e-4, 1 - 1e-4)
        presence_logits = torch.log((1.0 - p_none) / p_none).to(curve_logits.dtype)

        return presence_logits, curve_logits


def build_optimizer(
    model: CurveModel, lr_head: float, wd_head: float, lr_lora: float, wd_lora: float
) -> torch.optim.Optimizer:
    lora_params: list[nn.Parameter] = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            lora_params += [m.lora_A, m.lora_B]
    lora_param_ids = {id(p) for p in lora_params}
    head_params = [p for p in model.parameters() if p.requires_grad and id(p) not in lora_param_ids]
    return torch.optim.AdamW(
        [
            {"params": head_params, "lr": lr_head, "weight_decay": wd_head},
            {"params": lora_params, "lr": lr_lora, "weight_decay": wd_lora},
        ]
    )


def _make_oct_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((ORIG_H, ORIG_W), interpolation=InterpolationMode.BICUBIC),
            MaybeToTensor(),
            Ensure3CH(),
            PerImageZScore(eps=1e-6),
        ]
    )


def _fallback_post_train_split_indices(entries: np.ndarray, *, seed: int) -> tuple[list[int], list[int]]:
    """Seeded random 90/10 image-level split for datasets without split metadata."""
    codes = entries["code"]
    curve_idx = np.nonzero(codes == 1)[0].astype(np.int64)
    bg_idx = np.nonzero(codes == 2)[0].astype(np.int64)
    if curve_idx.size < 2:
        raise ValueError("Post-train fallback split requires at least two labeled curve samples.")
    if bg_idx.size == 0:
        raise ValueError("Post-train fallback split requires background samples (entries with code==2).")

    rng = np.random.default_rng(int(seed))
    curve_idx = curve_idx.copy()
    bg_idx = bg_idx.copy()
    rng.shuffle(curve_idx)
    rng.shuffle(bg_idx)

    val_frac = 0.1
    val_curve = max(1, min(int(round(curve_idx.size * val_frac)), int(curve_idx.size) - 1))
    if bg_idx.size >= 2:
        val_bg = max(1, min(int(round(bg_idx.size * val_frac)), int(bg_idx.size) - 1))
    else:
        val_bg = 0

    train_idx = np.concatenate([curve_idx[val_curve:], bg_idx[val_bg:]]).astype(np.int64)
    val_idx = np.concatenate([curve_idx[:val_curve], bg_idx[:val_bg]]).astype(np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx.tolist(), val_idx.tolist()


def _post_train_split_indices(
    entries: np.ndarray, *, seed: int, split_mode: str = "auto"
) -> tuple[list[int], list[int], str, int, int]:
    """Return train/val supervised indices and unlabeled bookkeeping counts."""
    mode = (split_mode or "auto").strip().lower()
    if mode not in ("auto", "all"):
        raise ValueError(f"Unknown post-train split_mode {split_mode!r}; expected 'auto' or 'all'.")
    if mode == "all":
        codes = entries["code"]
        curve_idx = np.nonzero(codes == 1)[0].astype(np.int64)
        bg_idx = np.nonzero(codes == 2)[0].astype(np.int64)
        if curve_idx.size < 2:
            raise ValueError("Post-train split_mode=all requires at least two labeled curve samples.")
        if bg_idx.size == 0:
            raise ValueError("Post-train split_mode=all requires background samples (entries with code==2).")
        rng = np.random.default_rng(int(seed))
        train_idx = np.concatenate([curve_idx, bg_idx])
        rng.shuffle(train_idx)
        # Deployment mode: every labeled curve + background sample is trained on. The
        # val subset is drawn FROM the training set purely so the periodic validation /
        # best-checkpoint machinery keeps working — its metrics are train-set metrics.
        n_val_curve = max(1, int(round(curve_idx.size * 0.1)))
        n_val_bg = int(round(bg_idx.size * 0.1))
        val_idx = np.concatenate(
            [
                rng.permutation(curve_idx)[:n_val_curve],
                rng.permutation(bg_idx)[:n_val_bg],
            ]
        )
        rng.shuffle(val_idx)
        return (
            train_idx.tolist(),
            val_idx.tolist(),
            "all",
            int(np.count_nonzero(codes == 0)),
            0,
        )
    if "split" in entries.dtype.names:
        split_values = np.char.lower(entries["split"].astype(str))
        train_idx_all = np.nonzero(split_values == "train")[0].tolist()
        val_idx_all = np.nonzero(split_values == "val")[0].tolist()
        if train_idx_all and val_idx_all:
            train_codes_all = entries[train_idx_all]["code"]
            val_codes_all = entries[val_idx_all]["code"]
            train_idx = [idx for idx in train_idx_all if int(entries[idx]["code"]) in (1, 2)]
            val_idx = [idx for idx in val_idx_all if int(entries[idx]["code"]) in (1, 2)]
            train_codes = entries[train_idx]["code"]
            val_codes = entries[val_idx]["code"]
            if np.count_nonzero(train_codes == 1) == 0:
                raise ValueError("Split-defined post-train set has no labeled curve samples in train.")
            if np.count_nonzero(val_codes == 1) == 0:
                raise ValueError("Split-defined post-train set has no labeled curve samples in val.")
            if np.count_nonzero(train_codes == 2) == 0:
                raise ValueError("Split-defined post-train set has no background samples in train.")
            return (
                train_idx,
                val_idx,
                "explicit",
                int(np.count_nonzero(train_codes_all == 0)),
                int(np.count_nonzero(val_codes_all == 0)),
            )

    logger.warning(
        "No explicit train/val split found (extra/manifest.csv + extra/splits.csv): using a "
        "seeded RANDOM 90/10 image-level split. This is fine for general training on your own "
        "data, but frames from the same recording can land in both partitions, so val is NOT "
        "leak-safe. For recording-level splits (the paper protocol) generate metadata with "
        "tools/data/build_oct_manifest.py + tools/data/build_oct_splits.py."
    )
    train_idx, val_idx = _fallback_post_train_split_indices(entries, seed=seed)
    return train_idx, val_idx, "fallback", 0, 0


class _OCTCodeSubset(Dataset):
    def __init__(
        self,
        base: OCT,
        indices: list[int],
        *,
        augment_cfg: PostTrainAugmentCfg | None = None,
    ) -> None:
        self.base = base
        self.indices = [int(idx) for idx in indices]
        self.augment_cfg = augment_cfg

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, np.ndarray | None, int]:
        dataset_idx = int(self.indices[int(index)])
        image, target = self.base[dataset_idx]
        image = _apply_post_train_aug(image, self.augment_cfg)
        return image, target, self.base.get_code(dataset_idx)


def _collate_oct(batch: Iterable[tuple]) -> dict[str, torch.Tensor]:
    images: list[torch.Tensor] = []
    zs: list[torch.Tensor] = []
    is_bgs: list[int] = []
    for item in batch:
        if len(item) == 3:
            img, target, code = item
            code_i = int(code)
        else:
            img, target = item
            code_i = None
        images.append(img)
        is_bg = target is None if code_i is None else code_i == 2
        if target is None:
            is_bgs.append(1)
            zs.append(torch.zeros(ORIG_W, dtype=torch.float32))
        else:
            t = torch.from_numpy(target.astype("float32"))
            is_bgs.append(1 if is_bg else 0)
            zs.append(t)
    images_t = torch.stack(images, dim=0)
    zs_t = torch.stack(zs, dim=0)
    is_bgs_t = torch.tensor(is_bgs, dtype=torch.long)
    return {"image": images_t, "z": zs_t, "is_bg": is_bgs_t}


def _grad_norm_l2(parameters: list[nn.Parameter]) -> torch.Tensor:
    norms: list[torch.Tensor] = []
    for p in parameters:
        if p.grad is None:
            continue
        norms.append(torch.linalg.vector_norm(p.grad.detach()))
    if not norms:
        return torch.tensor(0.0, device=parameters[0].device if parameters else "cpu")
    return torch.linalg.vector_norm(torch.stack(norms))


def train_step(
    batch: dict[str, torch.Tensor],
    model: CurveModel,
    criterion: CurveLoss,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler | None,
    *,
    sam_rho: float | None = None,
) -> dict[str, float]:
    model.train()
    images = batch["image"].cuda(non_blocking=True)
    is_bg = batch["is_bg"].cuda(non_blocking=True).long()
    z = batch["z"].cuda(non_blocking=True)
    optimizer.zero_grad(set_to_none=True)

    use_sam = sam_rho is not None and float(sam_rho) > 0
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_stepped = False

    if scaler is None:
        with amp.autocast(device_type="cuda", enabled=False):
            presence_logits_1, curve_logits_1 = model(images)
            loss_1, metrics = criterion(curve_logits_1, z, is_bg)
        loss_1.backward()

        loss = loss_1
        presence_logits, curve_logits = presence_logits_1, curve_logits_1

        if use_sam:
            rho = float(sam_rho)
            grad_norm = _grad_norm_l2(trainable_params)
            if torch.isfinite(grad_norm):
                scale = rho / (grad_norm + 1e-12)
                eps_list: list[tuple[nn.Parameter, torch.Tensor]] = []
                with torch.no_grad():
                    for p in trainable_params:
                        if p.grad is None:
                            continue
                        e_w = p.grad * scale
                        p.add_(e_w)
                        eps_list.append((p, e_w))
                optimizer.zero_grad(set_to_none=True)
                with amp.autocast(device_type="cuda", enabled=False):
                    presence_logits_2, curve_logits_2 = model(images)
                    loss_2, _ = criterion(curve_logits_2, z, is_bg)
                loss_2.backward()
                with torch.no_grad():
                    for p, e_w in eps_list:
                        p.sub_(e_w)
            else:
                optimizer.zero_grad(set_to_none=True)

        if trainable_params:
            grad_norm_final = _grad_norm_l2(trainable_params)
            if not torch.isfinite(grad_norm_final):
                optimizer.zero_grad(set_to_none=True)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer_stepped = True
    else:
        with amp.autocast(device_type="cuda", enabled=True):
            presence_logits_1, curve_logits_1 = model(images)
            loss_1, metrics = criterion(curve_logits_1, z, is_bg)
        scaler.scale(loss_1).backward()

        loss = loss_1
        presence_logits, curve_logits = presence_logits_1, curve_logits_1

        if use_sam:
            rho = float(sam_rho)
            grad_norm = _grad_norm_l2(trainable_params)
            if torch.isfinite(grad_norm):
                scale = rho / (grad_norm + 1e-12)
                eps_list: list[tuple[nn.Parameter, torch.Tensor]] = []
                with torch.no_grad():
                    for p in trainable_params:
                        if p.grad is None:
                            continue
                        e_w = p.grad * scale
                        p.add_(e_w)
                        eps_list.append((p, e_w))
                optimizer.zero_grad(set_to_none=True)
                with amp.autocast(device_type="cuda", enabled=True):
                    presence_logits_2, curve_logits_2 = model(images)
                    loss_2, _ = criterion(curve_logits_2, z, is_bg)
                scaler.scale(loss_2).backward()
                with torch.no_grad():
                    for p, e_w in eps_list:
                        p.sub_(e_w)

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        prev_scale = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        optimizer_stepped = float(scaler.get_scale()) >= prev_scale
    with torch.no_grad():
        p_curve = torch.sigmoid(presence_logits)
        mask = (1 - is_bg).float()
        z_hat = soft_argmax_height(curve_logits[:, :-1, :])
        mae = ((z_hat - z).abs().mean(dim=1) * mask).sum() / (mask.sum() + 1e-8)
    return {
        "loss": float(loss.detach().cpu()),
        "mae_px": float(mae.detach().cpu()),
        **{k: float(v.cpu()) for k, v in metrics.items()},
        "p_curve": float(p_curve.mean().detach().cpu()),
        "optimizer_stepped": float(1.0 if optimizer_stepped else 0.0),
    }


@torch.no_grad()
def validate(
    model: CurveModel,
    data_loader: DataLoader,
    device: torch.device,
    criterion: CurveLoss,
    *,
    acc_tolerances: tuple[float, ...] = DEFAULT_ACC_TOLERANCES,
) -> dict[str, float]:
    model.eval()
    loss_col_ce_num_sum = 0.0
    loss_col_ce_weight_sum = 0.0
    loss_smooth_num_sum = 0.0
    loss_smooth_weight_sum = 0.0
    p_curve_sum = 0.0
    n_samples = 0.0
    curve_cnt = 0.0
    metric_sums: dict[str, float] = {
        "mae_px": 0.0,
        "p95_px": 0.0,
        "bias_px": 0.0,
        "abs_bias_px": 0.0,
    }
    for tau in acc_tolerances:
        metric_sums[metric_name_for_tolerance(tau)] = 0.0
    for batch in data_loader:
        images = batch["image"].to(device, non_blocking=True)
        z = batch["z"].to(device, non_blocking=True)
        is_bg = batch["is_bg"].to(device, non_blocking=True).long()
        presence_logits, curve_logits = model(images)

        _, metrics = criterion(curve_logits, z, is_bg)
        bsz = float(images.shape[0])
        non_bg_cnt = float((is_bg == 0).sum().item())
        bg_cnt = bsz - non_bg_cnt
        loss_col_ce_weight = non_bg_cnt + float(criterion.cfg.bg_weight) * bg_cnt
        n_samples += bsz
        loss_col_ce_num_sum += float(metrics.get("loss_col_ce", torch.tensor(0.0)).detach().cpu()) * loss_col_ce_weight
        loss_col_ce_weight_sum += loss_col_ce_weight
        loss_smooth_num_sum += float(metrics.get("loss_smooth", torch.tensor(0.0)).detach().cpu()) * non_bg_cnt
        loss_smooth_weight_sum += non_bg_cnt
        p_curve_sum += float(torch.sigmoid(presence_logits).detach().sum().cpu())

        curve_mask = is_bg == 0
        if curve_mask.any():
            z_hat = soft_argmax_height(curve_logits[:, :-1, :])
            batch_curve_metrics = curve_metrics_batch(
                z_hat[curve_mask],
                z[curve_mask],
                acc_tolerances=acc_tolerances,
            )
            curve_cnt += float(curve_mask.sum().item())
            for metric_name, metric_values in batch_curve_metrics.items():
                metric_sums[metric_name] = metric_sums.get(metric_name, 0.0) + float(metric_values.sum().item())
    denom = max(n_samples, 1.0)
    val_loss_col_ce = loss_col_ce_num_sum / max(loss_col_ce_weight_sum, 1.0)
    val_loss_smooth = loss_smooth_num_sum / max(loss_smooth_weight_sum, 1.0) if loss_smooth_weight_sum > 0 else 0.0
    out = {
        "val_loss": float(criterion.cfg.lambda_curve) * val_loss_col_ce
        + float(criterion.cfg.lambda_curv) * val_loss_smooth,
        "val_loss_col_ce": val_loss_col_ce,
        "val_loss_smooth": val_loss_smooth,
        "val_p_curve": p_curve_sum / denom,
    }
    for metric_name, metric_sum in metric_sums.items():
        out[f"val_{metric_name}"] = metric_sum / max(curve_cnt, 1.0) if curve_cnt > 0 else float("nan")
    return out


def run_post_training(
    *,
    backbone: nn.Module,
    patch_size: int,
    dataset_str: str,
    seed: int = 0,
    split_mode: str = "auto",
    steps: int,
    batch_size: int,
    num_workers: int,
    lr_head: float,
    wd_head: float,
    lr_lora: float,
    wd_lora: float,
    lr_warmup: int = 50,
    min_lr_mult: float = 0.1,
    ema_decay: float = 0.0,
    sigma: float = 1.5,
    lambda_curve: float = 1.0,
    lambda_curv: float = 0.05,
    bg_weight: float = 5.0,
    eps_none: float = 0.02,
    curv_delta: float = 1.0,
    lora_blocks: int = 3,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_use_mlp: bool = False,
    unfreeze_dwconv: bool = False,
    norm_mode: str = "all_norms",
    curve_head_mid: int = 128,
    feature_layers: int = 1,
    train_aug_p: float = 0.0,
    train_aug_types: Sequence[str] | str | None = None,
    train_aug_severity: str = "medium",
    input_aa_strength: float = 0.0,
    init_curve_path: Path | None = None,
    method: str = "sam",
    sam_rho: float = 0.05,
    log_every: int = 10,
    val_every: int = 1,
    device: torch.device,
    output_path: Path,
    best_path: Path,
) -> tuple[Path, dict[str, float], dict[str, float] | None]:
    effective_seed = int(seed)
    fix_random_seeds(effective_seed)

    metrics_path = best_path.parent / "metrics.csv"
    metrics_fh = metrics_path.open("a", newline="")
    metrics_writer = csv.writer(metrics_fh)
    header = ["step", "loss", "mae_px", "loss_col_ce", "loss_smooth", "p_curve", "lr_head", "lr_lora"]
    if metrics_path.stat().st_size == 0:
        metrics_writer.writerow(header)
    else:
        try:
            with metrics_path.open("r", newline="") as fh:
                first = fh.readline().strip()
            if first != ",".join(header):
                metrics_writer.writerow(header)
        except Exception:
            pass

    ds_full = make_dataset(dataset_str=dataset_str, transform=_make_oct_transform())
    if not isinstance(ds_full, OCT):
        raise TypeError(f"Expected OCT dataset for post-training; got {type(ds_full)}")
    entries = ds_full._get_entries()
    train_idx, val_idx, split_source, excluded_train_unlabeled, excluded_val_unlabeled = _post_train_split_indices(
        entries,
        seed=effective_seed,
        split_mode=split_mode,
    )
    train_codes = entries[train_idx]["code"]
    val_codes = entries[val_idx]["code"]

    logger.info(
        "effective post-train split (%s): train=%d (labeled=%d, background=%d), val=%d (labeled=%d, background=%d)",
        split_source,
        len(train_idx),
        int(np.count_nonzero(train_codes == 1)),
        int(np.count_nonzero(train_codes == 2)),
        len(val_idx),
        int(np.count_nonzero(val_codes == 1)),
        int(np.count_nonzero(val_codes == 2)),
    )
    logger.info(
        "excluded unlabeled split entries from supervised post-train: train=%d, val=%d",
        excluded_train_unlabeled,
        excluded_val_unlabeled,
    )
    logger.info("effective post-train seed: %d", effective_seed)
    if split_source == "all":
        logger.warning(
            "post-train split_mode=all: training on ALL labeled+background samples; "
            "val metrics are computed on a subset of the TRAINING data (monitoring only, "
            "not comparable to held-out numbers)."
        )

    train_aug_cfg = None
    if float(train_aug_p) > 0.0:
        train_aug_cfg = PostTrainAugmentCfg(
            p=float(train_aug_p),
            types=_normalize_aug_types(train_aug_types),
            severity=str(train_aug_severity).strip().lower(),
        )
        logger.info(
            "post-train intensity augmentation enabled: p=%.3f types=%s severity=%s",
            train_aug_cfg.p,
            ",".join(train_aug_cfg.types),
            train_aug_cfg.severity,
        )

    ds = _OCTCodeSubset(ds_full, train_idx, augment_cfg=train_aug_cfg)
    ds_val = _OCTCodeSubset(ds_full, val_idx)
    train_generator = torch.Generator()
    train_generator.manual_seed(effective_seed)
    val_generator = torch.Generator()
    val_generator.manual_seed(effective_seed + 1)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=_collate_oct,
        generator=train_generator,
        worker_init_fn=seed_worker,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
        drop_last=False,
        collate_fn=_collate_oct,
        generator=val_generator,
        worker_init_fn=seed_worker,
    )
    if len(dl) == 0:
        raise ValueError(
            f"Post-train train loader is empty: {len(ds)} training samples with "
            f"batch_size={batch_size} and drop_last=True yields zero batches, so the "
            "step loop would spin forever. Reduce post_train.batch_size or add data."
        )

    model = CurveModel(
        backbone,
        patch_size=patch_size,
        lora_cfg={
            "blocks": lora_blocks,
            "r": lora_r,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "use_mlp": lora_use_mlp,
        },
        curve_head_mid=int(curve_head_mid),
        feature_layers=int(feature_layers),
        input_aa_strength=float(input_aa_strength),
        unfreeze_dwconv=bool(unfreeze_dwconv),
        norm_mode=str(norm_mode),
    ).to(device)
    logger.info("post-train input anti-alias strength: %.3f", float(input_aa_strength))
    if init_curve_path is not None:
        init_curve_path = Path(init_curve_path)
        if not init_curve_path.exists():
            raise FileNotFoundError(f"post-train init curve checkpoint not found: {init_curve_path}")
        init_ckpt = torch.load(init_curve_path, map_location="cpu")
        init_state = init_ckpt.get("model", init_ckpt) if isinstance(init_ckpt, dict) else init_ckpt
        if not isinstance(init_state, dict):
            raise ValueError(f"Invalid post-train init curve checkpoint: {init_curve_path}")
        init_state = remap_legacy_curve_head_keys(init_state)
        missing, unexpected, skipped = _load_shape_compatible_state(model, init_state)
        logger.info(
            "initialized post-train curve model from %s (missing=%d unexpected=%d skipped_shape=%d)",
            init_curve_path,
            len(missing),
            len(unexpected),
            len(skipped),
        )
        if skipped:
            logger.info("first shape-incompatible init key skipped: %s", skipped[0])
    logger.info("post-train backbone norm mode: all_norms")
    logger.info("post-train backbone checkpointing: True")
    ema: ModelEMA | None = None
    if 0.0 < float(ema_decay) < 1.0:
        ema = ModelEMA(model, decay=float(ema_decay))
    criterion = CurveLoss(
        LossCfg(
            sigma=sigma,
            lambda_curve=lambda_curve,
            lambda_curv=lambda_curv,
            bg_weight=bg_weight,
            eps_none=eps_none,
            curv_delta=curv_delta,
        )
    )
    logger.info(
        "post-train loss cfg: sigma=%.3f lambda_curve=%.3f lambda_curv=%.3f bg_weight=%.3f "
        "eps_none=%.3f curv_delta=%.3f",
        float(sigma),
        float(lambda_curve),
        float(lambda_curv),
        float(bg_weight),
        float(eps_none),
        float(curv_delta),
    )
    opt = build_optimizer(model, lr_head=lr_head, wd_head=wd_head, lr_lora=lr_lora, wd_lora=wd_lora)

    warmup_steps = min(max(int(lr_warmup), 0), max(int(steps), 1))
    min_lr_mult_f = float(min_lr_mult)

    def lr_mult_for_step(step_num: int) -> float:
        step_num = max(int(step_num), 1)
        if warmup_steps > 0 and step_num <= warmup_steps:
            return step_num / warmup_steps
        t = (step_num - warmup_steps) / max(1, int(steps) - warmup_steps)
        return min_lr_mult_f + (1.0 - min_lr_mult_f) * 0.5 * (1.0 + math.cos(math.pi * t))

    lr_mult_1 = float(lr_mult_for_step(1))
    if not math.isfinite(lr_mult_1) or lr_mult_1 <= 0:
        lr_mult_1 = 1.0
    for pg in opt.param_groups:
        pg["lr"] = float(pg.get("lr", 0.0)) * lr_mult_1

    scheduler = (
        torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: lr_mult_for_step(step + 2) / lr_mult_1)
        if int(steps) > 1
        else None
    )
    scaler = amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    run_meta: dict[str, object] = {
        "meta_format": 1,
        "backbone_class": type(backbone).__name__,
        "patch_size": int(patch_size),
        "curve_head_mid": int(curve_head_mid),
        "feature_layers": int(feature_layers),
        "input_aa_strength": float(input_aa_strength),
        "norm_mode": str(norm_mode),
        "unfreeze_dwconv": bool(unfreeze_dwconv),
        "lora": {
            "blocks": int(lora_blocks),
            "r": int(lora_r),
            "alpha": int(lora_alpha),
            "dropout": float(lora_dropout),
            "use_mlp": bool(lora_use_mlp),
        },
        "loss": {
            "sigma": float(sigma),
            "lambda_curve": float(lambda_curve),
            "lambda_curv": float(lambda_curv),
            "bg_weight": float(bg_weight),
            "eps_none": float(eps_none),
            "curv_delta": float(curv_delta),
        },
        "optim": {
            "method": str(method),
            "sam_rho": float(sam_rho),
            "steps": int(steps),
            "batch_size": int(batch_size),
            "lr_head": float(lr_head),
            "wd_head": float(wd_head),
            "lr_lora": float(lr_lora),
            "wd_lora": float(wd_lora),
            "lr_warmup": int(lr_warmup),
            "min_lr_mult": float(min_lr_mult),
            "ema_decay": float(ema_decay),
        },
        "data": {
            "dataset_str": str(dataset_str),
            "split_mode": str(split_mode),
            "split_source": str(split_source),
            "seed": int(effective_seed),
            "train_aug_p": float(train_aug_p),
            "manifest_md5": file_md5(ds_full._get_extra_full_path("manifest.csv")),
            "splits_md5": file_md5(ds_full._get_extra_full_path("splits.csv")),
        },
        "preprocess": (
            "RGB decode (gray replicated) -> Resize(512x500, bicubic) -> ToTensor [0,1] "
            "-> per-image z-score over all pixels (population std, eps=1e-6)"
        ),
        "git": git_state(),
        "versions": runtime_versions(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    val_every_epochs = max(1, int(val_every))
    best_val_mae = float("inf")
    best_step = 0
    vm_best: dict[str, float] | None = None
    vm_final: dict[str, float] | None = None
    ema_reset_step = max(int(warmup_steps), 1)

    def current_eval_model() -> CurveModel:
        if ema is not None and seen >= ema_reset_step:
            return ema.ema
        return model

    def current_checkpoint_payload() -> dict[str, object]:
        if ema is None or seen < ema_reset_step:
            return {"model": model.state_dict(), "step": int(seen), "meta": run_meta}
        return {
            "model": ema.ema.state_dict(),
            "raw_model": model.state_dict(),
            "ema_decay": float(ema.decay),
            "step": int(seen),
            "meta": run_meta,
        }

    seen = 0
    epoch = 0
    while seen < steps:
        epoch += 1
        for batch in dl:
            method_l = str(method).lower()
            if method_l == "adamw":
                sam = None
            elif method_l == "sam":
                sam = float(sam_rho)
            else:
                raise ValueError(f"Unknown post-train method {method!r}; expected 'sam' or 'adamw'.")
            lr_head_cur = float(opt.param_groups[0].get("lr", 0.0)) if opt.param_groups else 0.0
            lr_lora_cur = float(opt.param_groups[1].get("lr", lr_head_cur)) if len(opt.param_groups) > 1 else lr_head_cur
            stats = train_step(batch, model, criterion, opt, scaler=scaler, sam_rho=sam)
            seen += 1
            if ema is not None and seen >= ema_reset_step:
                if seen == ema_reset_step:
                    ema.ema.load_state_dict(model.state_dict(), strict=True)
                elif seen > ema_reset_step:
                    ema.update(model)
            stats["lr_head"] = lr_head_cur
            stats["lr_lora"] = lr_lora_cur

            if scheduler is not None and seen < steps and bool(stats.get("optimizer_stepped", 0.0) > 0.5):
                scheduler.step()

            if seen % log_every == 0 or seen == 1:
                print(
                    f"[post {seen}/{steps}] loss={stats.get('loss', 0):.4f} "
                    f"mae_px={stats.get('mae_px', 0):.2f} "
                    f"Lcol={stats.get('loss_col_ce', 0):.4f} "
                    f"Lsmooth={stats.get('loss_smooth', 0):.4f} "
                    f"lrh={stats.get('lr_head', 0):.2e} "
                    f"lrl={stats.get('lr_lora', 0):.2e}"
                )
                metrics_writer.writerow(
                    [
                        seen,
                        stats.get("loss", 0.0),
                        stats.get("mae_px", 0.0),
                        stats.get("loss_col_ce", 0.0),
                        stats.get("loss_smooth", 0.0),
                        stats.get("p_curve", 0.0),
                        stats.get("lr_head", 0.0),
                        stats.get("lr_lora", 0.0),
                    ]
                )
                metrics_fh.flush()
            if seen >= steps:
                break

        if epoch % val_every_epochs == 0 or seen >= steps:
            vm_cur = validate(current_eval_model(), dl_val, device, criterion, acc_tolerances=DEFAULT_ACC_TOLERANCES)
            cur_val_mae = float(vm_cur.get("val_mae_px", float("inf")))
            logger.info(
                "[post val epoch %d step %d/%d] val_loss=%.6f val_mae_px=%.3f val_p95_px=%.3f val_acc_2px=%.3f",
                epoch,
                seen,
                steps,
                float(vm_cur.get("val_loss", float("nan"))),
                cur_val_mae,
                float(vm_cur.get("val_p95_px", float("nan"))),
                float(vm_cur.get("val_acc_2px", float("nan"))),
            )
            if math.isfinite(cur_val_mae) and cur_val_mae < best_val_mae:
                best_val_mae = cur_val_mae
                best_step = seen
                vm_best = dict(vm_cur)
                torch.save(current_checkpoint_payload(), best_path)
            if seen >= steps:
                vm_final = vm_cur

    if ema is None:
        torch.save({"model": model.state_dict(), "meta": run_meta}, output_path)
    else:
        torch.save(
            {
                "model": ema.ema.state_dict(),
                "raw_model": model.state_dict(),
                "ema_decay": float(ema.decay),
                "meta": run_meta,
            },
            output_path,
        )
    if vm_final is None:
        vm_final = validate(current_eval_model(), dl_val, device, criterion, acc_tolerances=DEFAULT_ACC_TOLERANCES)
    if vm_best is None:
        vm_best = dict(vm_final)
        best_val_mae = float(vm_final.get("val_mae_px", float("inf")))
        best_step = seen
        torch.save(current_checkpoint_payload(), best_path)
    metrics_fh.close()

    best_source = "final" if best_step == seen else "best_ckpt"
    final_val_loss = float(vm_final.get("val_loss", float("nan")))
    best_ckpt_val_loss = float(vm_best.get("val_loss", float("nan")))
    best_val_loss = float(vm_best.get("val_loss", float("nan")))
    best_val_p95 = float(vm_best.get("val_p95_px", float("nan")))
    best_val_acc2 = float(vm_best.get("val_acc_2px", float("nan")))
    print(
        f"[post] done. val_loss_final={final_val_loss:.6f} "
        f"val_loss_best_ckpt={best_ckpt_val_loss:.6f} "
        f"best_val_mae_px={best_val_mae:.6f} ({best_source}@{best_step}) "
        f"val_p95_px={best_val_p95:.3f} val_acc_2px={best_val_acc2:.3f}"
    )

    try:
        summary_path = best_path.parent / "val_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "final": vm_final,
                    "best_ckpt": vm_best,
                    "best_val_loss": best_val_loss,
                    "best_source": best_source,
                    "best_step": best_step,
                    "best_val_mae_px": best_val_mae,
                    "val_every": val_every_epochs,
                    "val_every_unit": "epochs",
                    "split_source": split_source,
                    "provenance": run_meta,
                },
                indent=2,
            )
            + "\n"
        )
    except Exception:
        pass
    return output_path, vm_final, vm_best


__all__ = [
    "run_post_training",
    "CurveModel",
    "CurveLoss",
    "CurveHead",
    "LoRALinear",
]
