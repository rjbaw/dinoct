from __future__ import annotations

import csv
import copy
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.transforms import functional as tvf
from torchvision.transforms import InterpolationMode

from ..data import make_dataset
from ..data.datasets import OCT
from ..data.transforms import Ensure3CH, MaybeToTensor, PerImageZScore
from ..eval import DEFAULT_ACC_TOLERANCES, curve_metrics_batch, estimate_spike_kappa_from_curves, metric_name_for_tolerance
from ..utils import fix_random_seeds

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


def gaussian_targets_from_y(y: torch.Tensor, H: int, sigma: float = 1.5) -> torch.Tensor:
    """y: (B, W) pixel coords -> (B, H, W) Gaussian targets per column."""
    B, W = y.shape
    grid = torch.arange(H, device=y.device, dtype=y.dtype).view(1, H, 1)
    g = torch.exp(-0.5 * ((grid - y.unsqueeze(1)) / sigma) ** 2)
    g = g / (g.sum(dim=1, keepdim=True) + 1e-8)
    return g


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
    y_hat = soft_argmax_height(logits_hw)
    d2 = y_hat[:, 2:] - 2 * y_hat[:, 1:-1] + y_hat[:, :-2]
    curv_per_sample = d2.abs().mean(dim=1)
    m = non_bg_mask.float()
    if m.sum() == 0:
        return logits_hw.new_zeros(())
    return (curv_per_sample * m).sum() / (m.sum() + 1e-8)


def entropy_weight(curve_logits_y: torch.Tensor) -> torch.Tensor:
    """Return (B,W) normalized entropy weights in ~[0,1] (detached)."""
    p = F.softmax(curve_logits_y.float(), dim=1)
    ent = -(p * torch.log(p + 1e-8)).sum(dim=1)  # (B,W)
    ent = ent / max(math.log(p.shape[1]), 1e-8)
    return ent.detach()


def robust_curv_loss(curve_logits_y: torch.Tensor, non_bg_mask: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Confidence-weighted (via entropy) robust smoothness over W using a Huber penalty on d2."""
    y_hat = soft_argmax_height(curve_logits_y.float())  # (B,W)
    if y_hat.shape[1] < 3:
        return curve_logits_y.new_zeros(())
    d2 = y_hat[:, 2:] - 2 * y_hat[:, 1:-1] + y_hat[:, :-2]  # (B,W-2)

    w = entropy_weight(curve_logits_y)[:, 1:-1]  # align to W-2

    absd = d2.abs()
    delta_f = max(float(delta), 1e-6)
    huber = torch.where(absd < delta_f, 0.5 * (d2**2) / delta_f, absd - 0.5 * delta_f)
    per_sample = (w * huber).mean(dim=1)

    m = non_bg_mask.float()
    if m.sum() == 0:
        return curve_logits_y.new_zeros(())
    return (per_sample * m).sum() / (m.sum() + 1e-8)


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


@dataclass(frozen=True)
class PostTrainAugmentConfig:
    hflip_prob: float = 0.0
    hshift_max: int = 0
    vshift_max: int = 0
    gamma_prob: float = 0.0
    gamma_min: float = 1.0
    gamma_max: float = 1.0
    contrast_prob: float = 0.0
    contrast_min: float = 1.0
    contrast_max: float = 1.0
    blur_prob: float = 0.0
    blur_sigma_min: float = 0.0
    blur_sigma_max: float = 0.0
    noise_prob: float = 0.0
    noise_std_max: float = 0.0
    occlusion_prob: float = 0.0
    occlusion_count_min: int = 1
    occlusion_count_max: int = 1
    occlusion_width_min: int = 0
    occlusion_width_max: int = 0
    occlusion_noise_std: float = 0.0
    occlusion_alpha_min: float = 0.0
    occlusion_alpha_max: float = 0.0


def _augment_cfg_from_preset(preset: str) -> PostTrainAugmentConfig:
    preset_l = str(preset).strip().lower()
    if preset_l in {"", "none"}:
        return PostTrainAugmentConfig()
    if preset_l in {"geo", "geom", "geometry"}:
        return PostTrainAugmentConfig(
            hflip_prob=0.5,
            hshift_max=1,
            vshift_max=2,
        )
    if preset_l == "light":
        return PostTrainAugmentConfig(
            hflip_prob=0.5,
            hshift_max=2,
            vshift_max=3,
            gamma_prob=0.10,
            gamma_min=0.96,
            gamma_max=1.06,
            contrast_prob=0.10,
            contrast_min=0.96,
            contrast_max=1.06,
            blur_prob=0.06,
            blur_sigma_min=0.20,
            blur_sigma_max=0.45,
            noise_prob=0.06,
            noise_std_max=0.008,
        )
    if preset_l in {"light_occ", "light_occlusion"}:
        return PostTrainAugmentConfig(
            hflip_prob=0.5,
            hshift_max=2,
            vshift_max=3,
            gamma_prob=0.10,
            gamma_min=0.96,
            gamma_max=1.06,
            contrast_prob=0.10,
            contrast_min=0.96,
            contrast_max=1.06,
            blur_prob=0.06,
            blur_sigma_min=0.20,
            blur_sigma_max=0.45,
            noise_prob=0.06,
            noise_std_max=0.008,
            occlusion_prob=0.10,
            occlusion_count_min=1,
            occlusion_count_max=1,
            occlusion_width_min=4,
            occlusion_width_max=8,
            occlusion_noise_std=0.008,
            occlusion_alpha_min=0.12,
            occlusion_alpha_max=0.22,
        )
    if preset_l == "robust":
        return PostTrainAugmentConfig(
            hflip_prob=0.5,
            hshift_max=3,
            vshift_max=4,
            gamma_prob=0.14,
            gamma_min=0.94,
            gamma_max=1.08,
            contrast_prob=0.14,
            contrast_min=0.94,
            contrast_max=1.08,
            blur_prob=0.08,
            blur_sigma_min=0.20,
            blur_sigma_max=0.60,
            noise_prob=0.08,
            noise_std_max=0.010,
        )
    if preset_l in {"robust_occ", "robust_occlusion"}:
        return PostTrainAugmentConfig(
            hflip_prob=0.5,
            hshift_max=3,
            vshift_max=4,
            gamma_prob=0.14,
            gamma_min=0.94,
            gamma_max=1.08,
            contrast_prob=0.14,
            contrast_min=0.94,
            contrast_max=1.08,
            blur_prob=0.08,
            blur_sigma_min=0.20,
            blur_sigma_max=0.60,
            noise_prob=0.08,
            noise_std_max=0.010,
            occlusion_prob=0.14,
            occlusion_count_min=1,
            occlusion_count_max=1,
            occlusion_width_min=5,
            occlusion_width_max=10,
            occlusion_noise_std=0.010,
            occlusion_alpha_min=0.15,
            occlusion_alpha_max=0.28,
        )
    raise ValueError(
        f"Unknown post-train augmentation preset {preset!r}. Expected one of: "
        "'none', 'geo', 'light', 'light_occ', 'robust', 'robust_occ'."
    )


def _rand_bool(prob: float) -> bool:
    if prob <= 0.0:
        return False
    return bool(torch.rand(()) < float(prob))


def _rand_uniform(lo: float, hi: float) -> float:
    if hi <= lo:
        return float(lo)
    return float(torch.empty((), dtype=torch.float32).uniform_(float(lo), float(hi)).item())


def _rand_int(lo: int, hi: int) -> int:
    if hi <= lo:
        return int(lo)
    return int(torch.randint(int(lo), int(hi) + 1, ()).item())


def _shift_image_hw(image: torch.Tensor, *, dx: int, dy: int, fill: float) -> torch.Tensor:
    c, h, w = image.shape
    out = image.new_full((c, h, w), float(fill))

    src_y0 = max(0, -int(dy))
    src_y1 = min(h, h - int(dy))
    src_x0 = max(0, -int(dx))
    src_x1 = min(w, w - int(dx))
    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return out

    dst_y0 = max(0, int(dy))
    dst_x0 = max(0, int(dx))
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    out[:, dst_y0:dst_y1, dst_x0:dst_x1] = image[:, src_y0:src_y1, src_x0:src_x1]
    return out


def _shift_target_x(y: np.ndarray, dx: int) -> np.ndarray:
    dx_i = int(dx)
    if dx_i == 0:
        return y.copy()
    out = np.empty_like(y)
    if dx_i > 0:
        out[:dx_i] = y[0]
        out[dx_i:] = y[:-dx_i]
    else:
        k = -dx_i
        out[-k:] = y[-1]
        out[:-k] = y[k:]
    return out


def _apply_gamma(image: torch.Tensor, gamma: float) -> torch.Tensor:
    gamma_f = max(float(gamma), 1e-4)
    return image.clamp(0.0, 1.0).pow(gamma_f)


def _apply_contrast(image: torch.Tensor, factor: float) -> torch.Tensor:
    mean = image.mean(dim=(1, 2), keepdim=True)
    return ((image - mean) * float(factor) + mean).clamp(0.0, 1.0)


def _apply_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    sigma_f = max(float(sigma), 1e-4)
    kernel = max(3, int(round(sigma_f * 6.0)))
    if kernel % 2 == 0:
        kernel += 1
    return tvf.gaussian_blur(image, kernel_size=[kernel, kernel], sigma=[sigma_f, sigma_f])


def _apply_noise(image: torch.Tensor, std: float) -> torch.Tensor:
    std_f = max(float(std), 0.0)
    if std_f <= 0.0:
        return image
    noise = torch.randn((1, image.shape[1], image.shape[2]), device=image.device, dtype=image.dtype) * std_f
    return (image + noise.expand_as(image)).clamp(0.0, 1.0)


def _apply_vertical_occlusion(image: torch.Tensor, cfg: PostTrainAugmentConfig) -> torch.Tensor:
    width_lo = max(1, int(cfg.occlusion_width_min))
    width_hi = max(width_lo, int(cfg.occlusion_width_max))
    count = _rand_int(int(cfg.occlusion_count_min), int(cfg.occlusion_count_max))
    c, h, w = image.shape
    out = image
    for _ in range(max(1, count)):
        band_w = min(w, _rand_int(width_lo, width_hi))
        if band_w <= 0 or band_w >= w:
            x0 = 0
            x1 = w
        else:
            x0 = _rand_int(0, w - band_w)
            x1 = x0 + band_w
        fill = torch.full((1, h, x1 - x0), float(out.mean().item()), dtype=out.dtype, device=out.device)
        if cfg.occlusion_noise_std > 0:
            fill = fill + torch.randn_like(fill) * float(cfg.occlusion_noise_std)
        alpha = _rand_uniform(float(cfg.occlusion_alpha_min), float(cfg.occlusion_alpha_max))
        alpha_t = torch.tensor(alpha, dtype=out.dtype, device=out.device)
        src = out[:, :, x0:x1]
        mixed = src * (1.0 - alpha_t) + fill.expand(c, -1, -1) * alpha_t
        out[:, :, x0:x1] = mixed.clamp(0.0, 1.0)
    return out


class PostTrainTensorDataset(Dataset[tuple[torch.Tensor, np.ndarray | None]]):
    def __init__(self, dataset: Dataset[tuple[torch.Tensor, np.ndarray | None]], *, augment_cfg: PostTrainAugmentConfig):
        self.dataset = dataset
        self.augment_cfg = augment_cfg
        self.normalize = PerImageZScore(eps=1e-6)

    def __len__(self) -> int:
        return len(self.dataset)

    def _augment(self, image: torch.Tensor, target: np.ndarray | None) -> tuple[torch.Tensor, np.ndarray | None]:
        cfg = self.augment_cfg
        out = image.clamp(0.0, 1.0)
        y = None if target is None else np.asarray(target, dtype=np.float32).copy()

        if _rand_bool(cfg.hflip_prob):
            out = torch.flip(out, dims=(2,))
            if y is not None:
                y = y[::-1].copy()

        dx = _rand_int(-int(cfg.hshift_max), int(cfg.hshift_max)) if int(cfg.hshift_max) > 0 else 0
        dy = _rand_int(-int(cfg.vshift_max), int(cfg.vshift_max)) if int(cfg.vshift_max) > 0 else 0
        if dx != 0 or dy != 0:
            out = _shift_image_hw(out, dx=dx, dy=dy, fill=float(out.mean().item()))
            if y is not None:
                if dx != 0:
                    y = _shift_target_x(y, dx)
                if dy != 0:
                    y = np.clip(y + float(dy), 0.0, float(ORIG_H - 1)).astype(np.float32, copy=False)

        if _rand_bool(cfg.gamma_prob):
            out = _apply_gamma(out, _rand_uniform(cfg.gamma_min, cfg.gamma_max))
        if _rand_bool(cfg.contrast_prob):
            out = _apply_contrast(out, _rand_uniform(cfg.contrast_min, cfg.contrast_max))
        if _rand_bool(cfg.blur_prob):
            out = _apply_blur(out, _rand_uniform(cfg.blur_sigma_min, cfg.blur_sigma_max))
        if _rand_bool(cfg.noise_prob):
            out = _apply_noise(out, _rand_uniform(0.0, cfg.noise_std_max))
        if _rand_bool(cfg.occlusion_prob):
            out = _apply_vertical_occlusion(out, cfg)

        return out.clamp(0.0, 1.0), y

    def __getitem__(self, index: int) -> tuple[torch.Tensor, np.ndarray | None]:
        image, target = self.dataset[index]
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected tensor image from base dataset; got {type(image)}")
        image, target = self._augment(image, target)
        return self.normalize(image), target


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
    blocks = list(vit.blocks)[-num_blocks:]
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
    classes 0..H-1 = curve at row y, class H = no-curve.
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
        self.out_y = nn.Conv2d(mid, 1, kernel_size=1, bias=True)
        self.out_none = nn.Conv2d(mid, 1, kernel_size=1, bias=True)

    def forward(self, tokens_hw: torch.Tensor, out_size_hw: tuple[int, int]) -> torch.Tensor:
        x = self.proj(tokens_hw)
        x = self.act1(self.vert1(x))
        x = self.act2(self.vert2(x))
        h = self.horiz1(self.h_act(self.h_norm(x)))
        gamma = self.h_gamma.to(dtype=h.dtype)
        x = x + self.h_drop(gamma * h)

        H_out, W_out = out_size_hw

        # y logits map -> upsample to (H_out, W_out)
        y_logits = self.out_y(x)
        y_logits = F.interpolate(y_logits, size=(H_out, W_out), mode="bilinear", align_corners=False)
        y_logits = y_logits.squeeze(1)

        # no-curve logits per column
        col_feat = x.mean(dim=2, keepdim=True)
        none_logits = self.out_none(col_feat)
        none_logits = F.interpolate(none_logits, size=(1, W_out), mode="bilinear", align_corners=False)
        none_logits = none_logits.squeeze(1).squeeze(1)

        return torch.cat([y_logits, none_logits.unsqueeze(1)], dim=1)


@dataclass
class LossCfg:
    sigma: float = 1.5
    lambda_curve: float = 1.0
    lambda_curv: float = 0.05
    bg_weight: float = 5.0
    eps_none: float = 0.02
    curv_delta: float = 1.0


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
        self, curve_logits: torch.Tensor, y_curve: torch.Tensor, is_bg: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self.cfg
        B, H1, W = curve_logits.shape
        H = H1 - 1
        non_bg = (1 - is_bg).float()

        with torch.no_grad():
            g = gaussian_targets_from_y(y_curve, H=H, sigma=cfg.sigma)
            eps_none = float(cfg.eps_none)
            targets_none = (is_bg.float().view(B, 1, 1) * 1.0) + ((1 - is_bg).float().view(B, 1, 1) * eps_none)
            targets_none = targets_none.expand(B, 1, W)
            targets_y = g * (1.0 - targets_none)  # ensures sum across (H+1) is 1
            targets = torch.cat([targets_y, targets_none], dim=1)

        w = torch.ones((B,), device=curve_logits.device, dtype=curve_logits.dtype)
        w = torch.where(is_bg.bool(), w * float(cfg.bg_weight), w)

        loss_curve = column_ce_loss_h1w(curve_logits, targets, sample_weight=w)
        loss_curv = robust_curv_loss(curve_logits[:, :H, :], non_bg_mask=non_bg, delta=float(cfg.curv_delta))
        total = cfg.lambda_curve * loss_curve + cfg.lambda_curv * loss_curv
        return total, {
            "loss_col_ce": loss_curve.detach(),
            "loss_smooth": loss_curv.detach(),
        }


def freeze_backbone_except_lora_and_norms(backbone: nn.Module, train_norms: bool = True):
    for p in backbone.parameters():
        p.requires_grad = False
    for m in backbone.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad = True
            m.lora_B.requires_grad = True

    if train_norms:
        for m in backbone.modules():
            if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
                for p in m.parameters():
                    p.requires_grad = True


class CurveModel(nn.Module):
    def __init__(self, backbone: nn.Module, *, patch_size: int, lora_cfg: dict[str, int | float | bool]):
        super().__init__()
        self.backbone = backbone
        apply_lora_to_backbone(
            backbone,
            num_blocks=int(lora_cfg.get("blocks", 3)),
            r=int(lora_cfg.get("r", 8)),
            alpha=int(lora_cfg.get("alpha", 16)),
            dropout=float(lora_cfg.get("dropout", 0.05)),
            use_mlp=bool(lora_cfg.get("use_mlp", False)),
        )
        freeze_backbone_except_lora_and_norms(backbone)
        C = getattr(backbone, "embed_dim", backbone.num_features if hasattr(backbone, "num_features") else 768)
        self.curve_head = CurveHead(C)
        self.patch_size = patch_size

    def forward(
        self, images_3chw: torch.Tensor, *, orig_hw: tuple[int, int] = (ORIG_H, ORIG_W)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, pads = pad_to_multiple_hw_center(images_3chw, self.patch_size)
        pt, _, pl, _ = pads
        H_pad, W_pad = x.shape[-2], x.shape[-1]
        outputs = self.backbone.forward_features(x)
        # cls = outputs[0]["x_norm_clstoken"]
        patch_tokens = outputs[0]["x_norm_patchtokens"]
        H_tokens = H_pad // self.patch_size
        W_tokens = W_pad // self.patch_size
        tokens_hw = patch_tokens.reshape(x.shape[0], H_tokens, W_tokens, -1).permute(0, 3, 1, 2).contiguous()

        logits_pad = self.curve_head(tokens_hw, (H_pad, W_pad))
        H0, W0 = orig_hw
        y_logits = logits_pad[:, pt : pt + H0, pl : pl + W0]
        none_logits = logits_pad[:, -1, pl : pl + W0]
        curve_logits = torch.cat([y_logits, none_logits.unsqueeze(1)], dim=1)
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


def _make_oct_transform(*, normalize: bool = True) -> transforms.Compose:
    transforms_list: list[object] = [
        transforms.Resize((ORIG_H, ORIG_W), interpolation=InterpolationMode.BICUBIC),
        MaybeToTensor(),
        Ensure3CH(),
    ]
    if normalize:
        transforms_list.append(PerImageZScore(eps=1e-6))
    return transforms.Compose(transforms_list)


def _collate_oct(batch: Iterable[tuple[torch.Tensor, np.ndarray | None]]) -> dict[str, torch.Tensor]:
    images: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    is_bgs: list[int] = []
    for img, target in batch:
        images.append(img)
        if target is None:
            is_bgs.append(1)
            ys.append(torch.zeros(ORIG_W, dtype=torch.float32))
        else:
            t = torch.from_numpy(target.astype("float32"))
            is_bgs.append(0 if t.sum() != 0 else 1)
            ys.append(t)
    images_t = torch.stack(images, dim=0)
    ys_t = torch.stack(ys, dim=0)
    is_bgs_t = torch.tensor(is_bgs, dtype=torch.long)
    return {"image": images_t, "y": ys_t, "is_bg": is_bgs_t}


def _grad_norm_l2(parameters: list[nn.Parameter]) -> torch.Tensor:
    norms: list[torch.Tensor] = []
    for p in parameters:
        if p.grad is None:
            continue
        norms.append(torch.linalg.vector_norm(p.grad.detach()))
    if not norms:
        return torch.tensor(0.0, device=parameters[0].device if parameters else "cpu")
    return torch.linalg.vector_norm(torch.stack(norms))


def _estimate_spike_kappa_for_indices(dataset: OCT, indices: list[int], *, quantile: float) -> float | None:
    curves: list[np.ndarray] = []
    entries = dataset._get_entries()
    for idx in indices:
        if int(entries[idx]["code"]) != 1:
            continue
        target = dataset.get_target(int(idx))
        if target is None:
            continue
        curves.append(np.asarray(target, dtype=np.float32))
    if not curves:
        return None
    return estimate_spike_kappa_from_curves(curves, quantile=quantile)


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
    y = batch["y"].cuda(non_blocking=True)
    optimizer.zero_grad(set_to_none=True)

    use_sam = sam_rho is not None and float(sam_rho) > 0
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_stepped = False

    if scaler is None:
        with amp.autocast(device_type="cuda", enabled=False):
            presence_logits_1, curve_logits_1 = model(images)
            loss_1, metrics = criterion(curve_logits_1, y, is_bg)
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
                    loss_2, _ = criterion(curve_logits_2, y, is_bg)
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
            loss_1, metrics = criterion(curve_logits_1, y, is_bg)
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
                    loss_2, _ = criterion(curve_logits_2, y, is_bg)
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
        y_hat = soft_argmax_height(curve_logits[:, :-1, :])
        mae = ((y_hat - y).abs().mean(dim=1) * mask).sum() / (mask.sum() + 1e-8)
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
    spike_kappa: float | None = None,
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
    if spike_kappa is not None:
        metric_sums["spike_rate"] = 0.0
    for batch in data_loader:
        images = batch["image"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        is_bg = batch["is_bg"].to(device, non_blocking=True).long()
        presence_logits, curve_logits = model(images)

        loss, metrics = criterion(curve_logits, y, is_bg)
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
            y_hat = soft_argmax_height(curve_logits[:, :-1, :])
            batch_curve_metrics = curve_metrics_batch(
                y_hat[curve_mask],
                y[curve_mask],
                acc_tolerances=acc_tolerances,
                spike_kappa=spike_kappa,
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
    if spike_kappa is not None:
        out["val_spike_kappa"] = float(spike_kappa)
    return out


def run_post_training(
    *,
    backbone: nn.Module,
    patch_size: int,
    dataset_str: str,
    seed: int = 0,
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
    sigma: float,
    lambda_curve: float,
    lambda_curv: float,
    eps_none: float = 0.02,
    curv_delta: float = 1.0,
    lora_blocks: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_use_mlp: bool,
    method: str = "sam",
    sam_rho: float = 0.05,
    log_every: int,
    val_every: int = 1,
    snapshot_every_steps: int = 0,
    best_metric: str = "val_mae_px",
    spike_kappa: float | None = None,
    spike_kappa_quantile: float = 0.99,
    aug_preset: str = "none",
    device: torch.device,
    output_path: Path,
    best_path: Path,
    snapshot_dir: Path | None = None,
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

    aug_cfg = _augment_cfg_from_preset(aug_preset)
    use_augment_wrapper = any(
        (
            aug_cfg.hflip_prob > 0.0,
            int(aug_cfg.hshift_max) > 0,
            int(aug_cfg.vshift_max) > 0,
            aug_cfg.gamma_prob > 0.0,
            aug_cfg.contrast_prob > 0.0,
            aug_cfg.blur_prob > 0.0,
            aug_cfg.noise_prob > 0.0,
            aug_cfg.occlusion_prob > 0.0,
        )
    )
    ds_full = make_dataset(
        dataset_str=dataset_str,
        transform=_make_oct_transform(normalize=not use_augment_wrapper),
    )
    if not isinstance(ds_full, OCT):
        raise TypeError(f"Expected OCT dataset for post-training; got {type(ds_full)}")
    entries = ds_full._get_entries()

    train_idx: list[int]
    val_idx: list[int]
    if "split" in entries.dtype.names:
        split_values = np.char.lower(entries["split"].astype(str))
        train_idx_np = np.nonzero(split_values == "train")[0]
        val_idx_np = np.nonzero(split_values == "val")[0]
        if train_idx_np.size > 0 and val_idx_np.size > 0:
            train_idx = train_idx_np.tolist()
            val_idx = val_idx_np.tolist()
        else:
            train_idx = []
            val_idx = []
    else:
        train_idx = []
        val_idx = []

    if not train_idx or not val_idx:
        # Stratified split to ensure labeled samples appear in train/val.
        curve_idx = np.nonzero(entries["code"] == 1)[0]
        bg_idx = np.nonzero(entries["code"] == 2)[0]
        if curve_idx.size == 0:
            raise ValueError("Post-train requires labeled curve samples (entries with code==1); none found.")

        rng = np.random.default_rng(int(seed))
        rng.shuffle(curve_idx)
        val_frac = 0.1
        val_curve = int(round(curve_idx.size * val_frac))
        val_curve = max(1, min(val_curve, int(curve_idx.size) - 1))

        if bg_idx.size == 0:
            logger.warning("Post-train dataset has no background samples; using labeled-only train/val split.")
            train_idx = curve_idx[val_curve:].tolist()
            val_idx = curve_idx[:val_curve].tolist()
        else:
            rng.shuffle(bg_idx)
            val_bg = int(round(bg_idx.size * val_frac))
            val_bg = max(1, min(val_bg, int(bg_idx.size) - 1))
            train_idx = np.concatenate([curve_idx[val_curve:], bg_idx[val_bg:]]).tolist()
            val_idx = np.concatenate([curve_idx[:val_curve], bg_idx[:val_bg]]).tolist()
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
    else:
        train_codes = entries["code"][train_idx]
        val_codes = entries["code"][val_idx]
        if np.count_nonzero(train_codes == 1) == 0:
            raise ValueError("Split-defined post-train set has no labeled curve samples in train.")
        if np.count_nonzero(val_codes == 1) == 0:
            raise ValueError("Split-defined post-train set has no labeled curve samples in val.")
        if np.count_nonzero(train_codes == 2) == 0:
            logger.warning("Split-defined post-train set has no background samples in train; continuing in labeled-only mode.")

    train_codes = entries["code"][train_idx]
    val_codes = entries["code"][val_idx]
    logger.info(
        "effective post-train split: train=%d (labeled=%d, background=%d), val=%d (labeled=%d, background=%d)",
        len(train_idx),
        int(np.count_nonzero(train_codes == 1)),
        int(np.count_nonzero(train_codes == 2)),
        len(val_idx),
        int(np.count_nonzero(val_codes == 1)),
        int(np.count_nonzero(val_codes == 2)),
    )

    logger.info("post-train augmentation preset: %s", str(aug_preset).strip().lower() or "none")
    logger.info("effective post-train seed: %d", effective_seed)
    ds: Dataset[tuple[torch.Tensor, np.ndarray | None]]
    dl: DataLoader
    if use_augment_wrapper:
        ds = PostTrainTensorDataset(Subset(ds_full, train_idx), augment_cfg=aug_cfg)
        ds_val = PostTrainTensorDataset(Subset(ds_full, val_idx), augment_cfg=PostTrainAugmentConfig())
    else:
        ds = Subset(ds_full, train_idx)
        ds_val = Subset(ds_full, val_idx)

    train_set_size = len(ds)
    val_set_size = len(ds_val)
    if train_set_size <= 0:
        raise ValueError('skin_db post-train split produced an empty train set.')
    if val_set_size <= 0:
        raise ValueError('skin_db post-train split produced an empty val set.')

    requested_batch_size = int(batch_size)
    effective_train_batch_size = min(requested_batch_size, train_set_size)
    effective_val_batch_size = min(requested_batch_size, val_set_size)
    train_drop_last = train_set_size >= requested_batch_size
    if effective_train_batch_size != requested_batch_size:
        logger.warning(
            'skin_db train set (%d samples) is smaller than requested batch size %d; using batch size %d with drop_last=False.',
            train_set_size,
            requested_batch_size,
            effective_train_batch_size,
        )

    train_generator = torch.Generator()
    train_generator.manual_seed(effective_seed)
    dl = DataLoader(
        ds,
        batch_size=effective_train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train_drop_last,
        collate_fn=_collate_oct,
        generator=train_generator,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=effective_val_batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
        drop_last=False,
        collate_fn=_collate_oct,
    )
    if len(dl) == 0:
        raise ValueError(
            f'skin_db post-train DataLoader is empty: train_set_size={train_set_size}, '
            f'batch_size={effective_train_batch_size}, drop_last={train_drop_last}'
        )
    spike_kappa_value = (
        float(spike_kappa)
        if spike_kappa is not None
        else _estimate_spike_kappa_for_indices(ds_full, val_idx, quantile=float(spike_kappa_quantile))
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
    ).to(device)
    ema: ModelEMA | None = None
    ema_decay_f = float(ema_decay)
    if 0.0 < ema_decay_f < 1.0:
        ema = ModelEMA(model, decay=ema_decay_f)
    criterion = CurveLoss(
        LossCfg(
            sigma=sigma,
            lambda_curve=lambda_curve,
            lambda_curv=lambda_curv,
            eps_none=eps_none,
            curv_delta=curv_delta,
        )
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_every = max(0, int(snapshot_every_steps))
    if snapshot_every > 0:
        snapshot_dir = snapshot_dir or (best_path.parent / "snapshots")
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    val_every_steps = max(1, int(val_every))
    best_metric_name = str(best_metric).strip() or "val_mae_px"
    maximize_best_metric = best_metric_name.startswith("val_acc_")
    best_metric_value = -float("inf") if maximize_best_metric else float("inf")
    best_step = 0
    vm_best: dict[str, float] | None = None
    vm_final: dict[str, float] | None = None

    def current_eval_model() -> CurveModel:
        if ema is not None and seen >= warmup_steps:
            return ema.ema
        return model

    def current_checkpoint_payload() -> dict[str, object]:
        if ema is None or seen < warmup_steps:
            return {"model": model.state_dict(), "step": int(seen)}
        return {
            "model": ema.ema.state_dict(),
            "raw_model": model.state_dict(),
            "ema_decay": float(ema.decay),
            "step": int(seen),
        }

    def is_better_metric(metric_value: float) -> bool:
        if not math.isfinite(metric_value):
            return False
        if maximize_best_metric:
            return metric_value > best_metric_value
        return metric_value < best_metric_value

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
                raise ValueError(f"Unknown post_train.method={method!r} (expected 'adamw' or 'sam').")

            lr_head_cur = float(opt.param_groups[0].get("lr", 0.0)) if opt.param_groups else 0.0
            lr_lora_cur = (
                float(opt.param_groups[1].get("lr", lr_head_cur)) if len(opt.param_groups) > 1 else lr_head_cur
            )
            stats = train_step(batch, model, criterion, opt, scaler=scaler, sam_rho=sam)
            seen += 1
            if ema is not None and seen >= warmup_steps:
                if seen == warmup_steps:
                    ema.ema.load_state_dict(model.state_dict(), strict=True)
                elif seen > warmup_steps:
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
            if snapshot_every > 0 and snapshot_dir is not None and (seen % snapshot_every == 0 or seen >= steps):
                snapshot_path = snapshot_dir / f"fused_curve_step_{seen:05d}.pth"
                torch.save(current_checkpoint_payload(), snapshot_path)
            if seen >= steps:
                break
        if seen % val_every_steps == 0 or seen >= steps:
            vm_cur = validate(
                current_eval_model(),
                dl_val,
                device,
                criterion,
                acc_tolerances=DEFAULT_ACC_TOLERANCES,
                spike_kappa=spike_kappa_value,
            )
            cur_metric_value = float(
                vm_cur.get(best_metric_name, -float("inf") if maximize_best_metric else float("inf"))
            )
            logger.info(
                "[post val step %d/%d] selected_metric[%s]=%.6f "
                "val_loss=%.6f val_mae_px=%.3f val_p95_px=%.3f val_acc_2px=%.3f",
                seen,
                steps,
                best_metric_name,
                cur_metric_value,
                float(vm_cur.get("val_loss", float("nan"))),
                float(vm_cur.get("val_mae_px", float("nan"))),
                float(vm_cur.get("val_p95_px", float("nan"))),
                float(vm_cur.get("val_acc_2px", float("nan"))),
            )
            if best_metric_name not in vm_cur:
                raise KeyError(
                    f"Requested best_metric={best_metric_name!r}, but validate() produced keys: {sorted(vm_cur.keys())}"
                )
            if is_better_metric(cur_metric_value):
                best_metric_value = cur_metric_value
                best_step = seen
                vm_best = dict(vm_cur)
                torch.save(current_checkpoint_payload(), best_path)
            if seen >= steps:
                vm_final = vm_cur

    if ema is None:
        torch.save({"model": model.state_dict()}, output_path)
    else:
        torch.save(
            {
                "model": ema.ema.state_dict(),
                "raw_model": model.state_dict(),
                "ema_decay": float(ema.decay),
            },
            output_path,
        )
    if vm_final is None:
        vm_final = validate(
            current_eval_model(),
            dl_val,
            device,
            criterion,
            acc_tolerances=DEFAULT_ACC_TOLERANCES,
            spike_kappa=spike_kappa_value,
        )
    if vm_best is None:
        vm_best = dict(vm_final)
        best_metric_value = float(
            vm_final.get(best_metric_name, -float("inf") if maximize_best_metric else float("inf"))
        )
        best_step = seen
        torch.save(current_checkpoint_payload(), best_path)
    metrics_fh.close()

    best_vm = vm_best
    best_source = "final" if best_step == seen else "best_ckpt"
    final_val_loss = float(vm_final.get("val_loss", float("nan")))
    best_ckpt_val_loss = float(vm_best.get("val_loss", float("nan")))
    best_val_loss = float(best_vm.get("val_loss", float("nan")))
    best_metric_result = float(best_vm.get(best_metric_name, float("nan")))
    best_val_mae = float(best_vm.get("val_mae_px", float("nan")))
    best_val_p95 = float(best_vm.get("val_p95_px", float("nan")))
    best_val_acc2 = float(best_vm.get("val_acc_2px", float("nan")))
    print(
        f"[post] done. val_loss_final={final_val_loss:.6f} "
        f"val_loss_best_ckpt={best_ckpt_val_loss:.6f} "
        f"best_metric[{best_metric_name}]={best_metric_result:.6f} ({best_source}@{best_step}) "
        f"val_mae_px={best_val_mae:.3f} val_p95_px={best_val_p95:.3f} val_acc_2px={best_val_acc2:.3f}"
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
                    "best_metric": best_metric_name,
                    "best_metric_value": best_metric_result,
                    "val_every": val_every_steps,
                    "val_every_unit": "steps",
                    "snapshot_every_steps": snapshot_every,
                    "spike_kappa": spike_kappa_value,
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
