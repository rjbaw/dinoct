"""Additive denoising / robustness auxiliary for OCT SSL.

Mechanism: train the student to RECONSTRUCT a clean crop from an OCT-CORRUPTED
version of it. This directly optimizes feature stability under realistic OCT
corruption (speckle + signal dropout). It is a reconstruction objective in pixel
space, distinct from appearance-jitter invariance augmentation.

Design constraints (multitask SSL auxiliaries are fragile; cf. Rivail et al. 2024):
  * ADDITIVE and LOW-WEIGHT, ramped from 0 -> target over a warmup, so it cannot
    swamp DINO/iBOT.
  * Fully isolated: disabled unless `train.denoise_aux: true`; when off, training is
    byte-identical to the baseline (no extra forward, no head, no loss term).
  * The reconstruction target is the (already-augmented) clean global crop, detached.

The recon forward reuses the student at the SAME 224 resolution as the existing
global crops, so it adds ~one global forward (no new compile shape).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenoiseHead(nn.Module):
    """Lightweight conv decoder: patch tokens (B,N,C) -> reconstructed image (B,3,S,S).

    The 16x16 token grid is upsampled to the crop size; the recon is intentionally
    low-frequency (it cannot recover high-freq from a 16x16 grid), which is exactly
    the retinal-band structure we want the backbone to denoise robustly.
    """

    def __init__(self, in_dim: int, out_size: int = 224, mid: int = 256, out_ch: int = 3):
        super().__init__()
        self.out_size = int(out_size)
        self.proj = nn.Conv2d(in_dim, mid, kernel_size=1)
        self.act = nn.GELU()
        self.conv = nn.Conv2d(mid, mid // 2, kernel_size=3, padding=1)
        self.out = nn.Conv2d(mid // 2, out_ch, kernel_size=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, n, c = tokens.shape
        g = int(round(n ** 0.5))
        if g * g != n:
            raise ValueError(f"DenoiseHead expects a square token grid, got N={n}")
        x = tokens.transpose(1, 2).reshape(b, c, g, g)
        x = self.act(self.proj(x))
        x = self.act(self.conv(x))
        x = self.out(x)
        return F.interpolate(x, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)


def corrupt_oct(
    x: torch.Tensor,
    *,
    noise_std: float = 0.4,
    dropout_p: float = 0.5,
    dropout_frac: float = 0.25,
    intensity: float = 0.15,
) -> torch.Tensor:
    """OCT-realistic corruption of a per-image z-scored crop (B,3,H,W).

    - additive Gaussian (speckle-like on a normalized input),
    - block signal-dropout (random ~dropout_frac of an 8x8 block grid zeroed, in
      dropout_p of images) -> mimics OCT A-line/signal loss,
    - per-image intensity/contrast shift.
    Returns a corrupted copy; does not modify x in place.
    """
    out = x
    if noise_std > 0:
        out = out + noise_std * torch.randn_like(out)
    if dropout_p > 0 and dropout_frac > 0:
        b, _, h, w = x.shape
        gh = gw = 8
        keep = (torch.rand(b, 1, gh, gw, device=x.device, dtype=x.dtype) >= dropout_frac).to(x.dtype)
        keep = F.interpolate(keep, size=(h, w), mode="nearest")
        apply = (torch.rand(b, 1, 1, 1, device=x.device, dtype=x.dtype) < dropout_p).to(x.dtype)
        # where apply: multiply by keep-mask (zero dropped blocks); else unchanged
        out = out * (1.0 - apply * (1.0 - keep))
    if intensity > 0:
        b = x.shape[0]
        scale = 1.0 + intensity * (2.0 * torch.rand(b, 1, 1, 1, device=x.device, dtype=x.dtype) - 1.0)
        shift = intensity * (2.0 * torch.rand(b, 1, 1, 1, device=x.device, dtype=x.dtype) - 1.0)
        out = out * scale + shift
    return out


def denoise_recon_loss(
    student: nn.Module,
    denoise_head: nn.Module,
    global_list: list[torch.Tensor],
    corrupt_kwargs: dict,
) -> torch.Tensor:
    """L1 reconstruction of each clean global crop from its OCT-corrupted version.

    Reuses the student at 224 (same shape as the existing globals -> no new compile
    graph). Target = the clean (augmented) global crop, detached.
    """
    corrupt_list = [corrupt_oct(g, **corrupt_kwargs) for g in global_list]
    outs = student.forward_features_list(corrupt_list, [None for _ in corrupt_list])
    total = global_list[0].new_zeros(())
    for o, clean in zip(outs, global_list):
        pred = denoise_head(o["x_norm_patchtokens"])
        total = total + F.l1_loss(pred, clean.detach())
    return total / max(1, len(global_list))
