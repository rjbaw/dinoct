from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from ..data import make_dataset
from ..data.datasets import OCT
from ..data.transforms import Ensure3CH, MaybeToTensor, PerImageZScore

ORIG_H, ORIG_W = 512, 500


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
    bg_weight: float = 20.0
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


def _make_oct_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((ORIG_H, ORIG_W), interpolation=InterpolationMode.BICUBIC),
            MaybeToTensor(),
            Ensure3CH(),
            PerImageZScore(eps=1e-6),
        ]
    )


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
        scaler.step(optimizer)
        scaler.update()
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
    }


@torch.no_grad()
def validate(
    model: CurveModel, data_loader: DataLoader, device: torch.device, criterion: CurveLoss
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    loss_col_ce_sum = 0.0
    loss_smooth_sum = 0.0
    p_curve_sum = 0.0
    n_samples = 0.0
    mae_sum = 0.0
    mae_cnt = 0.0
    for batch in data_loader:
        images = batch["image"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        is_bg = batch["is_bg"].to(device, non_blocking=True).long()
        presence_logits, curve_logits = model(images)

        loss, metrics = criterion(curve_logits, y, is_bg)
        bsz = float(images.shape[0])
        n_samples += bsz
        loss_sum += float(loss.detach().cpu()) * bsz
        loss_col_ce_sum += float(metrics.get("loss_col_ce", torch.tensor(0.0)).detach().cpu()) * bsz
        loss_smooth_sum += float(metrics.get("loss_smooth", torch.tensor(0.0)).detach().cpu()) * bsz
        p_curve_sum += float(torch.sigmoid(presence_logits).detach().sum().cpu())

        mask = (1 - is_bg).float()
        if mask.sum().item() > 0:
            y_hat = soft_argmax_height(curve_logits[:, :-1, :])
            mae_sum += ((y_hat - y).abs().mean(dim=1) * mask).sum().item()
            mae_cnt += mask.sum().item()
    denom = max(n_samples, 1.0)
    return {
        "val_loss": loss_sum / denom,
        "val_loss_col_ce": loss_col_ce_sum / denom,
        "val_loss_smooth": loss_smooth_sum / denom,
        "val_p_curve": p_curve_sum / denom,
        "val_mae_px": (mae_sum / max(mae_cnt, 1.0)) if mae_cnt > 0 else float("nan"),
    }


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
    device: torch.device,
    output_path: Path,
    best_path: Path,
) -> tuple[Path, dict[str, float], dict[str, float] | None]:
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

    # Stratified split to ensure both curve and background appear in train/val.
    curve_idx = np.nonzero(entries["code"] == 1)[0]
    bg_idx = np.nonzero(entries["code"] == 2)[0]
    if curve_idx.size == 0:
        raise ValueError("Post-train requires labeled curve samples (entries with code==1); none found.")
    if bg_idx.size == 0:
        raise ValueError("Post-train requires background samples (entries with code==2); none found.")

    rng = np.random.default_rng(int(seed))
    rng.shuffle(curve_idx)
    rng.shuffle(bg_idx)
    val_frac = 0.1
    val_curve = int(round(curve_idx.size * val_frac))
    val_bg = int(round(bg_idx.size * val_frac))
    val_curve = max(1, min(val_curve, int(curve_idx.size) - 1))
    val_bg = max(1, min(val_bg, int(bg_idx.size) - 1))

    train_idx = np.concatenate([curve_idx[val_curve:], bg_idx[val_bg:]]).tolist()
    val_idx = np.concatenate([curve_idx[:val_curve], bg_idx[:val_bg]]).tolist()
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    ds: Subset
    dl: DataLoader
    ds = Subset(ds_full, train_idx)

    ds_val = Subset(ds_full, val_idx)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=_collate_oct,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
        drop_last=False,
        collate_fn=_collate_oct,
    )

    num_bg = int((entries["code"] == 2).sum())
    num_curve = int((entries["code"] == 1).sum())

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

    best_mae = float("inf")
    seen = 0
    while seen < steps:
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
            stats["lr_head"] = lr_head_cur
            stats["lr_lora"] = lr_lora_cur

            if scheduler is not None and seen < steps:
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
            cur_mae = float(stats.get("mae_px", float("inf")))
            if cur_mae < best_mae:
                best_mae = cur_mae
                torch.save({"model": model.state_dict()}, best_path)
            if seen >= steps:
                break

    torch.save({"model": model.state_dict()}, output_path)
    vm_final = validate(model, dl_val, device, criterion)
    vm_best: dict[str, float] | None = None
    if best_path.exists():
        sd_final = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        try:
            ckpt = torch.load(best_path, map_location="cpu")
            sd_best = ckpt.get("model", ckpt)
            model.load_state_dict(sd_best, strict=False)
            vm_best = validate(model, dl_val, device, criterion)
        finally:
            model.load_state_dict(sd_final, strict=False)
    metrics_fh.close()

    best_vm = (
        vm_best
        if (vm_best and vm_best.get("val_loss", float("inf")) <= vm_final.get("val_loss", float("inf")))
        else vm_final
    )
    best_source = "best_ckpt" if best_vm is vm_best else "final"
    final_val_loss = float(vm_final.get("val_loss", float("nan")))
    best_ckpt_val_loss = float(vm_best.get("val_loss", float("nan"))) if vm_best else float("nan")
    best_val_loss = float(best_vm.get("val_loss", float("nan")))
    print(
        f"[post] done. val_loss_final={final_val_loss:.6f} "
        f"val_loss_best_ckpt={best_ckpt_val_loss:.6f} "
        f"best_val_loss={best_val_loss:.6f} ({best_source})"
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
