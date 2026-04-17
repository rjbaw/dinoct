from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


LEARNED_BASELINE_MODELS = ("unet", "unet_paper", "fcbr")
SEGMENTATION_BASELINE_MODELS = ("unet_paper",)


def curve_logits_to_presence(curve_logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(curve_logits.float(), dim=1)
    p_none = p[:, -1, :].mean(dim=1).clamp(1e-4, 1 - 1e-4)
    return torch.log((1.0 - p_none) / p_none).to(curve_logits.dtype)


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | None = None,
        groups: int = 1,
    ) -> None:
        if padding is None:
            if isinstance(kernel_size, tuple):
                padding = tuple(k // 2 for k in kernel_size)
            else:
                padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_ch),
            nn.GELU(),
        )


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(in_ch, out_ch, kernel_size=3),
            ConvNormAct(out_ch, out_ch, kernel_size=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpFuseBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.fuse = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


class DepthwiseSeparableBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        stride: int | tuple[int, int] = 1,
        dilation: int | tuple[int, int] = 1,
    ) -> None:
        super().__init__()
        if isinstance(dilation, tuple):
            pad = dilation
        else:
            pad = dilation
        self.depthwise = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=3,
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=in_ch,
            bias=False,
        )
        self.depthwise_norm = nn.GroupNorm(1, in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.pointwise_norm = nn.GroupNorm(1, out_ch)
        self.act = nn.GELU()
        if in_ch == out_ch and stride == 1:
            self.skip: nn.Module = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(1, out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.depthwise(x)
        x = self.act(self.depthwise_norm(x))
        x = self.pointwise(x)
        x = self.pointwise_norm(x)
        return self.act(x + residual)


class CurveDistributionHead(nn.Module):
    """Produce per-column depth logits with an extra no-curve class."""

    def __init__(self, in_channels: int, mid_channels: int = 128) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=True)
        self.vert1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(5, 1), padding=(2, 0), groups=mid_channels)
        self.vert2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(5, 1), padding=(2, 0), groups=mid_channels)
        self.act1 = nn.GELU()
        self.act2 = nn.GELU()
        self.h_norm = nn.GroupNorm(1, mid_channels)
        self.h_act = nn.GELU()
        self.horiz = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=(1, 9),
            padding=(0, 4),
            groups=mid_channels,
            padding_mode="replicate",
            bias=False,
        )
        nn.init.zeros_(self.horiz.weight)
        self.h_gamma = nn.Parameter(1e-3 * torch.ones(mid_channels, 1, 1))
        self.h_drop = nn.Dropout(p=0.1)
        self.out_y = nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True)
        self.out_none = nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True)

    def forward(self, features: torch.Tensor, out_size_hw: tuple[int, int]) -> torch.Tensor:
        x = self.proj(features)
        x = self.act1(self.vert1(x))
        x = self.act2(self.vert2(x))
        h = self.horiz(self.h_act(self.h_norm(x)))
        gamma = self.h_gamma.to(dtype=h.dtype)
        x = x + self.h_drop(gamma * h)

        out_h, out_w = out_size_hw
        y_logits = self.out_y(x)
        y_logits = F.interpolate(y_logits, size=(out_h, out_w), mode="bilinear", align_corners=False)
        y_logits = y_logits.squeeze(1)

        col_feat = x.mean(dim=2, keepdim=True)
        none_logits = self.out_none(col_feat)
        none_logits = F.interpolate(none_logits, size=(1, out_w), mode="bilinear", align_corners=False)
        none_logits = none_logits.squeeze(1).squeeze(1)
        return torch.cat([y_logits, none_logits.unsqueeze(1)], dim=1)


class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 3) -> None:
        super().__init__()
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, features: torch.Tensor, out_size_hw: tuple[int, int]) -> torch.Tensor:
        logits = self.out(features)
        if logits.shape[-2:] != out_size_hw:
            logits = F.interpolate(logits, size=out_size_hw, mode="bilinear", align_corners=False)
        return logits


class UNetCurveModel(nn.Module):
    def __init__(self, in_chans: int = 3, base_channels: int = 32, head_channels: int = 128) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 6
        bottleneck_ch = base_channels * 8

        self.enc1 = DoubleConv(in_chans, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(c4, bottleneck_ch)

        self.up4 = UpFuseBlock(bottleneck_ch, c4, c4)
        self.up3 = UpFuseBlock(c4, c3, c3)
        self.up2 = UpFuseBlock(c3, c2, c2)
        self.up1 = UpFuseBlock(c2, c1, c1)
        self.head = CurveDistributionHead(c1, mid_channels=head_channels)

    def forward(
        self, images_3chw: torch.Tensor, *, orig_hw: tuple[int, int] = (512, 500)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(images_3chw)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bottleneck = self.bottleneck(self.pool(e4))

        x = self.up4(bottleneck, e4)
        x = self.up3(x, e3)
        x = self.up2(x, e2)
        x = self.up1(x, e1)
        curve_logits = self.head(x, orig_hw)
        return curve_logits_to_presence(curve_logits), curve_logits


class PaperUNetSegModel(nn.Module):
    """Paper-style semantic-segmentation UNet adapted to a single-boundary OCT task."""

    def __init__(self, in_chans: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        bottleneck_ch = base_channels * 16

        self.enc1 = DoubleConv(in_chans, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(c4, bottleneck_ch)

        self.up4 = UpFuseBlock(bottleneck_ch, c4, c4)
        self.up3 = UpFuseBlock(c4, c3, c3)
        self.up2 = UpFuseBlock(c3, c2, c2)
        self.up1 = UpFuseBlock(c2, c1, c1)
        self.head = SegmentationHead(c1, out_channels=3)

    def forward(self, images_3chw: torch.Tensor, *, orig_hw: tuple[int, int] | None = None) -> torch.Tensor:
        out_hw = tuple(orig_hw or images_3chw.shape[-2:])

        e1 = self.enc1(images_3chw)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bottleneck = self.bottleneck(self.pool(e4))

        x = self.up4(bottleneck, e4)
        x = self.up3(x, e3)
        x = self.up2(x, e2)
        x = self.up1(x, e1)
        return self.head(x, out_hw)


class FCBRCurveModel(nn.Module):
    def __init__(self, in_chans: int = 3, base_channels: int = 32, head_channels: int = 96) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 3
        self.stem = ConvNormAct(in_chans, c1, kernel_size=5, stride=(2, 1))
        self.block1 = DepthwiseSeparableBlock(c1, c1, stride=1)
        self.down2 = DepthwiseSeparableBlock(c1, c2, stride=(2, 1))
        self.block2 = DepthwiseSeparableBlock(c2, c2, stride=1)
        self.down3 = DepthwiseSeparableBlock(c2, c3, stride=(2, 1))
        self.context = nn.Sequential(
            DepthwiseSeparableBlock(c3, c3, dilation=1),
            DepthwiseSeparableBlock(c3, c3, dilation=2),
            DepthwiseSeparableBlock(c3, c3, dilation=4),
            DepthwiseSeparableBlock(c3, c3, dilation=2),
            DepthwiseSeparableBlock(c3, c3, dilation=1),
        )
        self.fuse = nn.Sequential(
            ConvNormAct(c3, c2, kernel_size=1, padding=0),
            ConvNormAct(c2, c2, kernel_size=3),
        )
        self.head = CurveDistributionHead(c2, mid_channels=head_channels)

    def forward(
        self, images_3chw: torch.Tensor, *, orig_hw: tuple[int, int] = (512, 500)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(images_3chw)
        x = self.block1(x)
        x = self.down2(x)
        x = self.block2(x)
        x = self.down3(x)
        x = self.context(x)
        x = self.fuse(x)
        curve_logits = self.head(x, orig_hw)
        return curve_logits_to_presence(curve_logits), curve_logits


def decode_paper_unet_logits(
    seg_logits: torch.Tensor,
    *,
    epidermis_class: int = 1,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decode a semantic-segmentation UNet into a single surface curve.

    For each column, if the epidermis/boundary class exceeds the threshold,
    use the midpoint of the first and last positive pixel. Otherwise fall back
    to the class-probability argmax along depth.
    """
    probs = F.softmax(seg_logits.float(), dim=1)
    epi = probs[:, int(epidermis_class), :, :]
    mask = epi > float(threshold)

    argmax_y = epi.argmax(dim=1).float()
    y_hat = argmax_y.clone()

    top = mask.float().argmax(dim=1)
    bottom = mask.flip(1).float().argmax(dim=1)
    bottom = (mask.shape[1] - 1) - bottom
    present = mask.any(dim=1)
    midpoint = 0.5 * (top.float() + bottom.float())
    y_hat = torch.where(present, midpoint, y_hat)

    presence_prob = epi.amax(dim=1).mean(dim=1).clamp(1e-4, 1.0 - 1e-4)
    presence_logits = torch.log(presence_prob / (1.0 - presence_prob)).to(seg_logits.dtype)
    return y_hat.to(seg_logits.dtype), presence_logits


def is_segmentation_baseline_model_type(model_type: str) -> bool:
    return str(model_type).strip().lower() in SEGMENTATION_BASELINE_MODELS


def build_learned_baseline_model(
    model_type: str,
    *,
    in_chans: int = 3,
    base_channels: int = 32,
    head_channels: int | None = None,
) -> nn.Module:
    key = str(model_type).strip().lower()
    if key == "unet":
        return UNetCurveModel(
            in_chans=in_chans,
            base_channels=base_channels,
            head_channels=int(head_channels or max(base_channels * 4, 128)),
        )
    if key == "unet_paper":
        return PaperUNetSegModel(
            in_chans=in_chans,
            base_channels=base_channels,
        )
    if key == "fcbr":
        return FCBRCurveModel(
            in_chans=in_chans,
            base_channels=base_channels,
            head_channels=int(head_channels or max(base_channels * 3, 96)),
        )
    raise ValueError(f"Unknown learned baseline model_type={model_type!r}; expected one of {LEARNED_BASELINE_MODELS}")


def infer_model_type_from_checkpoint(ckpt: dict[str, Any]) -> str | None:
    model_type = ckpt.get("model_type") if isinstance(ckpt, dict) else None
    if model_type is None:
        return None
    key = str(model_type).strip().lower()
    return key if key in LEARNED_BASELINE_MODELS else None


__all__ = [
    "LEARNED_BASELINE_MODELS",
    "CurveDistributionHead",
    "UNetCurveModel",
    "PaperUNetSegModel",
    "FCBRCurveModel",
    "build_learned_baseline_model",
    "curve_logits_to_presence",
    "decode_paper_unet_logits",
    "infer_model_type_from_checkpoint",
    "is_segmentation_baseline_model_type",
]
