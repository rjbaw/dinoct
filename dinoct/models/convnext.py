import logging
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init
from torch import Tensor, nn

from ..layers.drop_path import DropPath

logger = logging.getLogger("dinoct")


class LayerNorm(nn.Module):
    """LayerNorm supporting channels_last or channels_first inputs."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_last") -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        if data_format not in {"channels_last", "channels_first"}:
            raise ValueError(f"Unsupported data_format={data_format!r}")
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: Tensor) -> Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # channels_first
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block using channels_last MLP for speed (DwConv -> LN -> MLP -> residual)."""

    def __init__(self, dim: int, *, drop_path: float = 0.0, layer_scale_init_value: float = 1e-6) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = residual + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    """
    ConvNeXt backbone adapted to the DINO/iBOT interface.

    Key differences vs vanilla ConvNeXt:
    - Exposes `forward_features_list()` returning dicts with `x_norm_clstoken` / `x_norm_patchtokens`.
    - Supports iBOT-style masking by zeroing masked *input* patches (pixel-space masking).
    - Optionally resizes the final feature map to match a given `patch_size` token grid.
    """

    def __init__(
        self,
        *,
        in_chans: int = 3,
        depths: Sequence[int] = (3, 3, 9, 3),
        dims: Sequence[int] = (96, 192, 384, 768),
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        patch_size: int | None = None,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.n_storage_tokens = 0

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, int(dims[0]), kernel_size=4, stride=4),
            LayerNorm(int(dims[0]), eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(int(dims[i]), eps=1e-6, data_format="channels_first"),
                nn.Conv2d(int(dims[i]), int(dims[i + 1]), kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        dp_rates = [x for x in np.linspace(0, drop_path_rate, sum(int(d) for d in depths))]
        cur = 0
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[
                    ConvNeXtBlock(
                        dim=int(dims[i]),
                        drop_path=float(dp_rates[cur + j]),
                        layer_scale_init_value=float(layer_scale_init_value),
                    )
                    for j in range(int(depths[i]))
                ]
            )
            self.stages.append(stage)
            cur += int(depths[i])

        self.embed_dim = int(dims[-1])
        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.head = nn.Identity()

        # Best-effort `n_blocks` for layer-wise decay code paths.
        self.n_blocks = int(sum(int(d) for d in depths))

    def init_weights(self) -> None:
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.LayerNorm):
            module.reset_parameters()
        if isinstance(module, LayerNorm):
            module.weight = nn.Parameter(torch.ones(module.normalized_shape))
            module.bias = nn.Parameter(torch.zeros(module.normalized_shape))
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _mask_input(self, x: Tensor, masks_flat: Tensor) -> Tensor:
        if self.patch_size is None:
            raise ValueError("ConvNeXt masking requires `patch_size` to be set.")
        if masks_flat is None:
            return x

        B, _, H, W = x.shape
        ph = int(self.patch_size)
        ht, wt = H // ph, W // ph
        expected = ht * wt
        if masks_flat.ndim != 2 or masks_flat.shape[0] != B or masks_flat.shape[1] != expected:
            raise ValueError(
                f"Expected masks shape (B, {expected}) for input {(H, W)} and patch_size={ph}, got {tuple(masks_flat.shape)}"
            )

        mask_grid = masks_flat.view(B, ht, wt)
        if (H % ph) == 0 and (W % ph) == 0:
            mask_px = mask_grid.repeat_interleave(ph, dim=1).repeat_interleave(ph, dim=2)
        else:
            mask_px = F.interpolate(mask_grid[:, None].float(), size=(H, W), mode="nearest").to(dtype=torch.bool)[:, 0]
        return x.masked_fill(mask_px[:, None].to(device=x.device, dtype=torch.bool), 0.0)

    def _maybe_resize_tokens(self, feats: Tensor, *, input_hw: tuple[int, int]) -> Tensor:
        if self.patch_size is None:
            return feats
        H, W = input_hw
        ph = int(self.patch_size)
        target_h, target_w = H // ph, W // ph
        if (feats.shape[-2], feats.shape[-1]) == (target_h, target_w):
            return feats
        return F.interpolate(feats, size=(target_h, target_w), mode="bilinear", antialias=True)

    def forward_features_list(self, x_list: list[Tensor], masks_list: list[Tensor | None]) -> list[dict[str, Tensor]]:
        output: list[dict[str, Tensor]] = []
        for x, masks in zip(x_list, masks_list):
            input_hw = (int(x.shape[-2]), int(x.shape[-1]))
            if masks is not None:
                x = self._mask_input(x, masks)

            for i in range(4):
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)

            x_pool = x.mean(dim=(-2, -1))  # (B, C)
            x_tokens = self._maybe_resize_tokens(x, input_hw=input_hw)
            x_seq = x_tokens.flatten(2).transpose(1, 2)  # (B, HW, C)

            x_norm = self.norm(torch.cat([x_pool.unsqueeze(1), x_seq], dim=1))
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1:1],  # empty
                    "x_norm_patchtokens": x_norm[:, 1:],
                    "x_prenorm": x_seq,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x: Tensor | list[Tensor], masks: Tensor | None = None) -> list[dict[str, Tensor]]:
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])
        if masks is None:
            return self.forward_features_list(x, [None for _ in x])
        if isinstance(masks, list):
            return self.forward_features_list(x, masks)
        raise TypeError("When `x` is a list, `masks` must be a list[Tensor|None] or None.")

    def forward(
        self, x: Tensor, *, is_training: bool = False, masks: Tensor | None = None
    ) -> Tensor | list[dict[str, Tensor]]:
        ret = self.forward_features(x, masks=masks)
        if is_training:
            return ret
        return self.head(ret[0]["x_norm_clstoken"])


_CONVNEXT_SPECS: dict[str, dict[str, Sequence[int]]] = {
    "tiny": {"depths": (3, 3, 9, 3), "dims": (96, 192, 384, 768)},
    "small": {"depths": (3, 3, 27, 3), "dims": (96, 192, 384, 768)},
}


def convnext_tiny(*, patch_size: int | None = None, **kwargs) -> ConvNeXt:
    return ConvNeXt(patch_size=patch_size, **_CONVNEXT_SPECS["tiny"], **kwargs)


def convnext_small(*, patch_size: int | None = None, **kwargs) -> ConvNeXt:
    return ConvNeXt(patch_size=patch_size, **_CONVNEXT_SPECS["small"], **kwargs)
