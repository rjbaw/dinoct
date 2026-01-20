import math
from typing import Literal

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, device=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device))

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


class RopePositionEmbedding(nn.Module):
    """
    Minimal rotary position embedding helper returning (sin, cos) tensors.
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device=None,
    ) -> None:
        super().__init__()
        if embed_dim % (4 * num_heads) != 0:
            raise ValueError(
                f"Expected embed_dim % (4*num_heads) == 0, got embed_dim={embed_dim}, num_heads={num_heads}"
            )

        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        head_dim = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.head_dim = head_dim
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype
        self.register_buffer(
            "periods",
            torch.empty(head_dim // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        device = self.periods.device
        dtype = self.dtype or self.periods.dtype

        if self.base is not None:
            periods = self.base ** (
                2 * torch.arange(self.head_dim // 4, device=device, dtype=dtype) / (self.head_dim // 2)
            )
        else:
            assert self.min_period is not None and self.max_period is not None
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.head_dim // 4, device=device, dtype=dtype)
            periods = base**exponents
            periods = periods / base
            periods = periods * self.max_period
        self.periods.data = periods

    def forward(self, *, H: int, W: int) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        device = self.periods.device
        dtype = self.dtype or self.periods.dtype
        dd = {"device": device, "dtype": dtype}

        if self.normalize_coords == "max":
            max_hw = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_hw
            coords_w = torch.arange(0.5, W, **dd) / max_hw
        elif self.normalize_coords == "min":
            min_hw = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_hw
            coords_w = torch.arange(0.5, W, **dd) / min_hw
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H
            coords_w = torch.arange(0.5, W, **dd) / W
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords!r}")

        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords = coords.flatten(0, 1)
        coords = 2.0 * coords - 1.0

        if self.training and self.shift_coords is not None:
            shift = float(self.shift_coords)
            shift_hw = torch.empty(2, **dd).uniform_(-shift, shift)
            coords = coords + shift_hw[None, :]

        if self.training and self.jitter_coords is not None:
            jitter = float(self.jitter_coords)
            jitter_log = math.log(jitter)
            jitter_hw = torch.empty(2, **dd).uniform_(-jitter_log, jitter_log).exp()
            coords = coords * jitter_hw[None, :]

        if self.training and self.rescale_coords is not None:
            rescale = float(self.rescale_coords)
            rescale_log = math.log(rescale)
            rescale_hw = torch.empty(1, **dd).uniform_(-rescale_log, rescale_log).exp()
            coords = coords * rescale_hw

        angles = (2 * math.pi) * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2)
        angles = angles.tile(2)
        return angles.sin(), angles.cos()


__all__ = ["RMSNorm", "RopePositionEmbedding"]
