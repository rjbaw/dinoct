import logging
from collections.abc import Callable

import torch
from torch import nn, Tensor

from ..utils import cat_keep_shapes, uncat_with_shapes

from .attention import CausalSelfAttention, SelfAttention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .ffn_layers import Mlp


logger = logging.getLogger("dinoct")


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = SelfAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        mask_k_bias: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
            device=device,
        )
        self.ls1 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()
        # self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            device=device,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    @staticmethod
    def _maybe_index_rope(rope: tuple[Tensor, Tensor] | None, indices: Tensor) -> tuple[Tensor, Tensor] | None:
        if rope is None:
            return None

        sin, cos = rope
        assert sin.ndim == cos.ndim
        if sin.ndim == 4:
            # sin/cos are [1, heads, tokens, dim]; avoid out-of-bounds when batch > 1
            if sin.shape[0] == 1:
                return sin, cos
            # [batch, heads, patchs, embed_dim]
            return sin[indices], cos[indices]
        else:
            # [heads, patchs, embed_dim] or [patches, embed_dim]
            return sin, cos

    def _forward(self, x: Tensor, rope=None) -> Tensor:
        """
        Reference implementation for a single tensor, matching the list variant below.
        """
        b, _, _ = x.shape
        sample_subset_size = max(int(b * (1 - self.sample_drop_ratio)), 1)
        residual_scale_factor = b / sample_subset_size

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1 = (torch.randperm(b, device=x.device))[:sample_subset_size]

            x_subset_1 = x[indices_1]
            rope_subset = self._maybe_index_rope(rope, indices_1)
            residual_1 = self.attn(self.norm1(x_subset_1), rope=rope_subset)

            x_attn = torch.index_add(
                x,
                dim=0,
                source=self.ls1(residual_1),
                index=indices_1,
                alpha=residual_scale_factor,
            )

            indices_2 = (torch.randperm(b, device=x.device))[:sample_subset_size]

            x_subset_2 = x_attn[indices_2]
            residual_2 = self.mlp(self.norm2(x_subset_2))

            x_ffn = torch.index_add(
                x_attn,
                dim=0,
                source=self.ls2(residual_2),
                index=indices_2,
                alpha=residual_scale_factor,
            )
        else:
            x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
            x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))

        return x_ffn

    def _forward_list(self, x_list: list[Tensor], rope_list: list | None = None) -> list[Tensor]:
        b_list = [x.shape[0] for x in x_list]
        sample_subset_sizes = [max(int(b * (1 - self.sample_drop_ratio)), 1) for b in b_list]
        residual_scale_factors = [b / sample_subset_size for b, sample_subset_size in zip(b_list, sample_subset_sizes)]

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_1_list = [x[indices_1] for x, indices_1 in zip(x_list, indices_1_list)]

            if rope_list is not None:
                rope_subset_list = [
                    self._maybe_index_rope(rope, indices_1) for rope, indices_1 in zip(rope_list, indices_1_list)
                ]
            else:
                rope_subset_list = rope_list

            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_1_list)
            norm1 = uncat_with_shapes(self.norm1(flattened), shapes, num_tokens)
            residual_1_list = self.attn.forward_list(norm1, rope_list=rope_subset_list)

            x_attn_list = [
                torch.index_add(
                    x,
                    dim=0,
                    source=self.ls1(residual_1),
                    index=indices_1,
                    alpha=residual_scale_factor,
                )
                for x, residual_1, indices_1, residual_scale_factor in zip(
                    x_list, residual_1_list, indices_1_list, residual_scale_factors
                )
            ]

            indices_2_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_2_list = [x[indices_2] for x, indices_2 in zip(x_attn_list, indices_2_list)]
            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_2_list)
            norm2_flat = self.norm2(flattened)
            norm2_list = uncat_with_shapes(norm2_flat, shapes, num_tokens)

            residual_2_list = self.mlp.forward_list(norm2_list)

            x_ffn = [
                torch.index_add(
                    x_attn,
                    dim=0,
                    source=self.ls2(residual_2),
                    index=indices_2,
                    alpha=residual_scale_factor,
                )
                for x_attn, residual_2, indices_2, residual_scale_factor in zip(
                    x_attn_list,
                    residual_2_list,
                    indices_2_list,
                    residual_scale_factors,
                )
            ]
        else:
            if rope_list is None:
                rope_list = [None for _ in x_list]
            x_out = []
            for x, rope in zip(x_list, rope_list):
                x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
                x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))
                x_out.append(x_ffn)
            x_ffn = x_out

        return x_ffn

    def forward(self, x_or_x_list: Tensor | list, rope_or_rope_list: list | None = None) -> list[Tensor]:
        if isinstance(x_or_x_list, Tensor):
            return self._forward_list([x_or_x_list], rope_list=[rope_or_rope_list])
        if isinstance(x_or_x_list, list):
            if rope_or_rope_list is None:
                rope_or_rope_list = [None for _ in x_or_x_list]
            return self._forward_list(x_or_x_list, rope_list=rope_or_rope_list)
        raise AssertionError("Unsupported input type for SelfAttentionBlock.forward")


class CausalAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        ls_init_value: float | None = None,
        is_causal: bool = True,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        dropout_prob: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.is_causal = is_causal
        self.ls1 = LayerScale(dim, init_values=ls_init_value) if ls_init_value else nn.Identity()
        self.attention_norm = norm_layer(dim)
        self.attention = CausalSelfAttention(dim, num_heads, attn_drop=dropout_prob, proj_drop=dropout_prob)

        self.ffn_norm = norm_layer(dim)
        ffn_hidden_dim = int(dim * ffn_ratio)
        self.feed_forward = Mlp(
            in_features=dim,
            hidden_features=ffn_hidden_dim,
            drop=dropout_prob,
            act_layer=act_layer,
        )

        self.ls2 = LayerScale(dim, init_values=ls_init_value) if ls_init_value else nn.Identity()

    def init_weights(
        self,
        init_attn_std: float | None = None,
        init_proj_std: float | None = None,
        init_fc_std: float | None = None,
        factor: float = 1.0,
    ) -> None:
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        init_fc_std = init_fc_std or (2 * self.dim) ** -0.5
        self.attention.init_weights(init_attn_std, init_proj_std)
        self.attention_norm.reset_parameters()
        nn.init.normal_(self.feed_forward.fc1.weight, std=init_fc_std)
        nn.init.normal_(self.feed_forward.fc2.weight, std=init_proj_std)
        self.ffn_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
    ):
        x_attn = x + self.ls1(self.attention(self.attention_norm(x), self.is_causal))
        x_ffn = x_attn + self.ls2(self.feed_forward(self.ffn_norm(x_attn)))
        return x_ffn


# def drop_add_residual_stochastic_depth(
#     x: Tensor,
#     residual_func: Callable[[Tensor], Tensor],
#     sample_drop_ratio: float = 0.0,
# ) -> Tensor:
#     # 1) extract subset using permutation
#     b, n, d = x.shape
#     sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
#     brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
#     x_subset = x[brange]

#     # 2) apply residual_func to get residual
#     residual = residual_func(x_subset)

#     x_flat = x.flatten(1)
#     residual = residual.flatten(1)

#     residual_scale_factor = b / sample_subset_size

#     # 3) add the residual
#     x_plus_residual = torch.index_add(
#         x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor
#     )
#     return x_plus_residual.view_as(x)


# def get_branges_scales(x, sample_drop_ratio=0.0):
#     b, n, d = x.shape
#     sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
#     brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
#     residual_scale_factor = b / sample_subset_size
#     return brange, residual_scale_factor


# def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
#     if scaling_vector is None:
#         x_flat = x.flatten(1)
#         residual = residual.flatten(1)
#         x_plus_residual = torch.index_add(
#             x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor
#         )
#     else:
#         x_plus_residual = scaled_index_add(
#             x,
#             brange,
#             residual.to(dtype=x.dtype),
#             scaling=scaling_vector,
#             alpha=residual_scale_factor,
#         )
#     return x_plus_residual


# attn_bias_cache: Dict[Tuple, Any] = {}


# def get_attn_bias_and_cat(x_list, branges=None):
#     """
#     this will perform the index select, cat the tensors, and provide the attn_bias from cache
#     """
#     batch_sizes = (
#         [b.shape[0] for b in branges]
#         if branges is not None
#         else [x.shape[0] for x in x_list]
#     )
#     all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
#     if all_shapes not in attn_bias_cache.keys():
#         seqlens = []
#         for b, x in zip(batch_sizes, x_list):
#             for _ in range(b):
#                 seqlens.append(x.shape[1])
#         attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
#         attn_bias._batch_sizes = batch_sizes
#         attn_bias_cache[all_shapes] = attn_bias

#     if branges is not None:
#         cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(
#             1, -1, x_list[0].shape[-1]
#         )
#     else:
#         tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
#         cat_tensors = torch.cat(tensors_bs1, dim=1)

#     return attn_bias_cache[all_shapes], cat_tensors


# def drop_add_residual_stochastic_depth_list(
#     x_list: List[Tensor],
#     residual_func: Callable[[Tensor, Any], Tensor],
#     sample_drop_ratio: float = 0.0,
#     scaling_vector=None,
# ) -> Tensor:
#     # 1) generate random set of indices for dropping samples in the batch
#     branges_scales = [
#         get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list
#     ]
#     branges = [s[0] for s in branges_scales]
#     residual_scale_factors = [s[1] for s in branges_scales]

#     # 2) get attention bias and index+concat the tensors
#     attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

#     # 3) apply residual_func to get residual, and split the result
#     residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

#     outputs = []
#     for x, brange, residual, residual_scale_factor in zip(
#         x_list, branges, residual_list, residual_scale_factors
#     ):
#         outputs.append(
#             add_residual(
#                 x, brange, residual, residual_scale_factor, scaling_vector
#             ).view_as(x)
#         )
#     return outputs


# class NestedTensorBlock(Block):
#     def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
#         """
#         x_list contains a list of tensors to nest together and run
#         """
#         assert isinstance(self.attn, MemEffAttention)

#         if self.training and self.sample_drop_ratio > 0.0:

#             def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
#                 return self.attn(self.norm1(x), attn_bias=attn_bias)

#             def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
#                 return self.mlp(self.norm2(x))

#             x_list = drop_add_residual_stochastic_depth_list(
#                 x_list,
#                 residual_func=attn_residual_func,
#                 sample_drop_ratio=self.sample_drop_ratio,
#                 scaling_vector=self.ls1.gamma
#                 if isinstance(self.ls1, LayerScale)
#                 else None,
#             )
#             x_list = drop_add_residual_stochastic_depth_list(
#                 x_list,
#                 residual_func=ffn_residual_func,
#                 sample_drop_ratio=self.sample_drop_ratio,
#                 scaling_vector=self.ls2.gamma
#                 if isinstance(self.ls1, LayerScale)
#                 else None,
#             )
#             return x_list
#         else:

#             def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
#                 return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

#             def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
#                 return self.ls2(self.mlp(self.norm2(x)))

#             attn_bias, x = get_attn_bias_and_cat(x_list)
#             x = x + attn_residual_func(x, attn_bias=attn_bias)
#             x = x + ffn_residual_func(x)
#             return attn_bias.split(x)

#     def forward(self, x_or_x_list):
#         if isinstance(x_or_x_list, Tensor):
#             return super().forward(x_or_x_list)
#         elif isinstance(x_or_x_list, list):
#             if not XFORMERS_AVAILABLE:
#                 raise AssertionError("xFormers is required for using nested tensors")
#             return self.forward_nested(x_or_x_list)
#         else:
#             raise AssertionError
