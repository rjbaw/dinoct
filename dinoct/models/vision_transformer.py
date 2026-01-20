from functools import partial
import logging
from collections.abc import Sequence
from typing import Any, Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import trunc_normal_

from ..layers import (
    LayerScale,
    Mlp,
    PatchEmbed,
    RMSNorm,
    RopePositionEmbedding,
    SelfAttentionBlock,
    SwiGLUFFN,
)
from ..utils import named_apply

logger = logging.getLogger("dinoct")

ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
}

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def init_weights_vit(module: nn.Module, name: str = ""):
    _ = name
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if hasattr(module, "bias_mask") and module.bias_mask is not None:
            o = module.out_features
            module.bias_mask.fill_(1)
            module.bias_mask[o // 3 : 2 * o // 3].fill_(0)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        module.reset_parameters()


# class BlockChunk(nn.ModuleList):
#     def forward(self, x):
#         for b in self:
#             x = b(x)
#         return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        layerscale_init: float | None = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        drop_path_rate: float = 0.0,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        # drop_path_uniform=False,
        # init_values=None,  # for layerscale: None or 0 => no layerscale
        # embed_layer=PatchEmbed,
        # act_layer=nn.GELU,
        # block_fn=Block,
        # block_chunks=1,
        # num_register_tokens=0,
        # interpolate_antialias=False,
        # interpolate_offset=0.1,
        device: Any | None = None,
        **ignored_kwargs,
    ):
        super().__init__()
        # norm_layer = partial(nn.LayerNorm, eps=1e-6)
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs
        norm_layer_cls = norm_layer_dict[norm_layer]

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        # self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        # self.num_register_tokens = num_register_tokens
        # self.interpolate_antialias = interpolate_antialias
        # self.interpolate_offset = interpolate_offset

        # self.patch_embed = embed_layer(
        #     img_size=img_size,
        #     patch_size=patch_size,
        #     in_chans=in_chans,
        #     embed_dim=embed_dim,
        # )
        # num_patches = self.patch_embed.num_patches

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.pos_embed = nn.Parameter(
        #     torch.zeros(1, num_patches + self.num_tokens, embed_dim)
        # )
        # assert num_register_tokens >= 0
        # self.register_tokens = (
        #     nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
        #     if num_register_tokens
        #     else None
        # )

        # if drop_path_uniform is True:
        #     dpr = [drop_path_rate] * depth
        # else:
        #     dpr = np.linspace(
        #         0, drop_path_rate, depth
        #     ).tolist()  # stochastic depth decay rule

        # if ffn_layer == "mlp":
        #     logger.info("using MLP layer as FFN")
        #     ffn_layer = Mlp
        # elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
        #     logger.info("using SwiGLU layer as FFN")
        #     ffn_layer = SwiGLUFFNFused
        # elif ffn_layer == "identity":
        #     logger.info("using Identity layer as FFN")

        #     def f(*args, **kwargs):
        #         return nn.Identity()

        #     ffn_layer = f
        # else:
        #     raise NotImplementedError

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim, device=device))
        logger.info(f"using base={pos_embed_rope_base} for rope new")
        logger.info(f"using min_period={pos_embed_rope_min_period} for rope new")
        logger.info(f"using max_period={pos_embed_rope_max_period} for rope new")
        logger.info(f"using normalize_coords={pos_embed_rope_normalize_coords} for rope new")
        logger.info(f"using shift_coords={pos_embed_rope_shift_coords} for rope new")
        logger.info(f"using rescale_coords={pos_embed_rope_rescale_coords} for rope new")
        logger.info(f"using jitter_coords={pos_embed_rope_jitter_coords} for rope new")
        logger.info(f"using dtype={pos_embed_rope_dtype} for rope new")
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )
        if isinstance(ffn_layer, bool):
            # Backward compatibility: treat legacy boolean as mlp
            ffn_layer = "mlp"
        logger.info(f"using {ffn_layer} layer as FFN")
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth

        blocks_list = [
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio_sequence[i],
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
                device=device,
            )
            for i in range(depth)
        ]

        self.chunked_blocks = True
        self.blocks = nn.ModuleList(blocks_list)

        # if block_chunks > 0:
        #     chunked_blocks = []
        #     chunksize = depth // block_chunks
        #     for i in range(0, depth, chunksize):
        #         # this is to keep the block index consistent if we chunk the block list
        #         chunked_blocks.append(
        #             [nn.Identity()] * i + blocks_list[i : i + chunksize]
        #         )
        #     self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        # else:
        #     self.chunked_blocks = False
        #     self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer_cls(embed_dim)
        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            self.cls_norm = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            self.local_cls_norm = norm_layer_cls(embed_dim)
        else:
            self.local_cls_norm = None

        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        # patch_pos_embed = nn.functional.interpolate(
        #     patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
        #     mode="bicubic",
        #     antialias=self.interpolate_antialias,
        #     **kwargs,
        # )
        # assert (w0, h0) == patch_pos_embed.shape[-2:]
        # patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        # return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(
        #     previous_dtype
        # )

    def init_weights(self):
        self.rope_embed._init_weights()
        # trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

    def prepare_tokens_with_masks(self, x: torch.Tensor, masks=None) -> tuple[Tensor, tuple[int, int]]:
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            # cls_token = self.cls_token + 0 * self.mask_token
            cls_token = self.cls_token

        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(
                1,
                0,
                cls_token.shape[-1],
                dtype=cls_token.dtype,
                device=cls_token.device,
            )

        x = torch.cat(
            [
                cls_token.expand(B, -1, -1),
                storage_tokens.expand(B, -1, -1),
                x,
            ],
            dim=1,
        )
        return x, (H, W)

    # def prepare_tokens_with_masks(self, x, masks=None):
    #     B, nc, w, h = x.shape
    #     x = self.patch_embed(x)
    #     if masks is not None:
    #         x = torch.where(
    #             masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x
    #         )

    #     x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    #     x = x + self.interpolate_pos_encoding(x, w, h)

    #     if self.register_tokens is not None:
    #         x = torch.cat(
    #             (
    #                 x[:, :1],
    #                 self.register_tokens.expand(x.shape[0], -1, -1),
    #                 x[:, 1:],
    #             ),
    #             dim=1,
    #         )

    #     return x

    # def interpolate_pos_encoding(self, x, w, h):
    #     previous_dtype = x.dtype
    #     npatch = x.shape[1] - 1
    #     N = self.pos_embed.shape[1] - 1
    #     if npatch == N and w == h:
    #         return self.pos_embed
    #     pos_embed = self.pos_embed.float()
    #     class_pos_embed = pos_embed[:, 0]
    #     patch_pos_embed = pos_embed[:, 1:]
    #     dim = x.shape[-1]
    #     w0 = w // self.patch_size
    #     h0 = h // self.patch_size
    #     M = int(math.sqrt(N))  # Recover the number of patches in each dimension
    #     assert N == M * M
    #     kwargs = {}
    #     if self.interpolate_offset:
    #         # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
    #         # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
    #         sx = float(w0 + self.interpolate_offset) / M
    #         sy = float(h0 + self.interpolate_offset) / M
    #         kwargs["scale_factor"] = (sx, sy)

    def forward_features_list(self, x_list: list[Tensor], masks_list: list[Tensor | None]) -> list[dict[str, Tensor]]:
        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)
        for _, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = [self.rope_embed(H=H, W=W) for (H, W) in rope]
            else:
                rope_sincos = [None for r in rope]
            x = blk(x, rope_sincos)
        all_x = x
        output = []

        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                    x_norm_cls_reg = self.local_cls_norm(x[:, : self.n_storage_tokens + 1])
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(x[:, : self.n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = self.norm(x[:, : self.n_storage_tokens + 1 :])
                x_norm_patch = self.norm(x[:, self.n_storage_tokens + 1 :])
            else:
                x_norm = self.norm(x)
                x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]

            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_norm_regtokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(
        self, x: Tensor | list[Tensor], masks: Tensor | list[Tensor | None] | None = None
    ) -> list[dict[str, Tensor]]:
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])
        if masks is None:
            return self.forward_features_list(x, [None for _ in x])
        if isinstance(masks, list):
            return self.forward_features_list(x, masks)
        raise TypeError("When `x` is a list, `masks` must be a list[Tensor|None] or None.")

        # x = self.prepare_tokens_with_masks(x, masks)

        # for blk in self.blocks:
        #     x = blk(x)

        # x_norm = self.norm(x)
        # return {
        #     "x_norm_clstoken": x_norm[:, 0],
        #     "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
        #     "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
        #     "x_prenorm": x,
        #     "masks": masks,
        # }

    def _get_intermediate_layers_not_chunked(self, x: Tensor, n: int = 1) -> list[Tensor]:
        x, (H, W) = self.prepare_tokens_with_masks(x)

        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            x = blk(x, rope_sincos)[0]

            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    # def _get_intermediate_layers_chunked(self, x, n=1):
    #     x = self.prepare_tokens_with_masks(x)
    #     output, i, total_block_len = [], 0, len(self.blocks[-1])
    #     # If n is an int, take the n last blocks. If it's a list, take them
    #     blocks_to_take = (
    #         range(total_block_len - n, total_block_len) if isinstance(n, int) else n
    #     )
    #     for block_chunk in self.blocks:
    #         for blk in block_chunk[i:]:  # Passing the nn.Identity()
    #             x = blk(x)
    #             if i in blocks_to_take:
    #                 output.append(x)
    #             i += 1
    #     assert len(output) == len(blocks_to_take), (
    #         f"only {len(output)} / {len(blocks_to_take)} blocks found"
    #     )
    #     return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: int | Sequence = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm=True,
    ) -> tuple[Tensor | tuple[Tensor, ...], ...]:
        outputs = self._get_intermediate_layers_not_chunked(x, n)

        # if self.chunked_blocks:
        #     outputs = self._get_intermediate_layers_chunked(x, n)
        # else:
        #     outputs = self._get_intermediate_layers_not_chunked(x, n)

        if norm:
            # outputs = [self.norm(out) for out in outputs]
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1 :])
                    outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=1))
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]

        if reshape:
            B, _, h, w = x.shape
            outputs = [
                out.reshape(B, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        elif return_class_token and return_extra_tokens:
            return tuple(zip(outputs, class_tokens, extra_tokens))
        raise NotImplementedError

    def forward(self, *args, is_training: bool = False, **kwargs) -> list[dict[str, Tensor]] | Tensor:
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret[0]["x_norm_clstoken"])


# def init_weights_vit_timm(module: nn.Module, name: str = ""):
#     """ViT weight initialization, original timm impl (for reproducibility)"""
#     if isinstance(module, nn.Linear):
#         trunc_normal_(module.weight, std=0.02)
#         if module.bias is not None:
#             nn.init.zeros_(module.bias)


def vit_small(patch_size: int = 16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        # mlp_ratio=4,
        # block_fn=partial(Block, attn_class=MemEffAttention),
        # num_register_tokens=num_register_tokens,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_base(patch_size: int = 16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        # mlp_ratio=4,
        # block_fn=partial(Block, attn_class=MemEffAttention),
        # num_register_tokens=num_register_tokens,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_large(patch_size: int = 16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        # mlp_ratio=4,
        # block_fn=partial(Block, attn_class=MemEffAttention),
        # num_register_tokens=num_register_tokens,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_so400m(patch_size: int = 16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1152,
        depth=27,
        num_heads=18,
        ffn_ratio=3.777777778,
        **kwargs,
    )
    return model


def vit_huge2(patch_size: int = 16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=20,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_giant2(patch_size: int = 16, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        # mlp_ratio=4,
        # block_fn=partial(Block, attn_class=MemEffAttention),
        # num_register_tokens=num_register_tokens,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_7b(patch_size: int = 16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=4096,
        depth=40,
        num_heads=32,
        ffn_ratio=3,
        **kwargs,
    )
    return model
