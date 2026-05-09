from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from torch import nn


def _without_backbone_prefix(name: str) -> str:
    for prefix in ("module.backbone.", "backbone.", "module."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def get_vit_lr_decay_rate(
    name: str,
    lr_decay_rate: float = 1.0,
    num_layers: int = 12,
    force_is_backbone: bool = False,
    chunked_blocks: bool = False,
    blocks_per_chunk: int | None = None,
    stage_block_offsets: Sequence[int] | None = None,
) -> float:
    layer_id = num_layers + 1
    if name.startswith("backbone") or name.startswith("module.backbone") or force_is_backbone:
        name = _without_backbone_prefix(name)
        if any(k in name for k in ["pos_embed", "patch_embed", "mask_token", "cls_token", "storage_tokens"]):
            layer_id = 0
        elif name.startswith("blocks."):
            parts = name.split(".")
            if chunked_blocks:
                if blocks_per_chunk is None:
                    raise ValueError("blocks_per_chunk is required when chunked_blocks=True")
                layer_idx = int(parts[1]) * int(blocks_per_chunk) + int(parts[2])
            else:
                layer_idx = int(parts[1])
            layer_id = min(layer_idx + 1, num_layers)
        elif name.startswith("stages.") and stage_block_offsets is not None:
            parts = name.split(".")
            stage_idx = int(parts[1])
            block_idx = int(parts[2])
            layer_id = min(int(stage_block_offsets[stage_idx]) + block_idx + 1, num_layers)
        else:
            layer_id = num_layers
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_params_groups_with_decay_fsdp(
    model: nn.Module,
    *,
    layerwise_decay: float = 1.0,
    patch_embed_lr_mult: float = 1.0,
) -> list[dict]:
    n_blocks = int(getattr(model, "n_blocks", 0) or 0)
    force_is_backbone = n_blocks > 0
    chunked_blocks = bool(getattr(model, "chunked_blocks", False))
    blocks_per_chunk = None
    if chunked_blocks:
        blocks = getattr(model, "blocks", None)
        n_chunks = len(blocks) if blocks is not None else 0
        if n_chunks <= 0 or n_blocks % n_chunks != 0:
            raise ValueError("chunked backbone has incompatible n_blocks/blocks metadata")
        blocks_per_chunk = n_blocks // n_chunks

    stage_block_offsets = None
    stages = getattr(model, "stages", None)
    if stages is not None:
        stage_block_offsets = []
        offset = 0
        for stage in stages:
            stage_block_offsets.append(offset)
            offset += len(stage)

    param_groups = defaultdict(list)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lr_mult = get_vit_lr_decay_rate(
            name,
            lr_decay_rate=layerwise_decay,
            num_layers=n_blocks or 12,
            force_is_backbone=force_is_backbone,
            chunked_blocks=chunked_blocks,
            blocks_per_chunk=blocks_per_chunk,
            stage_block_offsets=stage_block_offsets,
        )
        if "patch_embed" in name:
            lr_mult *= patch_embed_lr_mult

        no_decay = (
            param.ndim <= 1
            or name.endswith(".bias")
            or "norm" in name.lower()
            or "gamma" in name
            or "cls_token" in name
            or "mask_token" in name
            or "storage_tokens" in name
        )
        group_name = (0.0 if no_decay else 1.0, float(lr_mult))
        param_groups[group_name].append(param)

    groups = []
    for (weight_decay, lr_mult), params in param_groups.items():
        groups.append({"params": params, "weight_decay": weight_decay, "lr_multiplier": lr_mult})
    return groups


def fuse_params_groups(param_groups: list[dict]) -> list[dict]:
    fused = defaultdict(list)
    for group in param_groups:
        key = (group.get("weight_decay", 0.0), group.get("lr_multiplier", 1.0))
        fused[key].extend(group["params"])
    return [{"params": params, "weight_decay": wd, "lr_multiplier": lm} for (wd, lm), params in fused.items()]
