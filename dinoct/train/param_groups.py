from __future__ import annotations

from collections import defaultdict
from typing import List

from torch import nn


def get_vit_lr_decay_rate(
    name: str, lr_decay_rate: float = 1.0, num_layers: int = 12, force_is_backbone: bool = False
) -> float:
    layer_id = num_layers + 1
    if name.startswith("backbone") or force_is_backbone:
        if any(k in name for k in ["pos_embed", "patch_embed", "mask_token", "cls_token", "register_tokens"]):
            layer_id = 0
        else:
            name = name.replace("backbone.", "")
            if "blocks." in name:
                layer_id = int(name.split("blocks.")[1].split(".")[0]) + 1
            else:
                layer_id = num_layers
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_params_groups_with_decay_fsdp(
    model: nn.Module,
    *,
    layerwise_decay: float = 1.0,
    patch_embed_lr_mult: float = 1.0,
) -> List[dict]:
    param_group_names = defaultdict(list)
    param_groups = defaultdict(list)
    decay_names = set()
    no_decay_names = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name_parts = name.split(".")
        if any(nd in name_parts for nd in ["bias", "LayerNorm.weight", "norm", "bn"]):
            no_decay_names.add(name)
            continue
        decay_names.add(name)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = name.startswith("backbone")
        lr_mult = patch_embed_lr_mult if "patch_embed" in name else 1.0
        decay_mult = 0.0 if name in no_decay_names else 1.0
        lr = get_vit_lr_decay_rate(name, layerwise_decay, getattr(model, "n_blocks", 12), is_backbone)
        group_name = f"{'no_' if decay_mult == 0 else ''}decay_{lr:.3f}"
        param_group_names[group_name].append(name)
        param_groups[group_name].append(param)

    groups = []
    for group_name, params in param_groups.items():
        decay = 0.0 if group_name.startswith("no_decay") else 1.0
        lr_mult = float(group_name.split("_")[-1])
        groups.append({"params": params, "weight_decay": decay, "lr_multiplier": lr_mult})
    return groups


def fuse_params_groups(param_groups: List[dict]) -> List[dict]:
    fused = defaultdict(list)
    for group in param_groups:
        key = (group.get("weight_decay", 0.0), group.get("lr_multiplier", 1.0))
        fused[key].extend(group["params"])
    return [{"params": params, "weight_decay": wd, "lr_multiplier": lm} for (wd, lm), params in fused.items()]
