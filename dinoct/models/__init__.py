import logging
from typing import Any

from torch import nn

from .convnext import ConvNeXt, convnext_small, convnext_tiny
from .vision_transformer import DinoVisionTransformer, vit_small

logger = logging.getLogger("dinoct")

_BACKBONES = {
    "small": vit_small,
}

_CONVNEXT_BACKBONES = {
    "convnext_tiny": convnext_tiny,
    "convnext_small": convnext_small,
}


def build_backbone(name: str, *, patch_size: int = 14, **kwargs: Any) -> nn.Module:
    arch = name.replace("vit_", "") if name.startswith("vit_") else name

    vit_builder = _BACKBONES.get(arch)
    if vit_builder is not None:
        model = vit_builder(patch_size=patch_size, **kwargs)
        model.init_weights()
        return model

    convnext_builder = _CONVNEXT_BACKBONES.get(arch)
    if convnext_builder is not None:
        convnext_kwargs: dict[str, Any] = {}
        if "in_chans" in kwargs:
            convnext_kwargs["in_chans"] = kwargs["in_chans"]
        if "drop_path_rate" in kwargs:
            convnext_kwargs["drop_path_rate"] = kwargs["drop_path_rate"]
        if (layerscale := kwargs.get("layerscale_init", None)) is not None:
            convnext_kwargs["layer_scale_init_value"] = layerscale
        model = convnext_builder(patch_size=patch_size, **convnext_kwargs)
        model.init_weights()
        return model

    raise ValueError(f"Unknown backbone '{name}'")


__all__ = [
    "ConvNeXt",
    "DinoVisionTransformer",
    "build_backbone",
    "convnext_tiny",
    "convnext_small",
    "vit_small",
]
