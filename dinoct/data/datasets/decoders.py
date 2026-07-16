# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with the terms
# of the DINOv3 License Agreement, a copy of which is provided in
# LICENSE-DINOV3.md in the root directory of this source tree.

from io import BytesIO
from typing import Any

from PIL import Image


class Decoder:
    def decode(self) -> Any:
        raise NotImplementedError


class DenseTargetDecoder(Decoder):
    def __init__(self, image_data: bytes) -> None:
        self._image_data = image_data

    def decode(self) -> Image.Image:
        f = BytesIO(self._image_data)
        return Image.open(f)


class ImageDataDecoder(Decoder):
    def __init__(self, image_data: bytes) -> None:
        self._image_data = image_data

    def decode(self) -> Image.Image:
        f = BytesIO(self._image_data)
        return Image.open(f).convert(mode="RGB")


class TargetDecoder(Decoder):
    def __init__(self, target: Any):
        self._target = target

    def decode(self) -> Any:
        return self._target


__all__ = ["Decoder", "DenseTargetDecoder", "ImageDataDecoder", "TargetDecoder"]
