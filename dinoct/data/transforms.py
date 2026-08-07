# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


import torch
from torchvision import transforms


class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=p)


class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)


class Ensure3CH:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: CxHxW
        if x.ndim == 3 and x.shape[0] == 1:
            return x.expand(3, -1, -1)
        return x


class GaussianNoise:
    def __init__(self, std: float = 0.0, p: float = 1.0, clip: bool = True):
        self.std = float(std)
        self.p = float(p)
        self.clip = bool(clip)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.std <= 0.0 or self.p <= 0.0:
            return x
        if self.p < 1.0 and torch.rand(()) >= self.p:
            return x
        y = x.float() + torch.randn_like(x.float()) * self.std
        return y.clamp(0.0, 1.0) if self.clip else y


class PerImageZScore:
    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: CxHxW, float (ToTensor gives [0,1])
        x = x.float()
        mean = x.mean()
        std = x.std(unbiased=False)
        return (x - mean) / (std + self.eps)


# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
