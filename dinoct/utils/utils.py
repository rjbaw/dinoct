# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with the terms
# of the DINOv3 License Agreement, a copy of which is provided in
# LICENSE-DINOV3.md in the root directory of this source tree.

import logging
import random
# from urllib.parse import urlparse

from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor, nn


logger = logging.getLogger("dinoct")


def cat_keep_shapes(x_list: list[Tensor]) -> tuple[Tensor, list[torch.Size], list[int]]:
    shapes = [x.shape for x in x_list]
    num_tokens = [x.select(dim=-1, index=0).numel() for x in x_list]
    flattened = torch.cat([x.flatten(0, -2) for x in x_list])
    return flattened, shapes, num_tokens


def uncat_with_shapes(flattened: Tensor, shapes: list[torch.Size], num_tokens: list[int]) -> list[Tensor]:
    outputs_splitted = torch.split_with_sizes(flattened, num_tokens, dim=0)
    shapes_adjusted = [shape[:-1] + torch.Size([flattened.shape[-1]]) for shape in shapes]
    outputs_reshaped = [o.reshape(shape) for (o, shape) in zip(outputs_splitted, shapes_adjusted)]
    return outputs_reshaped


def named_apply(
    fn: Callable,
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name_o, child_module in list(module.named_children()):
        child_name = ".".join((name, child_name_o)) if name else child_name_o
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )

    if depth_first and include_root:
        fn(module=module, name=name)

    return module


def fix_random_seeds(seed: int = 31):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


