import logging
import os
import random
import subprocess
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


def named_replace(
    fn: Callable,
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    if not depth_first and include_root:
        module = fn(module=module, name=name)
    for child_name_o, child_module in list(module.named_children()):
        child_name = ".".join((name, child_name_o)) if name else child_name_o
        new_child = named_replace(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
        setattr(module, child_name_o, new_child)

    if depth_first and include_root:
        module = fn(module=module, name=name)

    return module


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


# def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
#     if urlparse(pretrained_weights).scheme:  # If it looks like an URL
#         state_dict = torch.hub.load_state_dict_from_url(
#             pretrained_weights, map_location="cpu"
#         )
#     else:
#         state_dict = torch.load(pretrained_weights, map_location="cpu")
#     if checkpoint_key is not None and checkpoint_key in state_dict:
#         logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
#         state_dict = state_dict[checkpoint_key]
#     # remove `module.` prefix
#     state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
#     # remove `backbone.` prefix induced by multicrop wrapper
#     state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
#     msg = model.load_state_dict(state_dict, strict=False)
#     logger.info(
#         "Pretrained weights found at {} and loaded with msg: {}".format(
#             pretrained_weights, msg
#         )
#     )


def fix_random_seeds(seed: int = 31):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_sha() -> str:
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommitted changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


# class CosineScheduler(object):
#     def __init__(
#         self,
#         base_value,
#         final_value,
#         total_iters,
#         warmup_iters=0,
#         start_warmup_value=0,
#         freeze_iters=0,
#     ):
#         super().__init__()
#         self.final_value = final_value
#         self.total_iters = total_iters

#         freeze_schedule = np.zeros((freeze_iters))

#         warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

#         iters = np.arange(total_iters - warmup_iters - freeze_iters)
#         schedule = final_value + 0.5 * (base_value - final_value) * (
#             1 + np.cos(np.pi * iters / len(iters))
#         )
#         self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

#         assert len(self.schedule) == self.total_iters

#     def __getitem__(self, it):
#         if it >= self.total_iters:
#             return self.final_value
#         else:
#             return self.schedule[it]


def get_conda_env() -> tuple[str | None, str | None]:
    conda_env_name = os.environ.get("CONDA_DEFAULT_ENV")
    conda_env_path = os.environ.get("CONDA_PREFIX")
    return conda_env_name, conda_env_path


def has_batchnorms(model: nn.Module) -> bool:
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for _, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False
