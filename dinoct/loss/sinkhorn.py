# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist


def _distributed_logsumexp_all(x: torch.Tensor) -> torch.Tensor:
    max_value = torch.max(x)
    if dist.is_initialized():
        dist.all_reduce(max_value, op=dist.ReduceOp.MAX)

    exp_sum = torch.sum(torch.exp(x - max_value))
    if dist.is_initialized():
        dist.all_reduce(exp_sum, op=dist.ReduceOp.SUM)

    return max_value + torch.log(exp_sum)


def _distributed_logsumexp(x: torch.Tensor, dim: int, *, reduce_across_processes: bool = True) -> torch.Tensor:
    max_value = torch.max(x, dim=dim, keepdim=True).values
    if reduce_across_processes and dist.is_initialized():
        dist.all_reduce(max_value, op=dist.ReduceOp.MAX)

    exp_sum = torch.sum(torch.exp(x - max_value), dim=dim, keepdim=True)
    if reduce_across_processes and dist.is_initialized():
        dist.all_reduce(exp_sum, op=dist.ReduceOp.SUM)

    return max_value + torch.log(exp_sum)


@torch.no_grad()
def sinkhorn_knopp_teacher_log(
    teacher_output: torch.Tensor,
    teacher_temp: float,
    *,
    n_iterations: int = 3,
    n_samples_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    log_q = (teacher_output.float() / teacher_temp).t()

    if n_samples_tensor is None:
        n_samples = log_q.new_tensor([log_q.shape[1]])
    else:
        n_samples = n_samples_tensor.detach().clone().to(device=log_q.device)

    if dist.is_initialized():
        dist.all_reduce(n_samples)

    n_samples = n_samples.to(dtype=log_q.dtype)
    n_prototypes = log_q.new_tensor([log_q.shape[0]])

    log_q = log_q - _distributed_logsumexp_all(log_q)
    log_k = torch.log(n_prototypes)
    log_b = torch.log(n_samples)

    for _ in range(n_iterations):
        log_q = log_q - _distributed_logsumexp(log_q, dim=1, reduce_across_processes=True)
        log_q = log_q - log_k

        # Columns are local samples. In DDP, rank-local column indices do not
        # refer to the same images, so column normalization must not all-reduce.
        log_q = log_q - _distributed_logsumexp(log_q, dim=0, reduce_across_processes=False)
        log_q = log_q - log_b

    log_q = log_q + log_b
    return torch.exp(log_q).t()
