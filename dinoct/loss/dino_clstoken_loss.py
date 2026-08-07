# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from .sinkhorn import sinkhorn_knopp_teacher_log


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        student_temp=0.1,
        center_momentum=0.9,
        sinkhorn_queue_size: int = 0,
        sinkhorn_queue_start_iter: int = 0,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None

        # Sinkhorn FIFO queue (DINO cls OT only)
        self.sinkhorn_queue_size = int(sinkhorn_queue_size)
        self.sinkhorn_queue_start_iter = int(sinkhorn_queue_start_iter)
        if self.sinkhorn_queue_size > 0:
            self.register_buffer("sk_queue", torch.zeros(self.sinkhorn_queue_size, out_dim))
            self.register_buffer("sk_queue_ptr", torch.zeros(1, dtype=torch.long))
            self.register_buffer("sk_queue_fill", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp):
        self.apply_center_update()
        # teacher centering and sharpening
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3, iteration: int = 0):
        use_queue = (
            self.sinkhorn_queue_size > 0
            and iteration >= self.sinkhorn_queue_start_iter
            and int(self.sk_queue_fill.item()) >= self.sinkhorn_queue_size
        )
        if use_queue:
            b_ot = teacher_output.shape[0]
            # cat current batch with queued logits; handle dtype mismatch (queue is fp32, input may be fp16)
            ot_input = torch.cat([teacher_output, self.sk_queue.to(teacher_output.dtype)], dim=0)
            full_targets = sinkhorn_knopp_teacher_log(
                ot_input,
                teacher_temp,
                n_iterations=n_iterations,
            )
            targets = full_targets[:b_ot]
        else:
            targets = sinkhorn_knopp_teacher_log(
                teacher_output,
                teacher_temp,
                n_iterations=n_iterations,
            )

        # Always enqueue after start_iter (regardless of fill state) so queue fills up
        if self.sinkhorn_queue_size > 0 and iteration >= self.sinkhorn_queue_start_iter:
            self._enqueue(teacher_output)

        return targets

    @torch.no_grad()
    def _enqueue(self, teacher_output: torch.Tensor) -> None:
        """FIFO enqueue of detached teacher logits into sk_queue."""
        logits = teacher_output.detach().to(self.sk_queue.dtype)
        batch = logits.shape[0]
        q_size = self.sinkhorn_queue_size
        ptr = int(self.sk_queue_ptr.item())

        # Wrap-safe: write in two segments if batch wraps around end of queue
        space = q_size - ptr
        if batch <= space:
            self.sk_queue[ptr : ptr + batch] = logits
        else:
            self.sk_queue[ptr:] = logits[:space]
            self.sk_queue[: batch - space] = logits[space:]

        new_ptr = (ptr + batch) % q_size
        self.sk_queue_ptr[0] = new_ptr

        filled = min(int(self.sk_queue_fill.item()) + batch, q_size)
        self.sk_queue_fill[0] = filled

    def forward(
        self,
        student_output_list,
        teacher_out_softmaxed_centered_list,
        *,
        ignore_diagonal: bool = False,
    ):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_logits = torch.stack(student_output_list, dim=0).float()
        teacher_probs = torch.stack(teacher_out_softmaxed_centered_list, dim=0).float()

        student_logp = F.log_softmax(student_logits / self.student_temp, dim=-1)
        student_crops, batch_size, _ = student_logp.shape
        teacher_crops = teacher_probs.shape[0]

        if not ignore_diagonal:
            return -torch.einsum("sbk,tbk->", student_logp, teacher_probs) / (
                batch_size * student_crops * teacher_crops
            )

        pair_loss = -torch.einsum("sbk,tbk->st", student_logp, teacher_probs)
        min_crops = min(student_crops, teacher_crops)
        valid_pairs = student_crops * teacher_crops - min_crops
        if valid_pairs <= 0:
            raise ValueError("ignore_diagonal=True requires at least one non-diagonal teacher/student pair.")
        keep = torch.ones((student_crops, teacher_crops), dtype=torch.bool, device=pair_loss.device)
        keep.diagonal()[:min_crops] = False
        return pair_loss[keep].sum() / (batch_size * valid_pairs)

    @torch.no_grad()
    def update_center(self, teacher_output):
        self.reduce_center_update(teacher_output)

    @torch.no_grad()
    def reduce_center_update(self, teacher_output):
        self.updated = False
        self.len_teacher_output = len(teacher_output)
        self.async_batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_output * world_size)

            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)

            self.updated = True
