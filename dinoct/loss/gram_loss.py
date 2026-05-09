import torch
import torch.nn as nn
import torch.nn.functional as F


class GramLoss(nn.Module):
    """MSE between student and teacher patch-token Gram matrices."""

    def __init__(
        self,
        *,
        apply_norm: bool = True,
        remove_neg: bool = False,
        remove_only_teacher_neg: bool = False,
    ) -> None:
        super().__init__()
        if remove_neg and remove_only_teacher_neg:
            raise ValueError("remove_neg and remove_only_teacher_neg are mutually exclusive")
        self.apply_norm = apply_norm
        self.remove_neg = remove_neg
        self.remove_only_teacher_neg = remove_only_teacher_neg
        self.mse_loss = nn.MSELoss()

    def forward(self, output_feats: torch.Tensor, target_feats: torch.Tensor, *, img_level: bool = True) -> torch.Tensor:
        if img_level and (output_feats.ndim != 3 or target_feats.ndim != 3):
            raise ValueError("img_level=True expects output_feats and target_feats with shape (B, N, D)")

        output_feats = output_feats.float()
        target_feats = target_feats.float()
        if self.apply_norm:
            output_feats = F.normalize(output_feats, dim=-1)
            target_feats = F.normalize(target_feats, dim=-1)
        if not img_level and output_feats.ndim == 3:
            output_feats = output_feats.flatten(0, 1)
        if not img_level and target_feats.ndim == 3:
            target_feats = target_feats.flatten(0, 1)

        student_sim = torch.matmul(output_feats, output_feats.transpose(-1, -2))
        target_sim = torch.matmul(target_feats, target_feats.transpose(-1, -2))
        if self.remove_neg:
            student_sim = student_sim.masked_fill(student_sim < 0, 0.0)
            target_sim = target_sim.masked_fill(target_sim < 0, 0.0)
        elif self.remove_only_teacher_neg:
            student_sim = student_sim.masked_fill((student_sim < 0) & (target_sim < 0), 0.0)
            target_sim = target_sim.masked_fill(target_sim < 0, 0.0)
        return self.mse_loss(student_sim, target_sim)
