from __future__ import annotations

import math
from typing import List


def cosine_schedule(start: float, end: float, total_steps: int) -> List[float]:
    if total_steps <= 1:
        return [end]
    return [end + 0.5 * (start - end) * (1 + math.cos(math.pi * t / (total_steps - 1))) for t in range(total_steps)]


def linear_warmup_cosine_decay(
    *,
    start: float,
    peak: float,
    end: float,
    warmup_iterations: int,
    total_iterations: int,
    cosine_iterations: int | None = None,
) -> List[float]:
    warmup: list[float] = []
    if warmup_iterations > 0:
        warmup = [start + (peak - start) * (i + 1) / warmup_iterations for i in range(warmup_iterations)]
    remain = max(total_iterations - warmup_iterations, 0)
    cosine_iters = cosine_iterations if cosine_iterations is not None else max(remain, 1)
    cosine: list[float] = []
    if remain > 0:
        cosine = [
            end + 0.5 * (peak - end) * (1 + math.cos(math.pi * t / max(cosine_iters - 1, 1))) for t in range(remain)
        ]
    schedule = warmup + cosine
    # Pad or trim to exact length
    if len(schedule) < total_iterations:
        schedule.extend([schedule[-1]] * (total_iterations - len(schedule)))
    return schedule[:total_iterations]
