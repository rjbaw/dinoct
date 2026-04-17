from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - used when running pure-numpy classical baselines
    torch = None  # type: ignore[assignment]


DEFAULT_ACC_TOLERANCES = (2.0, 4.0)


def metric_name_for_tolerance(tau: float) -> str:
    tau_f = float(tau)
    tau_i = int(round(tau_f))
    if abs(tau_f - tau_i) < 1e-6:
        return f"acc_{tau_i}px"
    return f"acc_{str(tau_f).replace('.', 'p')}px"


def curve_metrics_batch(
    y_pred: "torch.Tensor | np.ndarray",
    y_true: "torch.Tensor | np.ndarray",
    *,
    acc_tolerances: Iterable[float] = DEFAULT_ACC_TOLERANCES,
    spike_kappa: float | None = None,
) -> dict[str, "torch.Tensor | np.ndarray"]:
    if isinstance(y_pred, np.ndarray) or isinstance(y_true, np.ndarray):
        y_pred_f = np.asarray(y_pred, dtype=np.float32)
        y_true_f = np.asarray(y_true, dtype=np.float32)
        abs_err = np.abs(y_pred_f - y_true_f)
        bias = (y_pred_f - y_true_f).mean(axis=1)

        metrics_np: dict[str, np.ndarray] = {
            "mae_px": abs_err.mean(axis=1),
            "p95_px": np.quantile(abs_err, 0.95, axis=1),
            "bias_px": bias,
            "abs_bias_px": np.abs(bias),
        }

        for tau in acc_tolerances:
            metrics_np[metric_name_for_tolerance(float(tau))] = (abs_err <= float(tau)).astype(np.float32).mean(axis=1)

        if spike_kappa is not None:
            if y_pred_f.shape[1] < 3:
                metrics_np["spike_rate"] = np.zeros(y_pred_f.shape[0], dtype=np.float32)
            else:
                d2 = y_pred_f[:, 2:] - 2.0 * y_pred_f[:, 1:-1] + y_pred_f[:, :-2]
                metrics_np["spike_rate"] = (np.abs(d2) > float(spike_kappa)).astype(np.float32).mean(axis=1)

        return metrics_np

    if torch is None:
        raise ModuleNotFoundError("torch is required for tensor inputs to curve_metrics_batch.")

    y_pred_f = y_pred.float()
    y_true_f = y_true.float()
    abs_err = (y_pred_f - y_true_f).abs()
    bias = (y_pred_f - y_true_f).mean(dim=1)

    metrics: dict[str, torch.Tensor] = {
        "mae_px": abs_err.mean(dim=1),
        "p95_px": torch.quantile(abs_err, 0.95, dim=1),
        "bias_px": bias,
        "abs_bias_px": bias.abs(),
    }

    for tau in acc_tolerances:
        metrics[metric_name_for_tolerance(float(tau))] = (abs_err <= float(tau)).float().mean(dim=1)

    if spike_kappa is not None:
        if y_pred_f.shape[1] < 3:
            metrics["spike_rate"] = torch.zeros(y_pred_f.shape[0], device=y_pred_f.device, dtype=y_pred_f.dtype)
        else:
            d2 = y_pred_f[:, 2:] - 2.0 * y_pred_f[:, 1:-1] + y_pred_f[:, :-2]
            metrics["spike_rate"] = (d2.abs() > float(spike_kappa)).float().mean(dim=1)

    return metrics


def estimate_spike_kappa_from_curves(
    curves: "torch.Tensor | np.ndarray | Iterable[np.ndarray | torch.Tensor]",
    *,
    quantile: float = 0.99,
) -> float:
    q = float(quantile)
    if not (0.0 < q <= 1.0):
        raise ValueError(f"Expected 0 < quantile <= 1, got {quantile!r}")

    if isinstance(curves, torch.Tensor):
        curves_np = curves.detach().cpu().float().numpy()
    elif isinstance(curves, np.ndarray):
        curves_np = np.asarray(curves, dtype=np.float32)
    else:
        items = []
        for value in curves:
            if isinstance(value, torch.Tensor):
                items.append(value.detach().cpu().float().numpy())
            else:
                items.append(np.asarray(value, dtype=np.float32))
        if not items:
            raise ValueError("Cannot estimate spike kappa from an empty curve collection.")
        curves_np = np.stack(items, axis=0)

    curves_np = np.asarray(curves_np, dtype=np.float32)
    if curves_np.ndim == 1:
        curves_np = curves_np[None, :]
    if curves_np.ndim != 2:
        raise ValueError(f"Expected curves with shape (N, W), got {tuple(curves_np.shape)!r}")
    if curves_np.shape[1] < 3:
        raise ValueError("Need curves with width >= 3 to estimate spike kappa.")

    d2 = curves_np[:, 2:] - 2.0 * curves_np[:, 1:-1] + curves_np[:, :-2]
    abs_d2 = np.abs(d2.reshape(-1))
    finite = abs_d2[np.isfinite(abs_d2)]
    if finite.size == 0:
        raise ValueError("Could not estimate spike kappa: no finite second-order differences found.")
    return float(np.quantile(finite, q))


def average_metric_rows(
    rows: list[dict[str, Any]],
    *,
    group_key: str,
    metric_names: Iterable[str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_value = str(row.get(group_key, "") or "")
        if group_value:
            grouped[group_value].append(row)

    out: list[dict[str, Any]] = []
    metric_names_t = tuple(metric_names)
    for group_value in sorted(grouped):
        group_rows = grouped[group_value]
        agg: dict[str, Any] = {group_key: group_value, "num_samples": len(group_rows)}
        for carry_key in ("split", "acquisition_mode"):
            values = {str(row.get(carry_key, "") or "") for row in group_rows if row.get(carry_key, "")}
            if len(values) == 1:
                agg[carry_key] = next(iter(values))
        for metric_name in metric_names_t:
            values = np.asarray([float(row[metric_name]) for row in group_rows if metric_name in row], dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size > 0:
                agg[metric_name] = float(values.mean())
        out.append(agg)
    return out


def _summarize_scalar(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0.0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "q1": float("nan"),
            "q3": float("nan"),
        }
    return {
        "count": float(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1) if arr.size > 1 else 0.0),
        "median": float(np.median(arr)),
        "q1": float(np.quantile(arr, 0.25)),
        "q3": float(np.quantile(arr, 0.75)),
    }


def summarize_metric_rows(rows: list[dict[str, Any]], metric_names: Iterable[str]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        values = [float(row[metric_name]) for row in rows if metric_name in row]
        summary[str(metric_name)] = _summarize_scalar(values)
    return summary
