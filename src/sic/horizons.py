"""Evaluation time-horizon helpers (Concept A — independent of prediction dumps)."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def unique_sorted_times(durations: np.ndarray) -> np.ndarray:
    """Unique sorted follow-up times (e.g. unique test times for IBS)."""
    return np.unique(np.asarray(durations, dtype=np.float64))


def make_ibs_q90_grid(
    train_durations: np.ndarray,
    train_events: np.ndarray,
    *,
    percentile: float = 90.0,
    n_points: int = 100,
    max_time: Optional[float] = None,
) -> np.ndarray:
    """Linspace from min train event time to the train follow-up percentile.

    ``tau`` uses all train durations (events + censored). ``t_min`` uses event
    times when available, otherwise ``min(train_durations)``.
    """
    train_T = np.asarray(train_durations, dtype=np.float64)
    train_E = np.asarray(train_events)
    if train_T.size == 0:
        return np.asarray([], dtype=np.float64)

    event_times = train_T[train_E.astype(bool)]
    t_min = float(event_times.min()) if event_times.size > 0 else float(train_T.min())
    tau = float(np.percentile(train_T, percentile))
    if max_time is not None:
        tau = min(tau, float(max_time))

    if not np.isfinite(t_min) or not np.isfinite(tau) or tau <= t_min:
        return np.asarray([t_min], dtype=np.float64) if np.isfinite(t_min) else np.asarray([], dtype=np.float64)

    return np.linspace(t_min, tau, int(n_points), dtype=np.float64)


def make_pointwise_times(
    train_durations: np.ndarray,
    train_events: np.ndarray,
    percentiles: Sequence[float] = (0.25, 0.5, 0.75),
) -> np.ndarray:
    """Absolute times at train event-time quantiles (fallback: all train times)."""
    train_T = np.asarray(train_durations, dtype=np.float64)
    train_E = np.asarray(train_events)
    if train_T.size == 0:
        return np.asarray([], dtype=np.float64)

    event_times = train_T[train_E.astype(bool)]
    base = event_times if event_times.size >= len(percentiles) else train_T
    qs = [float(p) * 100.0 for p in percentiles]
    return np.asarray(np.percentile(base, qs), dtype=np.float64)


def clip_times_to_interval(
    times: np.ndarray,
    lower: Optional[float],
    upper: Optional[float],
    *,
    strict_upper: bool = False,
) -> np.ndarray:
    """Keep times inside ``[lower, upper]`` (or ``[lower, upper)`` if strict_upper)."""
    t = np.asarray(times, dtype=np.float64)
    if t.size == 0:
        return t
    mask = np.ones(t.shape, dtype=bool)
    if lower is not None:
        mask &= t >= float(lower)
    if upper is not None:
        if strict_upper:
            mask &= t < float(upper)
        else:
            mask &= t <= float(upper)
    return t[mask]
