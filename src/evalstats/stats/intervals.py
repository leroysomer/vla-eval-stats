"""Intervals for a success rate, a gap between two policies, or any statistic of the grid."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from statsmodels.stats.proportion import proportion_confint


def clopper_pearson(n_success, n: int, level: float = 0.95):
    """Exact binomial interval, as if the `n` episodes were independent with one probability."""
    return proportion_confint(n_success, n, alpha=1 - level, method="beta")


def bootstrap_replicates(
    statistic: Callable[..., np.ndarray],
    *arrays: np.ndarray,
    n_boot: int = 10_000,
    rng: np.random.Generator,
    batch: int = 500,
) -> np.ndarray:
    """Replicates of `statistic` when initial states are resampled within each task.

    Each array has shape (tasks, states, seeds). A replicate draws as many states as the task
    has, with replacement, and keeps the seeds of a drawn state together. All arrays get the
    same draw, so two policies stay paired. `statistic` receives the resampled arrays with a
    leading axis of replicates and returns one value, or one vector, per replicate.
    """
    n_tasks, n_init = arrays[0].shape[:2]
    tasks = np.arange(n_tasks)[None, :, None]
    out = []
    for start in range(0, n_boot, batch):
        states = rng.integers(n_init, size=(min(batch, n_boot - start), n_tasks, n_init))
        out.append(np.asarray(statistic(*(a[tasks, states] for a in arrays))))
    return np.concatenate(out)


def percentile_interval(replicates: np.ndarray, level: float = 0.95) -> np.ndarray:
    """Lower and upper percentile bounds over the first axis, shape (2, ...)."""
    tail = (1 - level) / 2
    return np.quantile(replicates, [tail, 1 - tail], axis=0)


def _mean(y: np.ndarray) -> np.ndarray:
    return y.mean(axis=(-3, -2, -1))


def bootstrap_rate(y: np.ndarray, level: float = 0.95, **kwargs) -> np.ndarray:
    """Cluster-aware interval for the success rate of one policy on a (task, state, seed) grid."""
    return percentile_interval(bootstrap_replicates(_mean, y, **kwargs), level)


def bootstrap_gap(y0: np.ndarray, y1: np.ndarray, level: float = 0.95, **kwargs) -> np.ndarray:
    """Cluster-aware interval for the gap p1 - p0, from the differences on each key."""
    diff = np.asarray(y1, dtype=float) - y0
    return percentile_interval(bootstrap_replicates(_mean, diff, **kwargs), level)
