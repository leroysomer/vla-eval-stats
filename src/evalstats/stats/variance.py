"""Variance components of the nested design: task, initial state within task, seed within state.

Outcomes are arrays whose last three axes are (task, initial state, seed), with the same number
of states for every task and of seeds for every state. Tasks are fixed and equally weighted;
initial states and seeds are random (docs/plan.md, Sections 2 and 3).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd


class Components(NamedTuple):
    """A variance (or a covariance between two policies) split over the three levels."""

    task: np.ndarray | float
    init: np.ndarray | float
    seed: np.ndarray | float

    @property
    def total(self):
        return self.task + self.init + self.seed


def episodes_to_array(episodes: pd.DataFrame, value: str = "success") -> np.ndarray:
    """Per-episode rows of one policy as an array indexed by (task, initial state, seed)."""
    index = ["task_id", "init_state_id", "seed"]
    shape = tuple(episodes[c].nunique() for c in index)
    if len(episodes) != np.prod(shape) or episodes.duplicated(index).any():
        raise ValueError(f"not a full {shape} grid of (task, initial state, seed): {len(episodes)} rows")  # fmt: skip
    return episodes.sort_values(index)[value].to_numpy().reshape(shape)


def components(x: np.ndarray, y: np.ndarray | None = None) -> Components:
    """Unbiased moment estimators of the components of Var(x), or of Cov(x, y) if `y` is given.

    `x` and `y` have shape (..., tasks, states, seeds) and the estimators run over the last
    three axes. The three components add up to an unbiased estimate of p(1 - p), or of
    E[xy] - E[x]E[y], for an episode drawn with a uniformly random task. Being unbiased, an
    estimate can be negative when its component is close to zero.
    """
    x = np.asarray(x, dtype=float)
    y = x if y is None else np.asarray(y, dtype=float)
    n_tasks, n_init, n_seeds = x.shape[-3:]
    if n_init < 2 or n_seeds < 2:
        raise ValueError("needs at least 2 initial states per task and 2 seeds per state")

    x_cell, y_cell = x.mean(-1), y.mean(-1)
    x_task, y_task = x_cell.mean(-1), y_cell.mean(-1)
    # Between seeds, within a state: sample covariance of the seeds of each state.
    seed = ((x * y).mean(-1) - x_cell * y_cell).mean(-1) * n_seeds / (n_seeds - 1)
    # Sample covariance of the state means of a task. It estimates the between-state component
    # plus 1/n_seeds of the within-state one.
    cells = ((x_cell - x_task[..., None]) * (y_cell - y_task[..., None])).sum(-1) / (n_init - 1)
    init = cells - seed / n_seeds
    # The task means are estimated with noise cells / n_init each, which inflates their spread.
    dx = x_task - x_task.mean(-1, keepdims=True)
    dy = y_task - y_task.mean(-1, keepdims=True)
    task = (dx * dy).mean(-1) - (n_tasks - 1) / n_tasks * (cells / n_init).mean(-1)
    return Components(task, init.mean(-1), seed.mean(-1))


def rate_variance(c: Components, n_tasks: int, n_init: int, n_seeds: int):
    """Variance of the success rate estimated with `n_init` states per task, `n_seeds` each."""
    return c.init / (n_tasks * n_init) + c.seed / (n_tasks * n_init * n_seeds)


def design_effect(c: Components, n_seeds: int):
    """Variance of the estimated rate over the binomial variance p(1 - p) / n.

    Below 1 the binomial interval is too wide (task differences counted as noise), above 1 it
    is too narrow (repeated seeds of one state counted as independent episodes).
    """
    return (n_seeds * c.init + c.seed) / c.total
