"""Sequential McNemar test: a paired comparison that may stop before the rollout budget.

Derivation in docs/plan.md, Section 5. Both policies run on the same key at each step. Only the
discordant pairs carry information, and under the null "same success rate" the running sum of
their signs is a symmetric simple random walk, whatever the rate of discordant pairs. The test
rejects when the walk reaches a boundary computed so that, given the steps at which the
discordant pairs arrived, a symmetric walk has crossed it by step n with probability at most
the risk allowed at step n. That probability is computed exactly by propagating the
distribution of the walk over the paths that have not stopped yet.

The test looks at the walk each time a discordant pair arrives, and once more at the last step
of the budget, where all the risk left may be used.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class SequentialResult(NamedTuple):
    """Outcome of a sequential test on each evaluation of a batch.

    `stop` is the number of rollouts per policy at which the test stopped: the budget when no
    decision was reached. `decision` is +1 if the second policy was declared better, -1 if the
    first was, 0 for no decision. When boundaries are requested, `boundary[n - 1]` is the
    absolute value the walk must reach at step n (0 where the test does not look), and
    `spent[n - 1]` the probability that a symmetric walk has crossed by step n.
    """

    stop: np.ndarray
    decision: np.ndarray
    boundary: np.ndarray | None = None
    spent: np.ndarray | None = None


def risk_schedule(budget: int, alpha: float = 0.05, spending: str = "uniform") -> np.ndarray:
    """Cumulative false-positive risk the test may have used after each of `budget` steps.

    "uniform" allows alpha n / budget after n steps. "final" allows nothing before the last
    step, which makes the test McNemar's exact test at n = budget.
    """
    if spending == "uniform":
        return alpha * np.arange(1, budget + 1) / budget
    if spending == "final":
        return np.where(np.arange(1, budget + 1) == budget, alpha, 0.0)
    raise ValueError(f"unknown spending {spending!r}")


def sequential_mcnemar(
    y0: np.ndarray,
    y1: np.ndarray,
    alpha: float = 0.05,
    spending: str | np.ndarray = "uniform",
    return_boundary: bool = False,
) -> SequentialResult:
    """Run the test on outcomes of two policies on the same keys, in the order they arrive.

    `y0` and `y1` have shape (budget,) or (evaluations, budget): the length of the sequence is
    the rollout budget. `spending` names a schedule of `risk_schedule`, or is the cumulative
    risk allowed after each step (non-decreasing, at most `alpha`).
    """
    d = np.atleast_2d(np.asarray(y1, dtype=np.int64) - np.asarray(y0, dtype=np.int64))
    n_eval, budget = d.shape
    risk = risk_schedule(budget, alpha, spending) if isinstance(spending, str) else spending
    risk = np.asarray(risk, dtype=float)
    if risk.shape != (budget,) or (np.diff(risk) < 0).any() or risk[0] < 0 or risk[-1] > alpha:
        raise ValueError("the risk schedule must be non-decreasing, one value per step, <= alpha")

    # dist[e, half + w]: probability, under the null and given the steps at which the
    # discordant pairs of evaluation e arrived, that the walk is at w and has not stopped.
    half = budget + 1
    position = np.abs(np.arange(-half, half + 1))[None, :]
    dist = np.zeros((n_eval, 2 * half + 1))
    dist[:, half] = 1.0
    crossed = np.zeros(n_eval)
    walk = np.zeros(n_eval, dtype=np.int64)
    alive = np.ones(n_eval, dtype=bool)
    stop = np.full(n_eval, budget)
    decision = np.zeros(n_eval, dtype=np.int64)
    boundary = np.zeros((n_eval, budget), dtype=np.int64) if return_boundary else None
    spent = np.zeros((n_eval, budget)) if return_boundary else None

    for n in range(1, budget + 1):
        # The boundary does not depend on the signs, so it can be followed past a stop.
        tracked = np.ones_like(alive) if return_boundary else alive
        step = d[:, n - 1]
        moved = np.flatnonzero(tracked & (step != 0))
        if len(moved):
            p = dist[moved]
            p[:, 1:-1] = 0.5 * (p[:, :-2] + p[:, 2:])
            dist[moved] = p
            walk[moved] += step[moved]
        look = np.flatnonzero(tracked) if n == budget else moved
        if len(look):
            p = dist[look]
            # tail[:, a - 1]: probability of not having stopped and being at distance >= a
            # from zero, for a = 1, ..., half + 1 (the last one is zero).
            away = p[:, half + 1 :] + p[:, half - 1 :: -1]
            tail = np.cumsum(away[:, ::-1], axis=1)[:, ::-1]
            tail = np.concatenate([tail, np.zeros((len(look), 1))], axis=1)
            # Smallest distance whose crossing keeps the total within the risk allowed so far.
            a = 1 + np.argmax(crossed[look, None] + tail <= risk[n - 1], axis=1)
            crossed[look] += tail[np.arange(len(look)), a - 1]
            p[position >= a[:, None]] = 0.0
            dist[look] = p
            if return_boundary:
                boundary[look, n - 1] = a
            hit = look[alive[look] & (np.abs(walk[look]) >= a)]
            stop[hit] = n
            decision[hit] = np.sign(walk[hit])
            alive[hit] = False
        if return_boundary:
            spent[:, n - 1] = crossed
        elif not alive.any():
            break

    return SequentialResult(stop, decision, boundary, spent)


def rejection_curve(result: SequentialResult, budget: int) -> np.ndarray:
    """Share of evaluations that have rejected by each step 1, ..., budget."""
    steps = np.asarray(result.stop)[np.asarray(result.decision) != 0]
    return np.cumsum(np.bincount(steps, minlength=budget + 1)[1:]) / len(result.stop)
