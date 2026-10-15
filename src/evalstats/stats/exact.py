"""Fixed-sample exact tests of "two policies have the same success rate", two-sided.

Unpaired design: Barnard's test on the two success counts. Paired design: McNemar's exact test
on the two counts of discordant pairs. Both are written to be called on thousands of simulated
evaluations: they take arrays of counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
from scipy import optimize, stats


def mcnemar_pvalue(b, c) -> np.ndarray:
    """McNemar's exact two-sided p-value from the counts of the two kinds of discordant pairs.

    Same value as `statsmodels.stats.contingency_tables.mcnemar(table, exact=True)`.
    """
    b, c = np.asarray(b), np.asarray(c)
    return np.minimum(1.0, 2 * stats.binom.cdf(np.minimum(b, c), b + c, 0.5))


def barnard_pvalue(x0: int, x1: int, n: int) -> float:
    """Barnard's exact two-sided p-value (scipy) for `x0` and `x1` successes out of `n` each."""
    return float(stats.barnard_exact([[x0, x1], [n - x0, n - x1]], pooled=True).pvalue)


@dataclass(frozen=True)
class BarnardRegion:
    """Rejection region of Barnard's test for two samples of `n` episodes.

    With `x0` successes in the first sample, the test rejects when the second has at most
    `low[x0]` or at least `high[x0]`. `size` is the largest rejection probability over the
    common success rate under the null.
    """

    n: int
    alpha: float
    low: np.ndarray
    high: np.ndarray
    size: float

    def reject(self, x0, x1) -> np.ndarray:
        x0, x1 = np.asarray(x0), np.asarray(x1)
        return (x1 <= self.low[x0]) | (x1 >= self.high[x0])


@cache
def barnard_region(n: int, alpha: float = 0.05) -> BarnardRegion:
    """Rejection region of Barnard's test at level `alpha`, computed once per `n`.

    scipy's `barnard_exact` takes too long to be called on every table of a power simulation,
    and its outcome depends only on the two counts. The test orders tables by the pooled Wald
    statistic and rejects those at or beyond a critical value, the smallest one whose rejection
    probability stays within `alpha` for every common success rate. The square of the
    statistic is 2n (x1 - x0)^2 / (s (2n - s)) with s = x0 + x1. For a given x0 it increases
    as x1 moves away from x0, so the region is two bounds per x0, found here by bisection on
    the critical value.
    """
    x = np.arange(n + 1)

    def bounds(critical: float) -> tuple[np.ndarray, np.ndarray]:
        # high[x0]: smallest x1 > x0 at or beyond the critical value, n + 1 if there is none.
        below, high = x.copy(), np.full(n + 1, n + 1)
        while (high - below > 1).any():
            mid = (below + high) // 2
            s = x + mid
            rejected = (mid > x) & (2 * n * (mid - x) ** 2 >= critical * s * (2 * n - s))
            high, below = np.where(rejected, mid, high), np.where(rejected, below, mid)
        # The statistic is unchanged when successes and failures are exchanged.
        return n - high[::-1], high

    # Common success rates at which the rejection probability is evaluated: it is symmetric
    # around 1/2, and its peaks get narrower towards 0, hence the geometric part of the grid.
    rates = np.union1d(np.geomspace(0.05 / n, 0.5, 1500), np.linspace(0, 0.5, 501)[1:])
    pmf = stats.binom.pmf(x[None, :], n, rates[:, None])
    cdf = np.concatenate([np.zeros((len(rates), 1)), np.cumsum(pmf, axis=1)], axis=1)

    def size_at(rate: float, low: np.ndarray, high: np.ndarray) -> float:
        f = stats.binom.pmf(x, n, rate)
        c = np.concatenate([[0.0], np.cumsum(f)])
        return float(f @ (c[low + 1] + 1 - c[high]))

    def size(low: np.ndarray, high: np.ndarray) -> float:
        on_grid = (pmf * (cdf[:, low + 1] + 1 - cdf[:, high])).sum(axis=1)
        best = on_grid.max()
        # Refine around the highest grid points, where a peak may sit between two of them.
        for i in np.argsort(on_grid)[-3:]:
            left, right = rates[max(i - 1, 0)], rates[min(i + 1, len(rates) - 1)]
            res = optimize.minimize_scalar(
                lambda r: -size_at(r, low, high), bounds=(left, right), method="bounded"
            )
            best = max(best, -res.fun)
        return float(best)

    # Invariant: the region of `valid` has size <= alpha, the region of `invalid` does not.
    invalid, valid = 0.0, 2.0 * n + 1
    while valid - invalid > 1e-12 * valid:
        mid = (invalid + valid) / 2
        if size(*bounds(mid)) <= alpha:
            valid = mid
        else:
            invalid = mid
    low, high = bounds(valid)
    return BarnardRegion(n, alpha, low, high, size(low, high))
