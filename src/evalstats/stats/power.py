"""Power of the paired and unpaired designs: by resampling a pool of episodes, and by formula.

`n` is always the number of rollouts per policy (docs/plan.md, Section 4).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from evalstats.stats.exact import barnard_region, mcnemar_pvalue

DESIGNS = ("unpaired", "paired")


def sigma2_unpaired(p0: float, p1: float) -> float:
    """Variance of the difference of two outcomes on independent keys."""
    return p0 * (1 - p0) + p1 * (1 - p1)


def sigma2_paired(b: float, c: float) -> float:
    """Variance of the difference of two outcomes on the same key.

    `b` is the probability that only the first policy succeeds, `c` that only the second does.
    """
    return b + c - (c - b) ** 2


def formula_n(delta: float, sigma2: float, power: float = 0.8, alpha: float = 0.05) -> float:
    """Rollouts per policy for a two-sided test of level `alpha` to detect the gap `delta`."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return z**2 * sigma2 / delta**2


def formula_power(n, delta: float, sigma2: float, alpha: float = 0.05):
    """Power at `n` rollouts per policy under the normal approximation behind `formula_n`."""
    return stats.norm.cdf(abs(delta) * np.sqrt(n / sigma2) - stats.norm.ppf(1 - alpha / 2))


def resampled_power(
    y0: np.ndarray,
    y1: np.ndarray,
    n: int,
    design: str,
    rng: np.random.Generator,
    reps: int = 5000,
    alpha: float = 0.05,
    batch: int = 500,
) -> float:
    """Share of `reps` simulated evaluations of size `n` in which the test rejects.

    `y0` and `y1` are the outcomes of the two policies on the same pool of keys, in the same
    order. Keys are drawn from the pool uniformly with replacement.

    - "unpaired": `n` keys for the first policy and `n` other draws for the second, Barnard.
    - "paired": the same `n` keys for both, McNemar.
    """
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {DESIGNS}")
    y0, y1 = np.ravel(y0).astype(np.int64), np.ravel(y1).astype(np.int64)
    if y0.shape != y1.shape:
        raise ValueError("the two policies must be given on the same keys")
    rejected = 0
    for start in range(0, reps, batch):
        shape = (min(batch, reps - start), n)
        keys = rng.integers(len(y0), size=shape)
        if design == "unpaired":
            other = rng.integers(len(y1), size=shape)
            reject = barnard_region(n, alpha).reject(y0[keys].sum(1), y1[other].sum(1))
        else:
            diff = y1[keys] - y0[keys]
            reject = mcnemar_pvalue((diff < 0).sum(1), (diff > 0).sum(1)) <= alpha
        rejected += int(reject.sum())
    return rejected / reps


def n_at_power(ns, powers, target: float = 0.8) -> float:
    """Smallest `n` at which a power curve reaches `target`, interpolated on a log scale.

    NaN if the curve never reaches the target.
    """
    ns, powers = np.asarray(ns, dtype=float), np.asarray(powers, dtype=float)
    reached = np.flatnonzero(powers >= target)
    if len(reached) == 0:
        return float("nan")
    i = reached[0]
    if i == 0:
        return float(ns[0])
    share = (target - powers[i - 1]) / (powers[i] - powers[i - 1])
    return float(np.exp(np.log(ns[i - 1]) + share * (np.log(ns[i]) - np.log(ns[i - 1]))))
