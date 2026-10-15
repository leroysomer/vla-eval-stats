"""STEP (Snyder et al., 2025): the unpaired sequential test, from the authors' package.

`sequentialized_barnard_tests` is used as a dependency (CC BY-NC 4.0); none of its code is
copied here. STEP is one-sided, so it runs in both directions at alpha / 2 each, which is the
package's `MirroredStepTest` with `inference_mode="ranking"`. The package looks for its decision
policy among its own files and synthesizes it on first use, which takes a few minutes of CPU
for a budget of 500. It is not tested by its authors beyond that budget.
"""

from __future__ import annotations

import logging
import time
from functools import cache

import numpy as np
from sequentialized_barnard_tests import Decision, Hypothesis, MirroredStepTest

from evalstats.stats.sequential import SequentialResult

logger = logging.getLogger(__name__)

MAX_BUDGET = 500


def step_test(budget: int = 500, alpha: float = 0.05) -> MirroredStepTest:
    """Two-sided STEP test for `budget` rollouts per policy, with its default uniform risk budget."""
    return _step_test(budget, alpha)


@cache
def _step_test(budget: int, alpha: float) -> MirroredStepTest:
    if budget > MAX_BUDGET:
        raise ValueError(f"STEP is not tested beyond a budget of {MAX_BUDGET}")
    start = time.perf_counter()
    test = MirroredStepTest(Hypothesis.P0LessThanP1, budget, alpha, inference_mode="ranking")
    if test.policy is None:
        # The constructor of the package drops the policy it has just loaded (0.0.9).
        test.load_existing_policy()
    if test.policy is None:
        raise RuntimeError(f"no STEP policy for a budget of {budget} at level {alpha}")
    logger.info("STEP policy for budget %d ready in %.1f s", budget, time.perf_counter() - start)
    return test


def run_step(
    y0: np.ndarray, y1: np.ndarray, rng: np.random.Generator, alpha: float = 0.05
) -> SequentialResult:
    """Run STEP on outcomes of two policies, each on its own keys, in the order they arrive.

    `y0` and `y1` have shape (budget,) or (evaluations, budget). STEP randomizes some of its
    decisions; `rng` seeds that randomness, one seed per evaluation.
    """
    y0, y1 = np.atleast_2d(y0), np.atleast_2d(y1)
    n_eval, budget = y0.shape
    test = step_test(budget, alpha)
    stop = np.full(n_eval, budget)
    decision = np.zeros(n_eval, dtype=np.int64)
    signs = {Decision.AcceptAlternative: 1, Decision.AcceptNull: -1}
    for e, (row0, row1) in enumerate(zip(y0.tolist(), y1.tolist(), strict=True)):
        test.random_seed = int(rng.integers(2**63))
        test.reset()
        for n, (a, b) in enumerate(zip(row0, row1, strict=True), start=1):
            outcome = test.step(a, b).decision
            if outcome != Decision.FailToDecide:
                stop[e], decision[e] = n, signs[outcome]
                break
    return SequentialResult(stop, decision)
