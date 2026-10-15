"""The statistics toolkit on simulated data with known truth (docs/plan.md, Section 6).

Simulation sizes are small enough for the whole file to run in about a minute. Tolerances on
simulated rates are three or four Monte Carlo standard errors. The full-size versions of these
checks are the tables written by scripts/validate_stats.py.
"""

import itertools
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.contingency_tables import mcnemar

from evalstats.stats import power
from evalstats.stats.exact import barnard_pvalue, barnard_region, mcnemar_pvalue
from evalstats.stats.intervals import bootstrap_gap, bootstrap_rate, clopper_pearson
from evalstats.stats.sequential import rejection_curve, risk_schedule, sequential_mcnemar
from evalstats.stats.simulate import NestedBernoulli
from evalstats.stats.step import run_step
from evalstats.stats.variance import (
    Components,
    components,
    design_effect,
    episodes_to_array,
    rate_variance,
)

# One policy run twice on other seeds: the null control.
SAME = NestedBernoulli(p=(0.6, 0.6), var_task=(0.03, 0.03), var_init=(0.14, 0.14))
# Two policies with a gap of 10 points, tied at every level.
GAP = NestedBernoulli(
    p=(0.7, 0.8), var_task=(0.02, 0.005), var_init=(0.08, 0.05),
    rho_task=0.8, rho_init=0.7, rho_seed=0.3,
)  # fmt: skip


def rate_se(rate: float, reps: int) -> float:
    return np.sqrt(rate * (1 - rate) / reps)


# --- variance components -------------------------------------------------------------------


def test_components_recover_the_truth():
    reps = 600
    y = GAP.sample_grid(50, 4, np.random.default_rng(0), size=(reps,))
    cases = [
        (components(y[0]), GAP.variance(0)),
        (components(y[1]), GAP.variance(1)),
        (components(y[0], y[1]), GAP.covariance),
    ]
    for estimates, truth in cases:
        for est, true in zip(estimates, truth, strict=True):
            assert abs(est.mean() - true) < 4 * est.std(ddof=1) / np.sqrt(reps)
    # The three components of a variance add up to p (1 - p).
    assert GAP.variance(0).total == pytest.approx(0.7 * 0.3)
    assert GAP.covariance.total == pytest.approx(GAP.joint[0] - 0.7 * 0.8)


def test_components_of_outcomes_that_depend_on_one_level_only():
    task_rates = np.array([0.0, 1.0, 1.0, 1.0])
    by_task = np.broadcast_to(task_rates[:, None, None], (4, 6, 3))
    c = components(by_task)
    assert c.task == pytest.approx(task_rates.var()) and c.init == c.seed == 0

    by_state = np.broadcast_to((np.arange(6) % 2)[None, :, None], (4, 6, 3))
    c = components(by_state)
    assert c.seed == 0
    assert c.init == pytest.approx(by_state[0, :, 0].var(ddof=1))
    # The task means are equal, and the estimator still removes the spread that the noise of
    # 6 states per task is expected to give them: it is unbiased, not non-negative.
    assert c.task == pytest.approx(-(3 / 4) * c.init / 6)


def test_rate_variance_and_design_effect():
    c = SAME.variance(0)
    y = SAME.sample_grid(50, 4, np.random.default_rng(1), size=(4000,))[0]
    simulated = y.mean(axis=(1, 2, 3)).var(ddof=1)
    assert rate_variance(c, 10, 50, 4) == pytest.approx(simulated, rel=0.1)
    binomial = 0.6 * 0.4 / y[0].size
    assert design_effect(c, 4) == pytest.approx(rate_variance(c, 10, 50, 4) / binomial)
    # Only seed noise: the grid is a stratified sample and the binomial variance is too large.
    assert design_effect(Components(0.1, 0.0, 0.1), n_seeds=4) == pytest.approx(0.5)
    # Only state differences: 4 seeds of a state are worth one episode.
    assert design_effect(Components(0.0, 0.2, 0.0), n_seeds=4) == pytest.approx(4)


def test_episodes_to_array():
    rng = np.random.default_rng(2)
    y = rng.integers(2, size=(3, 5, 4))
    t, i, s = np.indices(y.shape)
    df = pd.DataFrame(
        {"task_id": t.ravel(), "init_state_id": i.ravel(), "seed": s.ravel(), "success": y.ravel()}
    )
    shuffled = df.sample(frac=1, random_state=0)
    assert np.array_equal(episodes_to_array(shuffled), y)
    with pytest.raises(ValueError, match="not a full"):
        episodes_to_array(shuffled.iloc[1:])


# --- intervals -----------------------------------------------------------------------------


def coverage_of_rate(model: NestedBernoulli, seed: int, n_grids: int, n_bootstrapped: int):
    """Coverage of the true rate by Clopper-Pearson and by the bootstrap."""
    rng = np.random.default_rng(seed)
    y = model.sample_grid(50, 4, rng, size=(n_grids,))[0]
    low, high = clopper_pearson(y.sum(axis=(1, 2, 3)), y[0].size)
    naive = ((low <= model.p[0]) & (model.p[0] <= high)).mean()
    boot = np.array([bootstrap_rate(g, n_boot=400, rng=rng) for g in y[:n_bootstrapped]])
    cluster_aware = ((boot[:, 0] <= model.p[0]) & (model.p[0] <= boot[:, 1])).mean()
    return naive, cluster_aware


def test_bootstrap_covers_and_clopper_pearson_errs_as_the_design_effect_predicts():
    # Design effect 2.6: seeds of one state are far from independent, the naive interval is
    # too narrow. A normal interval 1/sqrt(2.6) too narrow covers 78% of the time.
    assert design_effect(SAME.variance(0), 4) == pytest.approx(2.625)
    naive, cluster_aware = coverage_of_rate(SAME, seed=3, n_grids=2000, n_bootstrapped=300)
    assert 0.74 < naive < 0.82
    assert 0.90 < cluster_aware < 0.99

    # Design effect 0.7: almost all the variance is between tasks, which are fixed, and the
    # naive interval is too wide.
    by_task = NestedBernoulli(p=(0.5, 0.5), var_task=(0.08, 0.08), var_init=(0.002, 0.002))
    assert design_effect(by_task.variance(0), 4) == pytest.approx(0.704)
    naive, cluster_aware = coverage_of_rate(by_task, seed=4, n_grids=2000, n_bootstrapped=300)
    assert naive > 0.965
    assert 0.90 < cluster_aware < 0.99


def test_bootstrap_gap_keeps_the_two_policies_paired():
    rng = np.random.default_rng(5)
    y = GAP.sample_grid(50, 4, rng, size=(300,))
    assert np.array_equal(bootstrap_gap(y[0, 0], y[0, 0], n_boot=50, rng=rng), [0, 0])
    gap = np.array([bootstrap_gap(a, b, n_boot=400, rng=rng) for a, b in zip(y[0], y[1])])
    covered = ((gap[:, 0] <= 0.1) & (0.1 <= gap[:, 1])).mean()
    assert 0.90 < covered < 0.99
    # The policies are positively correlated, so the paired interval is narrower than the
    # interval of two rates estimated on independent grids.
    rates = [np.array([bootstrap_rate(g, n_boot=200, rng=rng) for g in y[k, :50]]) for k in (0, 1)]
    width = [(r[:, 1] - r[:, 0]).mean() for r in rates]
    assert (gap[:, 1] - gap[:, 0]).mean() < np.hypot(*width)


# --- fixed-sample tests --------------------------------------------------------------------


def test_mcnemar_pvalue_is_statsmodels_exact_test():
    for b, c in itertools.product(range(13), repeat=2):
        expected = mcnemar([[5, b], [c, 7]], exact=True).pvalue
        assert mcnemar_pvalue(b, c) == pytest.approx(expected)
    assert np.array_equal(mcnemar_pvalue([0, 0], [0, 6]), [1.0, 2 * 0.5**6])


@pytest.mark.parametrize("n", [6, 15])
def test_barnard_region_is_scipy_test_on_every_table(n):
    region = barnard_region(n)
    for x0, x1 in itertools.product(range(n + 1), repeat=2):
        assert bool(region.reject(x0, x1)) == (barnard_pvalue(x0, x1, n) <= 0.05), (x0, x1)


def test_barnard_region_at_larger_n():
    # On the edge of the region: scipy rejects the table on it and not the one before it.
    region = barnard_region(60)
    for x0 in (3, 20, 30, 45):
        edge = int(region.high[x0])
        assert barnard_pvalue(x0, edge, 60) <= 0.05 < barnard_pvalue(x0, edge - 1, 60)
    for n in (60, 400):
        region = barnard_region(n)
        assert 0.04 < region.size <= 0.05
        assert np.array_equal(region.low, n - region.high[::-1])
        assert not region.reject(np.arange(n + 1), np.arange(n + 1)).any()


@pytest.mark.parametrize("model", [SAME, NestedBernoulli(p=(0.95, 0.95), var_task=(0.0006,) * 2, var_init=(0.012,) * 2)])  # fmt: skip
def test_fixed_tests_hold_their_level_for_one_policy_run_twice(model):
    rng = np.random.default_rng(6)
    reps, n = 20_000, 100
    pairs, other = (model.sample_keys((reps, n), rng).astype(int) for _ in range(2))
    limit = 0.05 + 3 * rate_se(0.05, reps)
    # Unpaired: the two policies on independent keys.
    assert barnard_region(n).reject(pairs[0].sum(1), other[1].sum(1)).mean() < limit
    # Paired: the same states, other seeds.
    diff = pairs[1] - pairs[0]
    assert (mcnemar_pvalue((diff < 0).sum(1), (diff > 0).sum(1)) <= 0.05).mean() < limit


# --- power ---------------------------------------------------------------------------------


def test_formula_n_and_formula_power_are_inverse():
    n = power.formula_n(0.02, 0.2, power=0.8)
    assert power.formula_power(n, 0.02, 0.2) == pytest.approx(0.8)
    assert n == pytest.approx((1.959964 + 0.841621) ** 2 * 0.2 / 0.02**2, rel=1e-5)
    # Pairing removes twice the covariance.
    b, c = GAP.joint[1], GAP.joint[2]
    unpaired = power.sigma2_unpaired(0.7, 0.8)
    assert power.sigma2_paired(b, c) == pytest.approx(unpaired - 2 * GAP.covariance.total)


def test_resampled_power_agrees_with_the_formula():
    rng = np.random.default_rng(7)
    # One simulated grid plays the population, as the measured grid does in the analysis.
    y0, y1 = GAP.sample_grid(50, 4, rng).astype(int)
    gap = y1.mean() - y0.mean()
    b, c = ((y0 == 1) & (y1 == 0)).mean(), ((y0 == 0) & (y1 == 1)).mean()
    sigma2 = {
        "unpaired": power.sigma2_unpaired(y0.mean(), y1.mean()),
        "paired": power.sigma2_paired(b, c),
    }
    assert sigma2["paired"] < sigma2["unpaired"]
    for design in power.DESIGNS:
        # At twice the n of 80% power the formula gives 97.9%.
        n = int(2 * power.formula_n(gap, sigma2[design]))
        resampled = power.resampled_power(y0, y1, n, design, rng, reps=3000)
        assert resampled == pytest.approx(power.formula_power(n, gap, sigma2[design]), abs=0.02)
        # At the n of 80% power the exact tests are a little below the normal approximation.
        n = int(power.formula_n(gap, sigma2[design]))
        resampled = power.resampled_power(y0, y1, n, design, rng, reps=3000)
        assert 0.70 < resampled < 0.83


def test_resampled_power_of_a_policy_against_itself_is_at_most_the_level():
    rng = np.random.default_rng(8)
    y0 = SAME.sample_grid(50, 4, rng)[0].astype(int)
    # Same states, other seeds: every state gives 12 ordered pairs of different seeds.
    first, second = np.array([(a, b) for a, b in itertools.permutations(range(4), 2)]).T
    limit = 0.05 + 3 * rate_se(0.05, 4000)
    assert power.resampled_power(y0[..., first], y0[..., second], 200, "paired", rng, reps=4000) < limit  # fmt: skip
    assert power.resampled_power(y0, y0, 200, "unpaired", rng, reps=4000) < limit


def test_n_at_power():
    assert power.n_at_power([100, 400], [0.6, 1.0]) == pytest.approx(200)
    assert power.n_at_power([100, 400], [0.9, 1.0]) == 100
    assert np.isnan(power.n_at_power([100, 400], [0.1, 0.5]))


# --- sequential McNemar test ---------------------------------------------------------------


def crossing_probability(discordant, boundary) -> list[Fraction]:
    """Exact probability that a symmetric walk has crossed `boundary` by each step, by
    enumerating the signs of the discordant pairs."""
    n_signs = int(sum(discordant))
    first_crossing = [Fraction(0)] * len(discordant)
    for signs in itertools.product([-1, 1], repeat=n_signs):
        walk, k = 0, 0
        for n, moves in enumerate(discordant):
            if moves:
                walk, k = walk + signs[k], k + 1
            if boundary[n] and abs(walk) >= boundary[n]:
                first_crossing[n] += Fraction(1, 2**n_signs)
                break
    return list(itertools.accumulate(first_crossing))


@pytest.mark.parametrize("alpha", [0.05, 0.4])
def test_sequential_boundary_is_exact_and_cannot_be_lowered(alpha):
    discordant = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0])
    budget = len(discordant)
    result = sequential_mcnemar(np.zeros(budget), discordant, alpha=alpha, return_boundary=True)
    boundary = result.boundary[0]
    # The test looks at each discordant pair and at the last step.
    assert np.array_equal(boundary > 0, (discordant == 1) | (np.arange(budget) == budget - 1))

    risk = [Fraction(alpha).limit_denominator(1000) * n / budget for n in range(1, budget + 1)]
    crossed = crossing_probability(discordant, boundary)
    assert np.allclose(np.array(crossed, dtype=float), result.spent[0])
    assert all(c <= r for c, r in zip(crossed, risk, strict=True))
    # Each threshold is the smallest one that respects the risk allowed at its step.
    for n in np.flatnonzero(boundary):
        lowered = boundary.copy()
        lowered[n] -= 1
        assert lowered[n] == 0 or crossing_probability(discordant, lowered)[n] > risk[n]


def test_sequential_with_all_the_risk_at_the_end_is_mcnemar_exact_test():
    rng = np.random.default_rng(9)
    for budget, p0, p1 in [(40, 0.5, 0.8), (150, 0.8, 0.9), (150, 0.7, 0.7)]:
        y0, y1 = rng.random((2000, budget)) < p0, rng.random((2000, budget)) < p1
        result = sequential_mcnemar(y0, y1, spending="final")
        diff = y1.astype(int) - y0
        pvalue = mcnemar_pvalue((diff < 0).sum(1), (diff > 0).sum(1))
        assert np.array_equal(result.decision != 0, pvalue <= 0.05)
        assert (result.stop == budget).all()
        assert np.array_equal(result.decision[pvalue <= 0.05], np.sign(diff.sum(1))[pvalue <= 0.05])  # fmt: skip


@pytest.mark.parametrize("discordance", [0.02, 0.1, 0.5])
def test_sequential_holds_its_level_whatever_the_discordance(discordance):
    rng = np.random.default_rng(10)
    reps, budget = 6000, 200
    cell = rng.random((reps, budget))
    y0, y1 = cell < discordance / 2, (cell >= discordance / 2) & (cell < discordance)
    result = sequential_mcnemar(y0, y1, return_boundary=True)
    assert (result.decision != 0).mean() < 0.05 + 3 * rate_se(0.05, reps)
    # Given the steps of the discordant pairs, the exact false-positive probability is within
    # the risk allowed at every step.
    assert (result.spent <= risk_schedule(budget) + 1e-15).all()
    assert (np.diff(result.spent, axis=1) >= 0).all()


def test_sequential_stops_early_and_names_the_better_policy():
    rng = np.random.default_rng(11)
    y0, y1 = rng.random((500, 300)) < 0.5, rng.random((500, 300)) < 0.8
    forward, backward = sequential_mcnemar(y0, y1), sequential_mcnemar(y1, y0)
    assert (forward.decision == 1).all() and (backward.decision == -1).all()
    assert np.array_equal(forward.stop, backward.stop)
    assert forward.stop.mean() < 150
    curve = rejection_curve(forward, 300)
    assert curve.shape == (300,) and curve[0] == 0 and curve[-1] == 1
    assert (np.diff(curve) >= 0).all()
    with pytest.raises(ValueError, match="risk schedule"):
        sequential_mcnemar(y0, y1, spending=np.full(300, 0.1))


# --- STEP ----------------------------------------------------------------------------------


def test_step_holds_its_level_and_names_the_better_policy():
    # A budget of 100 keeps the synthesis of the decision policy to a few seconds. The package
    # caches it among its own files.
    rng = np.random.default_rng(12)
    reps, budget = 4000, 100
    for p in (0.3, 0.9):
        result = run_step(rng.random((reps, budget)) < p, rng.random((reps, budget)) < p, rng)
        assert (result.decision != 0).mean() < 0.05 + 3 * rate_se(0.05, reps)
    y0, y1 = rng.random((300, budget)) < 0.3, rng.random((300, budget)) < 0.8
    forward, backward = run_step(y0, y1, rng), run_step(y1, y0, rng)
    assert (forward.decision == 1).all() and (backward.decision == -1).all()
    assert forward.stop.mean() < 50
    with pytest.raises(ValueError, match="not tested beyond"):
        run_step(np.zeros(501), np.zeros(501), rng)
