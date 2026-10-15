#!/usr/bin/env python
"""Step 5 validation: every method of the toolkit on simulated data with known truth.

    uv run scripts/validate_stats.py

CPU only, about ten minutes. Data come from the nested Bernoulli model of
evalstats.stats.simulate on the layout of the main grid (10 tasks x 50 states x 4 seeds).
Writes four tables in results/tables/ (docs/plan.md, Section 6):

- stats_recovery.csv  the moment estimators against the true components
- coverage.csv        coverage of the bootstrap and Clopper-Pearson intervals
- type1.csv           rejection rate of each test under a true null
- power_check.csv     resampled power against the formula
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evalstats.stats import power
from evalstats.stats.exact import barnard_region, mcnemar_pvalue
from evalstats.stats.intervals import (
    bootstrap_replicates,
    clopper_pearson,
    percentile_interval,
)
from evalstats.stats.sequential import sequential_mcnemar
from evalstats.stats.simulate import NestedBernoulli
from evalstats.stats.step import run_step
from evalstats.stats.variance import components, design_effect

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("validate_stats")

N_INIT, N_SEEDS = 50, 4
LEVELS = ("task", "init", "seed")

# One policy run twice on other seeds (the null control of the plan), in four variance regimes.
NULLS = {
    "balanced": NestedBernoulli(p=(0.8, 0.8), var_task=(0.01, 0.01), var_init=(0.05, 0.05)),
    "init_heavy": NestedBernoulli(p=(0.6, 0.6), var_task=(0.03, 0.03), var_init=(0.14, 0.14)),
    "task_heavy": NestedBernoulli(p=(0.5, 0.5), var_task=(0.08, 0.08), var_init=(0.002, 0.002)),
    "high_rate": NestedBernoulli(p=(0.95, 0.95), var_task=(0.0006, 0.0006), var_init=(0.012, 0.012)),
}  # fmt: skip
# Two different policies, tied at every level.
GAPS = {
    "gap_10_points": NestedBernoulli(
        p=(0.7, 0.8), var_task=(0.02, 0.005), var_init=(0.08, 0.05),
        rho_task=0.8, rho_init=0.7, rho_seed=0.3,
    ),
    "gap_4_points": NestedBernoulli(
        p=(0.90, 0.94), var_task=(0.002, 0.0004), var_init=(0.03, 0.02),
        rho_task=0.8, rho_init=0.7, rho_seed=0.3,
    ),
}  # fmt: skip


def recovery(rng: np.random.Generator, reps: int) -> pd.DataFrame:
    rows = []
    for name, model in {**NULLS, **GAPS}.items():
        y = model.sample_grid(N_INIT, N_SEEDS, rng, size=(reps,))
        cases = [("variance, first policy", components(y[0]), model.variance(0))]
        if name in GAPS:
            cases.append(("variance, second policy", components(y[1]), model.variance(1)))
            cases.append(("covariance", components(y[0], y[1]), model.covariance))
        for quantity, estimates, truth in cases:
            for level, est, true in zip(LEVELS, estimates, truth, strict=True):
                rows.append({
                    "scenario": name, "quantity": quantity, "level": level, "truth": true,
                    "mean_estimate": est.mean(), "mc_se": est.std(ddof=1) / np.sqrt(reps),
                    "sd_estimate": est.std(ddof=1), "reps": reps,
                })  # fmt: skip
    out = pd.DataFrame(rows)
    out["z"] = (out["mean_estimate"] - out["truth"]) / out["mc_se"]
    return out


def _component_stats(y: np.ndarray) -> np.ndarray:
    return np.stack(components(y), axis=-1)


def coverage(rng: np.random.Generator, reps: int, n_boot: int) -> pd.DataFrame:
    rows = []

    def add(scenario, quantity, method, truth, low, high, deff=np.nan, boot=0):
        covered = (low <= truth) & (truth <= high)
        rows.append({
            "scenario": scenario, "quantity": quantity, "method": method, "truth": truth,
            "design_effect": deff, "coverage": covered.mean(),
            "mc_se": np.sqrt(covered.mean() * (1 - covered.mean()) / len(covered)),
            "mean_width": (high - low).mean(), "reps": len(covered), "n_boot": boot,
        })  # fmt: skip

    for name, model in {**NULLS, **GAPS}.items():
        start = time.perf_counter()
        y = model.sample_grid(N_INIT, N_SEEDS, rng, size=(reps,)).astype(float)
        truth, deff = model.variance(0), design_effect(model.variance(0), N_SEEDS)
        low, high = clopper_pearson(y[0].sum(axis=(1, 2, 3)), y[0][0].size)
        add(name, "rate", "clopper_pearson", model.p[0], low, high, deff)

        mean = lambda a: a.mean(axis=(1, 2, 3))
        # The rate and the gap only need the mean of each state; the components need the seeds.
        rate = np.array([
            percentile_interval(bootstrap_replicates(mean, m, n_boot=n_boot, rng=rng))
            for m in y[0].mean(-1, keepdims=True)
        ])  # fmt: skip
        add(name, "rate", "bootstrap", model.p[0], rate[:, 0], rate[:, 1], deff, n_boot)
        if name in GAPS:
            gap = np.array([
                percentile_interval(bootstrap_replicates(mean, m, n_boot=n_boot, rng=rng))
                for m in (y[1] - y[0]).mean(-1, keepdims=True)
            ])  # fmt: skip
            add(name, "gap", "bootstrap", model.p[1] - model.p[0], gap[:, 0], gap[:, 1], boot=n_boot)  # fmt: skip
        # Fewer replicates for the components, which resample every episode.
        few, boot = reps // 4, n_boot // 2
        comp = np.array([
            percentile_interval(bootstrap_replicates(_component_stats, g, n_boot=boot, rng=rng))
            for g in y[0][:few]
        ])  # fmt: skip
        for j, level in enumerate(LEVELS):
            add(name, f"var_{level}", "bootstrap", truth[j], comp[:, 0, j], comp[:, 1, j], boot=boot)  # fmt: skip
        logger.info("coverage %s: %.0f s", name, time.perf_counter() - start)
    return pd.DataFrame(rows)


def type1(rng: np.random.Generator, reps: int, budget: int) -> pd.DataFrame:
    rows = []

    def add(test, setting, n, rejected, reps_):
        low, high = clopper_pearson(rejected, reps_)
        rows.append({
            "test": test, "setting": setting, "n": n, "reps": reps_, "rejections": rejected,
            "rate": rejected / reps_, "ci95_low": low, "ci95_high": high,
        })  # fmt: skip

    for name, model in NULLS.items():
        for n in (50, 200, 1000):
            pairs = model.sample_keys((reps, n), rng).astype(np.int64)
            other = model.sample_keys((reps, n), rng).astype(np.int64)
            reject = barnard_region(n).reject(pairs[0].sum(1), other[1].sum(1))
            add("barnard", f"{name}: one policy, independent keys", n, int(reject.sum()), reps)
            diff = pairs[1] - pairs[0]
            reject = mcnemar_pvalue((diff < 0).sum(1), (diff > 0).sum(1)) <= 0.05
            add("mcnemar", f"{name}: one policy, same states, other seeds", n, int(reject.sum()), reps)  # fmt: skip
    for discordance in (0.02, 0.05, 0.1, 0.2, 0.5):
        cell = rng.random((reps, budget))
        y0, y1 = cell < discordance / 2, (cell >= discordance / 2) & (cell < discordance)
        result = sequential_mcnemar(y0, y1)
        add("sequential_mcnemar", f"discordant pairs {discordance:.0%} of keys", budget, int((result.decision != 0).sum()), reps)  # fmt: skip
    for name, model in NULLS.items():
        pairs = model.sample_keys((reps, budget), rng)
        result = sequential_mcnemar(pairs[0], pairs[1])
        add("sequential_mcnemar", f"{name}: one policy, same states, other seeds", budget, int((result.decision != 0).sum()), reps)  # fmt: skip
    for p in (0.05, 0.2, 0.5, 0.8, 0.95):
        start = time.perf_counter()
        result = run_step(rng.random((reps, budget)) < p, rng.random((reps, budget)) < p, rng)
        add("step", f"independent Bernoulli sequences, both rates {p}", budget, int((result.decision != 0).sum()), reps)  # fmt: skip
        logger.info("type I, STEP at %.2f: %.0f s", p, time.perf_counter() - start)
    out = pd.DataFrame(rows)
    # A rate is compatible with the level if its interval does not lie entirely above 5%.
    out["compatible_with_5pct"] = out["ci95_low"] <= 0.05
    exact = [
        {"test": "barnard", "setting": "exact, worst common success rate", "n": n,
         "rate": barnard_region(n).size, "compatible_with_5pct": barnard_region(n).size <= 0.05}
        for n in (50, 200, 500, 1000, 5000)
    ]  # fmt: skip
    out = pd.concat([out, pd.DataFrame(exact)], ignore_index=True)
    return out.astype({"reps": "Int64", "rejections": "Int64"})


def power_check(rng: np.random.Generator, reps: int) -> pd.DataFrame:
    rows = []
    for name, model in GAPS.items():
        # One grid plays the population, as the measured grid does in the analysis.
        y0, y1 = model.sample_grid(N_INIT, N_SEEDS, rng).astype(np.int64)
        p0, p1 = y0.mean(), y1.mean()
        b, c = ((y0 == 1) & (y1 == 0)).mean(), ((y0 == 0) & (y1 == 1)).mean()
        sigma2 = {"unpaired": power.sigma2_unpaired(p0, p1), "paired": power.sigma2_paired(b, c)}
        for design in power.DESIGNS:
            n80 = power.formula_n(p1 - p0, sigma2[design])
            for n in sorted({*(round(n80 * f) for f in (0.25, 0.5, 1, 1.5, 2.5)), 500}):
                rows.append({
                    "scenario": name, "design": design, "gap": p1 - p0, "sigma2": sigma2[design],
                    "n": n, "reps": reps,
                    "resampled_power": power.resampled_power(y0, y1, n, design, rng, reps=reps),
                    "formula_power": power.formula_power(n, p1 - p0, sigma2[design]),
                    "formula_n_80": n80,
                })  # fmt: skip
    out = pd.DataFrame(rows)
    out["difference"] = out["resampled_power"] - out["formula_power"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tables-dir", type=Path, default=REPO_ROOT / "results/tables")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reps", type=int, default=20_000, help="simulated evaluations per test")
    parser.add_argument("--datasets", type=int, default=2000, help="simulated grids per scenario")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--budget", type=int, default=500, help="rollouts of the sequential tests")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 60)
    tables = {
        "stats_recovery": lambda rng: recovery(rng, args.datasets),
        "coverage": lambda rng: coverage(rng, args.datasets, args.n_boot),
        "type1": lambda rng: type1(rng, args.reps, args.budget),
        "power_check": lambda rng: power_check(rng, reps=5000),
    }
    # One random stream per table, so that a table does not depend on the others.
    streams = np.random.SeedSequence(args.seed).spawn(len(tables))
    for (name, build), stream in zip(tables.items(), streams, strict=True):
        start = time.perf_counter()
        table = build(np.random.default_rng(stream))
        table.to_csv(args.tables_dir / f"{name}.csv", index=False, float_format="%.5g")
        logger.info("%s: %.0f s\n%s", name, time.perf_counter() - start, table.round(4).to_string(index=False))  # fmt: skip


if __name__ == "__main__":
    main()
