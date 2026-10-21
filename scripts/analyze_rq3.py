#!/usr/bin/env python
"""Step 8, RQ3: how many rollouts when the evaluator may stop early? (docs/plan.md, Section 5)

    uv run scripts/analyze_rq3.py

CPU only, a few minutes; no new rollouts. Budget of 500 rollouts per policy, two-sided
alpha = 0.05, risk spent uniformly over the budget. For A vs A′ and A vs B, evaluations are
drawn from the main grid: unpaired (independent keys for each policy, STEP) and paired (the same
keys for both, sequential McNemar). Needs results/tables/rq2_power.csv and
rq2_comparisons.csv from Step 7 for the fixed-n designs. Writes:

- results/tables/rq3_summary.csv  power within the budget and expected rollouts (Table 3)
- results/tables/rq3_curves.csv   probability of having rejected by each step (Figure 4)
- results/tables/rq3_null.csv     null control: A against itself
- figures/rq3.png                 Figure 4
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evalstats.analysis import load_outcomes
from evalstats.plots import style
from evalstats.plots.style import INK_2, SERIES
from evalstats.stats.intervals import clopper_pearson
from evalstats.stats.sequential import SequentialResult, rejection_curve, sequential_mcnemar
from evalstats.stats.step import run_step

REPO_ROOT = Path(__file__).resolve().parents[1]

COMPARISONS = {"A vs A′": ("smolvla_n10", "smolvla"), "A vs B": ("smolvla_n10", "pi05")}
DESIGNS = {"unpaired": "STEP", "paired": "sequential McNemar"}
DESIGN_COLORS = {"unpaired": SERIES[0], "paired": SERIES[1]}
FIXED_TESTS = {"unpaired": "Barnard", "paired": "McNemar"}


def draw(y0: np.ndarray, y1: np.ndarray, design: str, reps: int, budget: int, rng):
    """`reps` evaluations of `budget` rollouts per policy, keys drawn uniformly from the pool."""
    y0, y1 = np.ravel(y0), np.ravel(y1)
    keys = rng.integers(len(y0), size=(reps, budget))
    other = rng.integers(len(y1), size=(reps, budget)) if design == "unpaired" else keys
    return y0[keys], y1[other]


def run(design: str, s0: np.ndarray, s1: np.ndarray, rng) -> SequentialResult:
    return run_step(s0, s1, rng) if design == "unpaired" else sequential_mcnemar(s0, s1)


def summarize(result: SequentialResult, budget: int, truth_sign: int) -> dict:
    decided = result.decision != 0
    return {
        "power_within_budget": decided.mean(),
        "wrong_direction": (result.decision == -truth_sign).mean(),
        "expected_rollouts": result.stop.mean(),
        "median_stop": float(np.median(result.stop)),
        "p90_stop": float(np.quantile(result.stop, 0.9)),
        "reached_budget_undecided": (~decided).mean(),
    }


def figure(curves: pd.DataFrame, summary: pd.DataFrame, path: Path, budget: int) -> None:
    style.apply()
    fig, axes = plt.subplots(1, len(COMPARISONS), figsize=(10, 3.6), sharey=True)
    for ax, name in zip(axes, COMPARISONS, strict=True):
        s = summary[summary["comparison"] == name].set_index(["design", "procedure"])
        means = []
        for (design, test), size in zip(DESIGNS.items(), (13, 6), strict=True):
            c = curves[(curves["comparison"] == name) & (curves["design"] == design)]
            color = DESIGN_COLORS[design]
            mean = s.loc[(design, "sequential"), "expected_rollouts"]
            ax.plot(c["n"], c["prob_rejected"], color=color, linewidth=2,
                    label=f"{design}, {test}")  # fmt: skip
            ax.axvline(mean, color=color, linewidth=0.8, linestyle=":")
            means.append(f"{test}: {mean:.0f}")
            # The two fixed-n powers can coincide: the first marker is drawn larger.
            fixed = s.loc[(design, "fixed n"), "power_within_budget"]
            ax.plot([budget], [fixed], marker="D", markersize=size, color=color,
                    linestyle="none", markeredgecolor="white", markeredgewidth=1.5,
                    clip_on=False, zorder=4,
                    label=f"{FIXED_TESTS[design]}, fixed n = {budget}")  # fmt: skip
        ax.text(0.97, 0.35, "mean rollouts before stopping (dotted)\n" + "\n".join(means),
                transform=ax.transAxes, ha="right", va="center", fontsize=8.5, color=INK_2)  # fmt: skip
        ax.set_xlim(0, budget)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("rollouts per policy so far, n")
        ax.set_title(name, loc="left", fontsize=10)
        style.clean(ax, grid_axis="both")
    axes[0].set_ylabel("probability of having rejected by n")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--grid", type=Path, default=REPO_ROOT / "configs/grid_main.yaml")
    parser.add_argument("--policies-config", type=Path, default=REPO_ROOT / "configs/policies.yaml")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "results/raw")
    parser.add_argument("--tables-dir", type=Path, default=REPO_ROOT / "results/tables")
    parser.add_argument("--figure", type=Path, default=REPO_ROOT / "figures/rq3.png")
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--reps", type=int, default=5000)
    parser.add_argument("--null-reps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    outcomes = load_outcomes(args.grid, args.policies_config, args.raw_dir)
    fixed_power = pd.read_csv(args.tables_dir / "rq2_power.csv")
    fixed_n80 = pd.read_csv(args.tables_dir / "rq2_comparisons.csv").set_index(["comparison", "quantity"])  # fmt: skip
    if args.budget not in set(fixed_power["n"]):
        raise SystemExit(f"rq2_power.csv has no fixed-n power at n = {args.budget}")
    streams = iter(np.random.default_rng(s) for s in np.random.SeedSequence(args.seed).spawn(8))

    summary, curves = [], []
    for (name, (first, second)), design in itertools.product(COMPARISONS.items(), DESIGNS):
        y0, y1 = outcomes[first], outcomes[second]
        sign = int(np.sign(y1.mean() - y0.mean()))
        rng = next(streams)
        result = run(design, *draw(y0, y1, design, args.reps, args.budget, rng), rng)
        curve = rejection_curve(result, args.budget)
        curves.append(pd.DataFrame({"comparison": name, "design": design, "test": DESIGNS[design],
                                    "n": np.arange(1, args.budget + 1), "prob_rejected": curve}))  # fmt: skip
        summary.append({"comparison": name, "design": design, "procedure": "sequential",
                        "test": DESIGNS[design], "reps": args.reps,
                        **summarize(result, args.budget, sign)})  # fmt: skip
        f = fixed_power[(fixed_power["comparison"] == name) & (fixed_power["design"] == design)]
        summary.append({
            "comparison": name, "design": design, "procedure": "fixed n",
            "test": FIXED_TESTS[design], "reps": int(f["reps"].iloc[0]),
            "power_within_budget": float(f.loc[f["n"] == args.budget, "resampled_power"].iloc[0]),
            "expected_rollouts": float(args.budget),
            "n80_if_gap_known": float(fixed_n80.loc[(name, f"curve_n80_{design}"), "estimate"]),
        })  # fmt: skip

    # Null control: A against itself; paired on the same states with other seeds.
    a = outcomes["smolvla_n10"]
    first, second = np.array(list(itertools.permutations(range(a.shape[-1]), 2))).T
    pools = {"unpaired": (a, a), "paired": (a[..., first], a[..., second])}
    null = []
    for design, test in DESIGNS.items():
        rng = next(streams)
        s0, s1 = draw(*pools[design], design, args.null_reps, args.budget, rng)
        rejected = int((run(design, s0, s1, rng).decision != 0).sum())
        low, high = clopper_pearson(rejected, args.null_reps)
        null.append({"comparison": "A vs A (other seeds)", "design": design,
                     "test": test, "budget": args.budget, "reps": args.null_reps,
                     "rejections": rejected, "rate": rejected / args.null_reps,
                     "ci95_low": low, "ci95_high": high, "compatible_with_5pct": low <= 0.05})  # fmt: skip

    tables = {
        "rq3_summary": pd.DataFrame(summary),
        "rq3_curves": pd.concat(curves, ignore_index=True),
        "rq3_null": pd.DataFrame(null),
    }
    pd.set_option("display.width", 220, "display.max_columns", 30)
    for stem, df in tables.items():
        df.to_csv(args.tables_dir / f"{stem}.csv", index=False, float_format="%.6g")
        if stem != "rq3_curves":
            print(f"\n{stem}\n{df.round(4).to_string(index=False)}")
    figure(tables["rq3_curves"], tables["rq3_summary"], args.figure, args.budget)

    s = tables["rq3_summary"].set_index(["comparison", "design", "procedure"])["expected_rollouts"]
    print(
        f"\nWith a budget of {args.budget} rollouts per policy, A vs A′ is decided after about "
        f"{s['A vs A′', 'unpaired', 'sequential']:.0f} rollouts unpaired and "
        f"{s['A vs A′', 'paired', 'sequential']:.0f} paired, and A vs B after about "
        f"{s['A vs B', 'unpaired', 'sequential']:.0f} and {s['A vs B', 'paired', 'sequential']:.0f}."
    )


if __name__ == "__main__":
    main()
