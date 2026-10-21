#!/usr/bin/env python
"""Step 7, RQ2: how many rollouts to tell two policies apart, paired or unpaired?

    uv run scripts/analyze_rq2.py

CPU only, a few minutes. Follows docs/plan.md, Section 4, for A vs A′ and A vs B. `n` is the
number of rollouts per policy throughout. Writes:

- results/tables/rq2_comparisons.csv  gap, discordance, covariance split, n at 80% (Table 2)
- results/tables/rq2_power.csv        resampled and formula power against n (Figure 2)
- results/tables/rq2_null.csv         null control: A against itself
- results/tables/rq2_gap.csv          rollouts needed against the true gap (Figure 3)
- results/tables/rq2_video_keys.csv   the 10 discordant A vs A′ keys to re-run with video
- figures/rq2_power.png, figures/rq2_gap.png
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evalstats.analysis import NAMES, load_outcomes
from evalstats.plots import style
from evalstats.plots.style import INK_2, MUTED, SERIES
from evalstats.stats import power
from evalstats.stats.intervals import bootstrap_replicates, clopper_pearson, percentile_interval
from evalstats.stats.variance import components

REPO_ROOT = Path(__file__).resolve().parents[1]

COMPARISONS = {"A vs A′": ("smolvla_n10", "smolvla"), "A vs B": ("smolvla_n10", "pi05")}
# Rollouts per policy of the power curves. The plan's range is 50 to 5,000; it is extended
# down to 10 because both observed gaps are large (lab notebook, Step 7).
NS = [10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 300, 400, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000]  # fmt: skip
NULL_NS = [20, 50, 200, 1000, 5000]
GAPS = (0.01, 0.02, 0.05, 0.10)
DESIGN_COLORS = {"unpaired": SERIES[0], "paired": SERIES[1]}
DESIGN_NAMES = {"unpaired": "unpaired, Barnard", "paired": "paired, McNemar"}


def comparison_stats(y0: np.ndarray, y1: np.ndarray) -> dict[str, np.ndarray]:
    """Everything Table 2 reports, from two paired (..., task, state, seed) arrays."""
    axes = (-3, -2, -1)
    p0, p1 = y0.mean(axes), y1.mean(axes)
    b = ((y0 == 1) & (y1 == 0)).mean(axes)
    c = ((y0 == 0) & (y1 == 1)).mean(axes)
    s2u, s2p = power.sigma2_unpaired(p0, p1), power.sigma2_paired(b, c)
    cov = components(y0, y1)
    with np.errstate(divide="ignore"):
        n_u, n_p = power.formula_n(p1 - p0, s2u), power.formula_n(p1 - p0, s2p)
    out = {
        "rate_first": p0, "rate_second": p1, "gap": p1 - p0,
        "only_first_succeeds_b": b, "only_second_succeeds_c": c, "discordance_b_plus_c": b + c,
        "covariance": (y0 * y1).mean(axes) - p0 * p1,
        "cov_task": cov.task, "cov_init": cov.init, "cov_seed": cov.seed,
        "sigma2_unpaired": s2u, "sigma2_paired": s2p,
        "formula_n80_unpaired": n_u, "formula_n80_paired": n_p,
        "formula_saving": 1 - s2p / s2u,
    }  # fmt: skip
    # The saving 2 Cov / sigma2_unpaired, split by the level the two policies share.
    for level in ("task", "init", "seed"):
        out[f"saving_from_{level}"] = 2 * out[f"cov_{level}"] / s2u
    # Post hoc (lab notebook, Step 9): both designs run every task equally often, as most
    # protocols do, instead of drawing the task of each rollout. The variances are then
    # averages of the within-task variances, and the task part no longer counts in either.
    t0, t1 = y0.mean((-2, -1)), y1.mean((-2, -1))
    bt = ((y0 == 1) & (y1 == 0)).mean((-2, -1))
    ct = ((y0 == 0) & (y1 == 1)).mean((-2, -1))
    s2u_t = power.sigma2_unpaired(t0, t1).mean(-1)
    s2p_t = power.sigma2_paired(bt, ct).mean(-1)
    with np.errstate(divide="ignore"):
        out["sigma2_unpaired_tasks_balanced"] = s2u_t
        out["sigma2_paired_tasks_balanced"] = s2p_t
        out["formula_n80_unpaired_tasks_balanced"] = power.formula_n(p1 - p0, s2u_t)
        out["formula_n80_paired_tasks_balanced"] = power.formula_n(p1 - p0, s2p_t)
        out["formula_saving_tasks_balanced"] = 1 - s2p_t / s2u_t
    return out


def table2(y0, y1, rng, n_boot: int, curves: pd.DataFrame, name: str) -> list[dict]:
    point = comparison_stats(y0, y1)
    keys = list(point)
    reps = bootstrap_replicates(
        lambda a, b: np.stack([comparison_stats(a, b)[k] for k in keys], axis=-1),
        y0, y1, n_boot=n_boot, rng=rng,
    )  # fmt: skip
    low, high = percentile_interval(reps)
    rows = [
        {"comparison": name, "quantity": k, "estimate": float(point[k]),
         "ci_low": low[j], "ci_high": high[j], "source": "full grid, bootstrap over states"}
        for j, k in enumerate(keys)
    ]  # fmt: skip
    n80 = {}
    for design in power.DESIGNS:
        d = curves[(curves["comparison"] == name) & (curves["design"] == design)]
        n80[design] = power.n_at_power(d["n"], d["resampled_power"])
        rows.append({"comparison": name, "quantity": f"curve_n80_{design}", "estimate": n80[design],
                     "source": "resampled power curve"})  # fmt: skip
    rows.append({"comparison": name, "quantity": "curve_saving",
                 "estimate": 1 - n80["paired"] / n80["unpaired"], "source": "resampled power curve"})  # fmt: skip
    return rows


def power_curves(outcomes, rng, reps: int) -> pd.DataFrame:
    rows = []
    for name, (first, second) in COMPARISONS.items():
        y0, y1 = outcomes[first], outcomes[second]
        gap = y1.mean() - y0.mean()
        b, c = ((y0 == 1) & (y1 == 0)).mean(), ((y0 == 0) & (y1 == 1)).mean()
        sigma2 = {
            "unpaired": power.sigma2_unpaired(y0.mean(), y1.mean()),
            "paired": power.sigma2_paired(b, c),
        }
        for design, n in itertools.product(power.DESIGNS, NS):
            rows.append({
                "comparison": name, "design": design, "n": n, "reps": reps,
                "resampled_power": power.resampled_power(y0, y1, n, design, rng, reps=reps),
                "formula_power": float(power.formula_power(n, gap, sigma2[design])),
            })  # fmt: skip
    return pd.DataFrame(rows)


def null_control(y: np.ndarray, rng, reps: int) -> pd.DataFrame:
    """A against itself. Unpaired: independent keys. Paired: the same initial state with two
    different seeds, every ordered pair of seeds, so the two sides have the same rate."""
    first, second = np.array(list(itertools.permutations(range(y.shape[-1]), 2))).T
    pools = {"unpaired": (y, y), "paired": (y[..., first], y[..., second])}
    rows = []
    for design, n in itertools.product(power.DESIGNS, NULL_NS):
        rate = power.resampled_power(*pools[design], n, design, rng, reps=reps)
        low, high = clopper_pearson(round(rate * reps), reps)
        rows.append({
            "comparison": "A vs A (other seeds)", "design": design, "n": n, "reps": reps,
            "rejection_rate": rate, "ci95_low": low, "ci95_high": high,
            "compatible_with_5pct": low <= 0.05,
        })  # fmt: skip
    return pd.DataFrame(rows)


def gap_chart(y0: np.ndarray, y1: np.ndarray, gaps) -> pd.DataFrame:
    """Rollouts per policy for 80% power against the true gap, from the formula.

    Paired: the rate of discordant pairs b + c held at its value for this comparison, so that
    sigma2 = (b + c) - gap^2 (docs/plan.md, Section 4). Unpaired: the two policies at
    p - gap/2 and p + gap/2. The plan does not fix p, and the saving depends on it, so two
    bases are given: the rate of the first policy (the reference, used in the figure), and the
    average of the two rates. A third row (post hoc) runs every task equally often in the
    unpaired design, with task t at p_t - gap/2 and p_t + gap/2, p_t the first policy's rate on
    it. Its paired variance is given two ways: with the discordance held (unchanged when the gap
    is the same on every task), and with the within-task covariance of the two policies (shared
    state and seed, from this comparison) held instead: sigma2 = sigma2_unpaired - 2 Cov.
    """
    discordance = (y0 != y1).mean()
    gaps = np.asarray(gaps, dtype=float)
    s2p = discordance - gaps**2
    frames = []
    for base, p in (("first policy", y0.mean()), ("average of the two", (y0.mean() + y1.mean()) / 2)):  # fmt: skip
        s2u = power.sigma2_unpaired(p - gaps / 2, p + gaps / 2)
        frames.append(pd.DataFrame({
            "unpaired_base": base, "gap": gaps, "base_rate": p,
            "discordance_b_plus_c": discordance,
            "n80_unpaired": power.formula_n(gaps, s2u), "n80_paired": power.formula_n(gaps, s2p),
            "saving": 1 - s2p / s2u,
        }))  # fmt: skip
    pt = y0.mean((-2, -1))
    s2u = power.sigma2_unpaired(pt[None, :] - gaps[:, None] / 2, pt[None, :] + gaps[:, None] / 2).mean(1)  # fmt: skip
    cov = components(y0, y1)
    s2c = s2u - 2 * (cov.init + cov.seed)
    frames.append(pd.DataFrame({
        "unpaired_base": "first policy, tasks balanced", "gap": gaps, "base_rate": y0.mean(),
        "discordance_b_plus_c": discordance,
        "n80_unpaired": power.formula_n(gaps, s2u), "n80_paired": power.formula_n(gaps, s2p),
        "saving": 1 - s2p / s2u,
        "within_task_covariance": cov.init + cov.seed,
        "n80_paired_covariance_held": power.formula_n(gaps, s2c),
        "saving_covariance_held": 1 - s2c / s2u,
    }))  # fmt: skip
    return pd.concat(frames, ignore_index=True)


def video_keys(y0: np.ndarray, y1: np.ndarray, rng, per_direction: int = 5) -> pd.DataFrame:
    """Discordant keys of A vs A′, as many where only A succeeds as where only A′ does, each
    from a different task as far as possible."""
    rows = []
    for only, mask in (("A", (y0 == 1) & (y1 == 0)), ("A′", (y0 == 0) & (y1 == 1))):
        keys = np.argwhere(mask)
        keys = keys[rng.permutation(len(keys))]
        chosen, tasks = [], set()
        for key in keys:  # first a new task each time, then any task
            if key[0] not in tasks and len(chosen) < per_direction:
                chosen.append(key)
                tasks.add(key[0])
        for key in keys:
            if len(chosen) < per_direction and not any((key == k).all() for k in chosen):
                chosen.append(key)
        for t, i, s in sorted(map(tuple, chosen)):
            rows.append({"task_id": t, "init_state_id": i, "seed": s, "only_success": only,
                         "success_A": int(y0[t, i, s]), "success_A_prime": int(y1[t, i, s])})  # fmt: skip
    return pd.DataFrame(rows)


def figure_power(curves: pd.DataFrame, table: pd.DataFrame, path: Path) -> None:
    style.apply()
    fig, axes = plt.subplots(1, len(COMPARISONS), figsize=(10, 3.6), sharey=True)
    for ax, name in zip(axes, COMPARISONS, strict=True):
        est = table[table["comparison"] == name].set_index("quantity")["estimate"]
        for design in power.DESIGNS:
            d = curves[(curves["comparison"] == name) & (curves["design"] == design)]
            color = DESIGN_COLORS[design]
            ax.plot(d["n"], d["resampled_power"], color=color, linewidth=2,
                    label=f"{DESIGN_NAMES[design]}, resampled")  # fmt: skip
            ax.plot(d["n"], d["formula_power"], color=color, linewidth=1.2, linestyle=(0, (3, 2)),
                    label=f"{DESIGN_NAMES[design]}, formula")  # fmt: skip
            n80 = est[f"curve_n80_{design}"]
            ax.plot([n80], [0.8], marker="o", markersize=6, color=color, markeredgecolor="white",
                    markeredgewidth=1.5, zorder=4)  # fmt: skip
            # The smaller n is labelled above-left of its point, the larger one below-right.
            other = est[f"curve_n80_{'paired' if design == 'unpaired' else 'unpaired'}"]
            first = n80 <= other
            ax.annotate(f"n ≈ {n80:.0f}", (n80, 0.8), xytext=(-8, 7) if first else (8, -15),
                        ha="right" if first else "left", textcoords="offset points",
                        fontsize=8.5, color=INK_2)  # fmt: skip
        ax.axhline(0.8, color=MUTED, linewidth=0.8, linestyle=":")
        ax.set_xscale("log")
        ax.set_xlim(NS[0], 500)  # every curve is at 1 beyond; the table goes to 5,000
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("rollouts per policy, n")
        ax.set_title(f"{name}: gap {100 * est['gap']:+.1f} points, discordant pairs "
                     f"{100 * est['discordance_b_plus_c']:.1f}%", loc="left", fontsize=10)  # fmt: skip
        style.clean(ax, grid_axis="both")
    axes[0].set_ylabel("power (two-sided, α = 0.05)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.0))  # fmt: skip
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def figure_gap(y0: np.ndarray, y1: np.ndarray, marks: pd.DataFrame, path: Path) -> None:
    style.apply()
    fine = gap_chart(y0, y1, np.linspace(0.01, 0.10, 200))
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    # Solid: the plan's assumptions (random task per rollout, discordance held). Dotted, post
    # hoc: every task run equally often, the within-task covariance of A vs A′ held.
    balanced = "first policy, tasks balanced"
    lines = [
        ("unpaired", "first policy", "n80_unpaired", "-", "unpaired, random task per rollout"),
        ("paired", "first policy", "n80_paired", "-", "paired, discordance of A vs A′ held"),
        ("unpaired", balanced, "n80_unpaired", (0, (1, 1.5)), "unpaired, every task equally often"),
        ("paired", balanced, "n80_paired_covariance_held", (0, (1, 1.5)),
         "paired, every task equally often,\nwithin-task covariance of A vs A′ held"),
    ]  # fmt: skip
    for design, base, column, linestyle, label in lines:
        f = fine[fine["unpaired_base"] == base]
        m = marks[marks["unpaired_base"] == base]
        color, solid = DESIGN_COLORS[design], linestyle == "-"
        ax.plot(100 * f["gap"], f[column], color=color, linewidth=2 if solid else 1.6,
                linestyle=linestyle, label=label)  # fmt: skip
        if not solid:
            continue
        ax.plot(100 * m["gap"], m[column], "o", color=color, markersize=6,
                markeredgecolor="white", markeredgewidth=1.5)  # fmt: skip
        for g, n in zip(m["gap"], m[column], strict=True):
            above = design == "unpaired"
            ax.annotate(f"{n:,.0f}", (100 * g, n), xytext=(6, 5) if above else (-6, -13),
                        ha="left" if above else "right", textcoords="offset points",
                        fontsize=8.5, color=INK_2)  # fmt: skip
    ax.set_yscale("log")
    ax.set_ylim(80, 60_000)
    ax.set_xlim(0.5, 10.6)
    ax.set_xticks(100 * np.array(GAPS), [f"{100 * g:.0f}" for g in GAPS])
    ax.set_xlabel("true gap between the two policies (points)")
    ax.set_ylabel("rollouts per policy for 80% power")
    ax.set_title("Rollouts needed against the gap, from the formula", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    m = marks[marks["unpaired_base"] == "first policy"]
    p, disc = m["base_rate"].iloc[0], m["discordance_b_plus_c"].iloc[0]
    fig.text(0.01, 0.01,
             f"Solid: the plan's assumptions, rates {100 * p:.0f}% ± gap/2 (A's rate), discordant "
             f"pairs held at {100 * disc:.1f}% of keys.\nDotted (post hoc): each task at A's rate "
             "on it ± gap/2. Two-sided α = 0.05; exact tests need a little more at small n.",
             fontsize=7.5, color=INK_2, va="bottom")  # fmt: skip
    style.clean(ax, grid_axis="both")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--grid", type=Path, default=REPO_ROOT / "configs/grid_main.yaml")
    parser.add_argument("--policies-config", type=Path, default=REPO_ROOT / "configs/policies.yaml")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "results/raw")
    parser.add_argument("--tables-dir", type=Path, default=REPO_ROOT / "results/tables")
    parser.add_argument("--figures-dir", type=Path, default=REPO_ROOT / "figures")
    parser.add_argument("--reps", type=int, default=5000, help="evaluations per point of a curve")
    parser.add_argument("--null-reps", type=int, default=20_000)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    outcomes = load_outcomes(args.grid, args.policies_config, args.raw_dir)
    # One random stream per part, so that a part does not depend on the others.
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(args.seed).spawn(4)]

    curves = power_curves(outcomes, rngs[0], args.reps)
    rows = []
    for name, (first, second) in COMPARISONS.items():
        rows += table2(outcomes[first], outcomes[second], rngs[1], args.n_boot, curves, name)
    table = pd.DataFrame(rows)
    a, a_prime = outcomes["smolvla_n10"], outcomes["smolvla"]
    tables = {
        "rq2_comparisons": table,
        "rq2_power": curves,
        "rq2_null": null_control(a, rngs[2], args.null_reps),
        "rq2_gap": gap_chart(a, a_prime, GAPS),
        "rq2_video_keys": video_keys(a, a_prime, rngs[3]),
    }
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    for stem, df in tables.items():
        df.to_csv(args.tables_dir / f"{stem}.csv", index=False, float_format="%.6g")
        print(f"\n{stem}\n{df.round(4).to_string(index=False)}")

    figure_power(curves, table, args.figures_dir / "rq2_power.png")
    figure_gap(a, a_prime, tables["rq2_gap"], args.figures_dir / "rq2_gap.png")
    gaps = tables["rq2_gap"]
    two = gaps[(gaps["unpaired_base"] == "first policy") & (gaps["gap"] == 0.02)].iloc[0]
    print(
        f"\nDetecting a 2-point gap at 80% power on libero_spatial takes about "
        f"{two['n80_unpaired']:,.0f} rollouts per policy unpaired and {two['n80_paired']:,.0f} "
        f"paired ({NAMES['smolvla_n10']} vs {NAMES['smolvla']} discordance)."
    )


if __name__ == "__main__":
    main()
