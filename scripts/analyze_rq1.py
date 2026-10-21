#!/usr/bin/env python
"""Step 6, RQ1: how uncertain is one success rate, and why? (docs/plan.md, Section 3)

    uv run scripts/analyze_rq1.py

CPU only. Reads the main grid from results/raw/main/ and, for each policy, estimates the success
rate with a Clopper-Pearson and a bootstrap interval, splits p(1 - p) into its task, initial
state and seed components, and predicts the uncertainty of the two protocols in common use (10
and 50 initial states per task, one seed each). Every interval comes from one bootstrap that
resamples initial states within each task. Writes:

- results/tables/rq1_rates.csv       rate, both intervals, design effect of the grid
- results/tables/rq1_components.csv  the three components and their shares
- results/tables/rq1_protocols.csv   design effect and half-width Y of each protocol (Table 1)
- figures/rq1.png                    Figure 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from evalstats.analysis import LABELS, load_outcomes
from evalstats.plots import style
from evalstats.plots.style import GRID, INK, INK_2, MUTED, SURFACE
from evalstats.stats.intervals import bootstrap_replicates, clopper_pearson, percentile_interval
from evalstats.stats.variance import (
    Components,
    components,
    design_effect,
    rate_variance,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

LEVELS = ("task", "init", "seed")
LEVEL_NAMES = {"task": "between tasks", "init": "between initial states", "seed": "between seeds"}
# Protocols: initial states per task, seeds per state. The first two are the ones in common use.
PROTOCOLS = {"10 per task": (10, 1), "50 per task": (50, 1), "this grid": (50, 4)}
Z = stats.norm.ppf(0.975)

LEVEL_COLORS = dict(zip(LEVELS, style.SERIES, strict=True))


def _statistics(y: np.ndarray) -> np.ndarray:
    """Rate and the three components, per bootstrap replicate: shape (replicates, 4)."""
    c = components(y)
    return np.stack([y.mean(axis=(-3, -2, -1)), *c], axis=-1)


def analyze(y: np.ndarray, rng: np.random.Generator, n_boot: int):
    """Point estimates and bootstrap replicates of everything RQ1 reports for one policy."""
    n_tasks = y.shape[0]
    reps = bootstrap_replicates(_statistics, y, n_boot=n_boot, rng=rng)

    def derived(s: np.ndarray) -> dict[str, np.ndarray]:
        c = Components(s[..., 1], s[..., 2], s[..., 3])
        out = {"rate": s[..., 0], "var_total": c.total}  # 0-d arrays for the point estimates
        out |= {f"var_{k}": v for k, v in zip(LEVELS, c, strict=True)}
        out |= {f"share_{k}": v / c.total for k, v in zip(LEVELS, c, strict=True)}
        for name, (k, m) in PROTOCOLS.items():
            out[f"deff|{name}"] = design_effect(c, m)
            out[f"y|{name}"] = Z * np.sqrt(rate_variance(c, n_tasks, k, m))
        return out

    point = {key: float(value) for key, value in derived(_statistics(y)).items()}
    return point, derived(reps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--grid", type=Path, default=REPO_ROOT / "configs/grid_main.yaml")
    parser.add_argument("--policies-config", type=Path, default=REPO_ROOT / "configs/policies.yaml")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "results/raw")
    parser.add_argument("--tables-dir", type=Path, default=REPO_ROOT / "results/tables")
    parser.add_argument("--figure", type=Path, default=REPO_ROOT / "figures/rq1.png")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    outcomes = load_outcomes(args.grid, args.policies_config, args.raw_dir)
    rng = np.random.default_rng(args.seed)

    rates, comps, protocols, sentences = [], [], [], []
    for name, y in outcomes.items():
        n, k = y.size, int(y.sum())
        est, reps = analyze(y, rng, args.n_boot)
        boot = percentile_interval(reps["rate"])
        cp = clopper_pearson(k, n)
        deff = float(est["deff|this grid"])
        rates.append({
            "policy": name, "label": LABELS[name], "n_episodes": n, "n_success": k,
            "rate": k / n, "cp_low": cp[0], "cp_high": cp[1],
            "boot_low": boot[0], "boot_high": boot[1],
            "cp_width": cp[1] - cp[0], "boot_width": boot[1] - boot[0],
            "width_ratio": (boot[1] - boot[0]) / (cp[1] - cp[0]),
            "design_effect": deff, "sqrt_design_effect": np.sqrt(deff), "n_boot": args.n_boot,
        })  # fmt: skip
        for level in (*LEVELS, "total"):
            low, high = percentile_interval(reps[f"var_{level}"])
            row = {
                "policy": name, "label": LABELS[name], "level": level,
                "estimate": est[f"var_{level}"], "ci_low": low, "ci_high": high,
            }  # fmt: skip
            if level != "total":
                s_low, s_high = percentile_interval(reps[f"share_{level}"])
                row |= {"share": est[f"share_{level}"], "share_low": s_low, "share_high": s_high}
            comps.append(row)
        for protocol, (states, seeds) in PROTOCOLS.items():
            d_low, d_high = percentile_interval(reps[f"deff|{protocol}"])
            y_low, y_high = percentile_interval(reps[f"y|{protocol}"])
            n_protocol = y.shape[0] * states * seeds
            protocols.append({
                "policy": name, "label": LABELS[name], "protocol": protocol,
                "states_per_task": states, "seeds_per_state": seeds, "n_episodes": n_protocol,
                "design_effect": est[f"deff|{protocol}"], "deff_low": d_low, "deff_high": d_high,
                "half_width": est[f"y|{protocol}"], "half_width_low": y_low,
                "half_width_high": y_high,
                "binomial_half_width": Z * np.sqrt(k / n * (1 - k / n) / n_protocol),
            })  # fmt: skip
        ten = protocols[-3]
        sentences.append(
            f"{LABELS[name]}: a reported {100 * k / n:.0f}% on libero_spatial at 10 episodes per "
            f"task is really {100 * k / n:.0f} ± {100 * ten['half_width']:.1f} points "
            f"(binomial formula: ± {100 * ten['binomial_half_width']:.1f})"
        )

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    tables = {"rq1_rates": rates, "rq1_components": comps, "rq1_protocols": protocols}
    pd.set_option("display.width", 220, "display.max_columns", 30)
    for stem, rows in tables.items():
        df = pd.DataFrame(rows)
        df.to_csv(args.tables_dir / f"{stem}.csv", index=False, float_format="%.6g")
        print(f"\n{stem}\n{df.drop(columns='label').round(4).to_string(index=False)}")
    print("\n" + "\n".join(sentences))
    figure(pd.DataFrame(rates), pd.DataFrame(comps), args.figure)
    print(f"\nwrote {args.figure}")


def figure(rates: pd.DataFrame, comps: pd.DataFrame, path: Path) -> None:
    style.apply()
    plt.rcParams["ytick.color"] = INK  # the policy names
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(10, 3.4), sharey=True, gridspec_kw={"width_ratios": [1.1, 1]}
    )
    fig.patch.set_facecolor(SURFACE)
    rows = list(rates["policy"])[::-1]  # first policy on top
    ypos = np.arange(len(rows))

    # Panel a: shares of p(1 - p), stacked. Negative estimates (possible for an unbiased
    # estimator) would be drawn as zero; the table keeps the signed values.
    shares = comps[comps["level"] != "total"].pivot(index="policy", columns="level", values="share")
    start = np.zeros(len(rows))
    for level in LEVELS:
        width = np.clip(shares.loc[rows, level].to_numpy(), 0, None)
        left.barh(ypos, width, left=start, height=0.5, color=LEVEL_COLORS[level],
                  edgecolor=SURFACE, linewidth=2, label=LEVEL_NAMES[level])  # fmt: skip
        for yp, s, w in zip(ypos, start, width, strict=True):
            if w >= 0.07:
                left.text(s + w / 2, yp, f"{w:.0%}", ha="center", va="center", fontsize=8.5,
                          color="white" if level != "seed" else INK)  # fmt: skip
        start += width
    left.set_xlim(0, 1)
    left.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    left.set_xlabel("share of the episode variance p(1 − p)")
    left.set_yticks(ypos, [rates.set_index("policy").loc[p, "label"] for p in rows])
    left.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
                fontsize=8.5, handlelength=1, columnspacing=1.2)  # fmt: skip
    left.set_title("a   Where the variance of one episode comes from", loc="left",
                   fontsize=10, pad=24, color=INK)  # fmt: skip

    # Panel b: rate with the naive and the cluster-aware interval.
    r = rates.set_index("policy").loc[rows]
    off = 0.13
    right.hlines(ypos + off, 100 * r["cp_low"], 100 * r["cp_high"], color=MUTED, linewidth=2,
                 label="Clopper-Pearson (episodes independent)")  # fmt: skip
    right.hlines(ypos - off, 100 * r["boot_low"], 100 * r["boot_high"], color=INK, linewidth=2,
                 label="bootstrap over initial states")  # fmt: skip
    right.scatter(100 * r["rate"], ypos, s=36, color=INK, zorder=3, edgecolors=SURFACE,
                  linewidths=2)  # fmt: skip
    for yp, (_, row) in zip(ypos, r.iterrows(), strict=True):
        right.text(100 * row["boot_high"] + 0.8, yp - off, f"{100 * row['rate']:.1f}%",
                   va="center", fontsize=8.5, color=INK_2)  # fmt: skip
    lo = 100 * min(r["boot_low"].min(), r["cp_low"].min())
    right.set_xlim(np.floor(lo / 5) * 5 - 2, 100)
    right.set_xlabel("success rate on libero_spatial, 2,000 episodes (%), 95% intervals")
    # Inside the axes, in the corner left empty by the highest rate.
    right.legend(loc="lower left", ncol=1, frameon=False, fontsize=8.5, handlelength=1.5)
    right.set_title("b   Success rate and two intervals", loc="left", fontsize=10, pad=24,
                    color=INK)  # fmt: skip
    right.grid(axis="x", color=GRID, linewidth=0.8)
    right.set_axisbelow(True)

    for ax in (left, right):
        ax.set_facecolor(SURFACE)
        ax.tick_params(length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_ylim(-0.6, len(rows) - 0.4)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
