#!/usr/bin/env python
"""Step 3 sanity check: success of each policy on the sanity grid, next to a published rate.

    uv run scripts/analyze_sanity.py

Reads the keys of configs/grid_sanity.yaml from results/raw/main/ and writes
results/tables/sanity_check.csv: one row per task and one row for the whole suite, with a 95%
Clopper-Pearson interval per policy. No test between policies is made here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from statsmodels.stats.proportion import proportion_confint

from evalstats.harness.grid import load_grid, load_policies
from evalstats.harness.schema import KEY_COLUMNS
from evalstats.harness.store import load_episodes

REPO_ROOT = Path(__file__).resolve().parents[1]


def summarize(label: str, df: pd.DataFrame) -> dict:
    n, k = len(df), int(df["success"].sum())
    low, high = proportion_confint(k, n, alpha=0.05, method="beta")
    return {
        "task": label,
        "n_episodes": n,
        "n_success": k,
        "success_rate": k / n,
        "ci95_low": low,
        "ci95_high": high,
        "mean_episode_len": df["episode_len"].mean(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--grid", type=Path, default=REPO_ROOT / "configs/grid_sanity.yaml")
    parser.add_argument("--policies-config", type=Path, default=REPO_ROOT / "configs/policies.yaml")
    parser.add_argument("--published", type=Path, default=REPO_ROOT / "configs/published.yaml")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "results/raw")
    parser.add_argument("--tables-dir", type=Path, default=REPO_ROOT / "results/tables")
    args = parser.parse_args()

    grid = load_grid(args.grid)
    specs = load_policies(args.policies_config)
    with open(args.published) as f:
        published = yaml.safe_load(f)
    episodes = load_episodes(args.raw_dir / grid.results_dir).set_index(KEY_COLUMNS)

    tables = []
    for name in grid.policies:
        keys = [k.as_tuple() for k in grid.keys(specs[name])]
        missing = [k for k in keys if k not in episodes.index]
        if missing:
            raise SystemExit(f"{name}: {len(missing)} of {len(keys)} sanity episodes are missing")
        df = episodes.loc[keys].reset_index()
        rows = [summarize(f"task {t}", d) for t, d in df.groupby("task_id")]
        rows.append(summarize("all", df))
        table = pd.DataFrame(rows)
        table.insert(0, "policy", name)
        table.insert(1, "policy_variant", specs[name].variant)
        ref = published.get(name)
        table["published_rate"] = ref["success_rate"] if ref else float("nan")
        table["published_source"] = ref["source"] if ref else ""
        tables.append(table)
        if ref:
            print(
                f"{name}: published {ref['success_rate']:.0%}, {ref['source']}\n  {ref['protocol']}"
            )

    out = pd.concat(tables, ignore_index=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.tables_dir / "sanity_check.csv", index=False, float_format="%.4g")
    print(out.drop(columns=["published_source"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
