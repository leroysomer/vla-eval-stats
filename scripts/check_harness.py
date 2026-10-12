#!/usr/bin/env python
"""Two checks of the harness on the Step 2 timing episodes.

    uv run scripts/check_harness.py

1. Reproducibility: results/raw/throughput_rerun/ holds a second run of the keys of
   results/raw/throughput/. Count the episodes that came out identical.
   -> results/tables/reproducibility.csv
2. Agreement with stock `lerobot-eval`, whose episode i of a task starts from initial state i:
   results/raw/stock_lerobot_eval/<policy>.json is its eval_info.json for 2 episodes per task,
   i.e. the same initial states as the timing grid, with other seeds.
   -> results/tables/stock_comparison.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evalstats.harness.grid import load_policies
from evalstats.harness.schema import KEY_COLUMNS
from evalstats.harness.store import load_episodes

REPO_ROOT = Path(__file__).resolve().parents[1]
# Settings under which each stock lerobot-eval run was made (the checkpoint defaults).
STOCK_VARIANT = "default"


def reproducibility(first: pd.DataFrame, rerun: pd.DataFrame) -> pd.DataFrame:
    both = first.merge(rerun, on=KEY_COLUMNS, suffixes=("", "_rerun"), validate="one_to_one")
    both["same_success"] = both["success"] == both["success_rerun"]
    both["identical"] = both["same_success"] & (both["episode_len"] == both["episode_len_rerun"])
    return (
        both.groupby(["policy_id", "policy_variant"])
        .agg(
            n_episodes=("identical", "size"),
            n_identical=("identical", "sum"),
            n_same_success=("same_success", "sum"),
            n_success=("success", "sum"),
            n_success_rerun=("success_rerun", "sum"),
        )
        .reset_index()
    )


def stock_comparison(episodes: pd.DataFrame, stock_dir: Path, policies: dict) -> pd.DataFrame:
    rows = []
    for path in sorted(stock_dir.glob("*.json")):
        policy_id = policies[path.stem].policy_id
        with open(path) as f:
            info = json.load(f)
        stock = pd.DataFrame(
            [
                {
                    "suite": t["task_group"],
                    "task_id": t["task_id"],
                    "init_state_id": i,
                    "stock": int(s),
                }
                for t in info["per_task"]
                for i, s in enumerate(t["metrics"]["successes"])
            ]
        )
        ours = episodes[episodes["policy_id"] == policy_id]
        for variant, df in ours.groupby("policy_variant"):
            both = df.merge(stock, on=["suite", "task_id", "init_state_id"], validate="one_to_one")
            rows.append(
                {
                    "policy_id": policy_id,
                    "harness_variant": variant,
                    "stock_variant": STOCK_VARIANT,
                    "n_episodes": len(both),
                    "n_success_harness": int(both["success"].sum()),
                    "n_success_stock": int(both["stock"].sum()),
                    "n_same_outcome": int((both["success"] == both["stock"]).sum()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "results/raw")
    parser.add_argument("--policies-config", type=Path, default=REPO_ROOT / "configs/policies.yaml")
    parser.add_argument("--tables-dir", type=Path, default=REPO_ROOT / "results/tables")
    args = parser.parse_args()

    episodes = load_episodes(args.raw_dir / "throughput")
    repro = reproducibility(episodes, load_episodes(args.raw_dir / "throughput_rerun"))
    stock = stock_comparison(
        episodes, args.raw_dir / "stock_lerobot_eval", load_policies(args.policies_config)
    )
    repro.to_csv(args.tables_dir / "reproducibility.csv", index=False)
    stock.to_csv(args.tables_dir / "stock_comparison.csv", index=False)
    print(repro.to_string(index=False), "\n")
    print(stock.to_string(index=False))


if __name__ == "__main__":
    main()
