#!/usr/bin/env python
"""Throughput of each policy and projected GPU-hours of the planned runs (Step 2).

    uv run scripts/analyze_throughput.py

Reads results/raw/throughput/ and writes results/tables/throughput.csv and
results/tables/cost_projection.csv.

Cost model, per policy: an episode of L steps takes `fixed_s + s_per_step * L` seconds (least
squares on the timed episodes). The first episode of each process is left out of the fit and
counted in `startup_s` together with the model loading time: it pays for one-off costs such as
torch.compile. Two projections follow:

  - observed mix: episodes are as long, on average, as in the timing run (same suite only);
  - worst case: every episode runs to the step limit of its suite, i.e. the policy always fails.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from evalstats.harness.grid import load_policies
from evalstats.harness.store import load_episodes

REPO_ROOT = Path(__file__).resolve().parents[1]


def suite_max_steps() -> dict[str, int]:
    from lerobot.envs.libero import TASK_SUITE_MAX_STEPS

    return dict(TASK_SUITE_MAX_STEPS)


def throughput_table(episodes: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (policy_id, variant, suite), df in episodes.groupby(
        ["policy_id", "policy_variant", "suite"]
    ):
        df = df.sort_values("timestamp")
        is_first = ~df.duplicated("run_id")
        steady = df[~is_first]
        s_per_step, fixed_s = np.polyfit(steady["episode_len"], steady["wall_time_s"], 1)
        mean_len = steady["episode_len"].mean()
        mean_episode_s = steady["wall_time_s"].mean()
        # Extra time of a process's first episode over what the fit predicts for its length.
        first = df[is_first]
        warmup_s = (first["wall_time_s"] - (fixed_s + s_per_step * first["episode_len"])).max()

        # Runs that ended normally also tell the time spent outside episodes (env creation, I/O).
        r = runs[(runs["policy_id"] == policy_id) & (runs["policy_variant"] == variant)]
        in_episodes = df.groupby("run_id")["wall_time_s"].sum().reindex(r["run_id"]).to_numpy()
        overhead_s = ((r["total_wall_s"].to_numpy() - in_episodes) / r["n_run"].to_numpy()).mean()
        model_load_s = r["model_load_s"].mean()

        rows.append(
            {
                "policy_id": policy_id,
                "policy_variant": variant,
                "suite": suite,
                "n_episodes": len(df),
                "n_success": int(df["success"].sum()),
                "mean_episode_len": mean_len,
                "mean_episode_s": mean_episode_s,
                "fixed_s": fixed_s,
                "s_per_step": s_per_step,
                "overhead_s_per_episode": overhead_s,
                "model_load_s": model_load_s,
                "warmup_s": warmup_s,
                "startup_s": model_load_s + warmup_s,
                "peak_gpu_mem_mib": r["peak_gpu_mem_mib"].max(),
                "episodes_per_gpu_hour": 3600 / (mean_episode_s + overhead_s),
            }
        )
    return pd.DataFrame(rows)


def cost_table(throughput: pd.DataFrame, scenarios: dict, policies: dict) -> pd.DataFrame:
    max_steps = suite_max_steps()
    rows = []
    for name, scenario in scenarios.items():
        for policy in scenario["policies"]:
            spec = policies[policy]
            t = throughput[
                (throughput["policy_id"] == spec.policy_id)
                & (throughput["policy_variant"] == spec.variant)
            ]
            if t.empty:
                continue
            t = t.iloc[0]
            n = scenario["episodes_per_policy"]
            worst_s = t["fixed_s"] + t["s_per_step"] * max_steps[scenario["suite"]]
            worst_h = (t["startup_s"] + n * (worst_s + t["overhead_s_per_episode"])) / 3600
            observed_h = np.nan
            if scenario["suite"] == t["suite"]:
                observed_s = t["mean_episode_s"] + t["overhead_s_per_episode"]
                observed_h = (t["startup_s"] + n * observed_s) / 3600
            rows.append(
                {
                    "scenario": name,
                    "description": scenario["description"],
                    "policy": policy,
                    "suite": scenario["suite"],
                    "n_episodes": n,
                    "gpu_hours_observed_mix": observed_h,
                    "gpu_hours_worst_case": worst_h,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "results/raw/throughput")
    parser.add_argument("--scenarios", type=Path, default=REPO_ROOT / "configs/cost_scenarios.yaml")
    parser.add_argument("--policies-config", type=Path, default=REPO_ROOT / "configs/policies.yaml")
    parser.add_argument("--tables-dir", type=Path, default=REPO_ROOT / "results/tables")
    args = parser.parse_args()

    episodes = load_episodes(args.raw_dir)
    # One run log per machine (runs.<host>.jsonl), plus runs.jsonl from before that change.
    lines = [
        line
        for path in sorted(args.raw_dir.glob("runs*.jsonl"))
        for line in path.read_text().splitlines()
    ]
    runs = pd.DataFrame([json.loads(line) for line in lines])
    runs = runs[runs["n_run"] > 0]
    with open(args.scenarios) as f:
        scenarios = yaml.safe_load(f)

    throughput = throughput_table(episodes, runs)
    cost = cost_table(throughput, scenarios, load_policies(args.policies_config))

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    throughput.to_csv(args.tables_dir / "throughput.csv", index=False, float_format="%.4g")
    cost.to_csv(args.tables_dir / "cost_projection.csv", index=False, float_format="%.3g")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(throughput.drop(columns=["policy_id"]).round(3).T.to_string(), "\n")
    print(cost.drop(columns=["description"]).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
