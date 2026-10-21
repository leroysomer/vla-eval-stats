#!/usr/bin/env python
"""Replay a shard of the main grid in the order the grid ran it, in one environment. GPU.

    uv run scripts/replay_shard.py --policy smolvla --task 4 --until 22 1

Takes the episodes of one policy and one task written by one grid run, in the order of their
timestamps, up to and including the key (initial state, seed) given by --until. It runs them
back to back in a single environment, as the grid did, and compares each with its stored row.
This tests whether an episode depends on the episodes before it in its shard (lab notebook,
2026-10-19). Writes results/tables/replay_shard_<policy>_task<task>.csv; stores nothing.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--policy", required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--until", type=int, nargs=2, metavar=("STATE", "SEED"), required=True)
    parser.add_argument("--grid", type=Path, default=REPO_ROOT / "configs/grid_main.yaml")
    parser.add_argument("--policies-config", type=Path, default=REPO_ROOT / "configs/policies.yaml")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "results/raw")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("replay_shard")

    from evalstats.harness.grid import load_grid, load_policies
    from evalstats.harness.schema import EpisodeKey
    from evalstats.harness.store import load_episodes

    grid = load_grid(args.grid)
    spec = load_policies(args.policies_config)[args.policy]
    rows = load_episodes(args.raw_dir / grid.results_dir)
    rows = rows[
        (rows["policy_id"] == spec.policy_id)
        & (rows["policy_variant"] == spec.variant)
        & (rows["task_id"] == args.task)
    ]
    state, seed = args.until
    target = rows[(rows["init_state_id"] == state) & (rows["seed"] == seed)]
    if len(target) != 1:
        raise SystemExit(f"no stored episode for initial state {state}, seed {seed}")
    run_id = target["run_id"].iloc[0]
    shard = rows[rows["run_id"] == run_id].sort_values("timestamp").reset_index(drop=True)
    if shard["timestamp"].duplicated().any():
        raise SystemExit("ties in the timestamps: the order of the grid run is ambiguous")
    shard = shard.iloc[: shard.index[(shard["init_state_id"] == state) & (shard["seed"] == seed)][0] + 1]  # fmt: skip
    keys = [
        EpisodeKey(spec.policy_id, spec.variant, r.suite, r.task_id, r.init_state_id, r.seed)
        for r in shard.itertuples()
    ]
    logger.info(
        "%d episodes of run %s, in its order, up to (%d, %d)", len(keys), run_id, state, seed
    )
    if args.dry_run:
        return

    from evalstats.harness.runner import PolicyRunner

    replayed = PolicyRunner(spec).rerun_in_one_env(keys)
    shard["replay_success"] = [r["success"] for r in replayed]
    shard["replay_len"] = [r["episode_len"] for r in replayed]
    shard["same"] = (shard["success"] == shard["replay_success"]) & (
        shard["episode_len"] == shard["replay_len"]
    )
    out = shard[["init_state_id", "seed", "success", "replay_success", "episode_len", "replay_len", "same"]]  # fmt: skip
    path = REPO_ROOT / f"results/tables/replay_shard_{args.policy}_task{args.task}.csv"
    out.rename(columns={"success": "stored_success", "episode_len": "stored_len"}).to_csv(path, index=False)  # fmt: skip
    logger.info("%d of %d identical; differences:\n%s", out["same"].sum(), len(out),
                out[~out["same"]].to_string(index=False))  # fmt: skip


if __name__ == "__main__":
    main()
