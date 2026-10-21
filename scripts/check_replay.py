#!/usr/bin/env python
"""Do stored episodes of the main grid replay exactly? Uses the GPU, about 15 minutes.

    uv run scripts/check_replay.py              # run it, on an idle GPU
    uv run scripts/check_replay.py --dry-run    # only list the episodes it would run

Re-runs, without video and without writing to the store:
- `--per-policy` stored keys of each policy, drawn at random with a fixed seed;
- the key that did not replay in Step 7 (A′, task 4, initial state 22, seed 1), `--repeats` times.
Writes results/tables/replay_check.csv: stored and replayed success and length, and whether
the GPU had other processes on it when each episode started.
"""

from __future__ import annotations

import argparse
import logging
import socket
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
MISMATCH = ("smolvla", 4, 22, 1)  # policy, task, initial state, seed (lab notebook, 2026-10-19)


def gpu_processes() -> int:
    """Processes on the GPU, this one included once it has loaded a model."""
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    return len(out.stdout.split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--per-policy", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid", type=Path, default=REPO_ROOT / "configs/grid_main.yaml")
    parser.add_argument("--policies-config", type=Path, default=REPO_ROOT / "configs/policies.yaml")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "results/raw")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results/tables/replay_check.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("check_replay")

    from evalstats.analysis import NAMES
    from evalstats.harness.grid import load_grid, load_policies
    from evalstats.harness.schema import KEY_COLUMNS
    from evalstats.harness.store import load_episodes

    grid = load_grid(args.grid)
    specs = load_policies(args.policies_config)
    stored = load_episodes(args.raw_dir / grid.results_dir).set_index(KEY_COLUMNS)
    rng = np.random.default_rng(args.seed)

    plan = []  # (policy, key, purpose)
    for name in grid.policies:
        keys = grid.keys(specs[name])
        for j in rng.choice(len(keys), size=args.per_policy, replace=False):
            plan.append((name, keys[j], "random stored key"))
        if name == MISMATCH[0]:
            (key,) = [k for k in keys if (k.task_id, k.init_state_id, k.seed) == MISMATCH[1:]]
            plan += [(name, key, "Step 7 mismatch")] * args.repeats
    if missing := [k for _, k, _ in plan if k.as_tuple() not in stored.index]:
        raise SystemExit(f"not stored: {missing[:3]}")
    counts = pd.Series([n for n, _, _ in plan]).value_counts().to_dict()
    logger.info("%d episodes planned: %s", len(plan), counts)
    if args.dry_run:
        return

    import gc

    import torch

    from evalstats.harness.runner import PolicyRunner

    rows = []
    for name in grid.policies:
        runner = PolicyRunner(specs[name])
        for policy, key, purpose in plan:
            if policy != name:
                continue
            others = gpu_processes() - 1  # minus this process
            new, _ = runner.rerun(key)
            old = stored.loc[key.as_tuple()]
            same = (int(old["success"]), int(old["episode_len"])) == (new["success"], new["episode_len"])  # fmt: skip
            rows.append({
                "policy": name, "name": NAMES[name], "purpose": purpose, "task_id": key.task_id,
                "init_state_id": key.init_state_id, "seed": key.seed,
                "stored_success": int(old["success"]), "replay_success": new["success"],
                "stored_len": int(old["episode_len"]), "replay_len": new["episode_len"],
                "same": same, "stored_run_id": old["run_id"], "host": socket.gethostname(),
                "other_gpu_processes": others,
            })  # fmt: skip
            logger.info(
                "%s %s task %d init %d seed %d: %s (stored %d/%d, replay %d/%d), %d other GPU processes",
                name, purpose, key.task_id, key.init_state_id, key.seed,
                "same" if same else "DIFFERENT", old["success"], old["episode_len"],
                new["success"], new["episode_len"], others,
            )  # fmt: skip
        del runner
        gc.collect()
        torch.cuda.empty_cache()

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    summary = out.groupby(["name", "purpose"])["same"].agg(["sum", "count"])
    logger.info("identical replays:\n%s", summary.to_string())


if __name__ == "__main__":
    main()
