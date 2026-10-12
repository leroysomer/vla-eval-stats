#!/usr/bin/env python
"""Run a grid of LIBERO episodes, one parquet row per episode.

    uv run scripts/run_grid.py --grid configs/grid_throughput.yaml

Rows go to results/raw/<results_dir of the grid, by default its name>/, one file per (policy variant, suite, task). The command is
idempotent: episodes already stored are skipped, so rerun it after a crash or a kill.
"""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--policies-config", type=Path, default=REPO_ROOT / "configs/policies.yaml")
    parser.add_argument(
        "--out-dir", type=Path, help="default: results/raw/<results_dir of the grid>"
    )
    parser.add_argument("--policies", nargs="+", help="run only these policies of the grid")
    parser.add_argument("--dry-run", action="store_true", help="only count what is left to run")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("run_grid")

    from evalstats.harness.grid import load_grid, load_policies
    from evalstats.harness.store import EpisodeStore

    grid = load_grid(args.grid)
    specs = load_policies(args.policies_config)
    names = args.policies or grid.policies
    unknown = [n for n in names if n not in specs or n not in grid.policies]
    if unknown:
        raise SystemExit(f"policies not in the grid or not in {args.policies_config}: {unknown}")

    store = EpisodeStore(args.out_dir or REPO_ROOT / "results/raw" / grid.results_dir)
    for name in names:
        keys = grid.keys(specs[name])
        n_pending = sum(k not in store for k in keys)
        if args.dry_run or n_pending == 0:
            logger.info("%s: %d planned, %d to run", name, len(keys), n_pending)
            continue

        # Imported late: loading LeRobot and torch takes a while and a dry run needs neither.
        import torch

        from evalstats.harness.runner import PolicyRunner

        runner = PolicyRunner(specs[name])
        summary = runner.run(keys, store)
        logger.info("%s: %s", name, summary)
        del runner
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
