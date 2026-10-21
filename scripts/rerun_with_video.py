#!/usr/bin/env python
"""Re-run episodes of the main grid with video (Step 7, failure modes). Uses the GPU.

    uv run scripts/rerun_with_video.py              # the 10 discordant A vs A′ keys, both policies
    uv run scripts/rerun_with_video.py --dry-run    # only check the keys against the stored rows

Reads (task, initial state, seed) keys from results/tables/rq2_video_keys.csv, re-runs each with
each policy, and writes one video per episode to figures/videos/. Nothing is added to the
episode store. Each re-run is compared with the stored row of the same key in
results/tables/rq2_video_rerun.csv, which is also a reproducibility check across machines: the
stored rows were run on other machines.
"""

from __future__ import annotations

import argparse
import logging
import socket
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
FPS = 80  # LeRobot's LiberoEnv.metadata["render_fps"], as in lerobot-eval videos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--keys", type=Path, default=REPO_ROOT / "results/tables/rq2_video_keys.csv"
    )
    parser.add_argument("--policies", nargs="+", default=["smolvla_n10", "smolvla"])
    parser.add_argument("--grid", type=Path, default=REPO_ROOT / "configs/grid_main.yaml")
    parser.add_argument("--policies-config", type=Path, default=REPO_ROOT / "configs/policies.yaml")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "results/raw")
    parser.add_argument("--videos-dir", type=Path, default=REPO_ROOT / "figures/videos")
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "results/tables/rq2_video_rerun.csv"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("rerun_with_video")

    from evalstats.analysis import NAMES
    from evalstats.harness.grid import load_grid, load_policies
    from evalstats.harness.schema import KEY_COLUMNS, EpisodeKey
    from evalstats.harness.store import load_episodes

    grid = load_grid(args.grid)
    specs = load_policies(args.policies_config)
    stored = load_episodes(args.raw_dir / grid.results_dir).set_index(KEY_COLUMNS)
    (suite,) = grid.suites
    wanted = pd.read_csv(args.keys)

    def key_of(name: str, row) -> EpisodeKey:
        spec = specs[name]
        return EpisodeKey(
            spec.policy_id, spec.variant, suite, int(row.task_id), int(row.init_state_id), int(row.seed)
        )  # fmt: skip

    for name in args.policies:
        for row in wanted.itertuples():
            if key_of(name, row).as_tuple() not in stored.index:
                raise SystemExit(f"{key_of(name, row)} is not a stored episode of the grid")
    logger.info("%d keys x %d policies, all stored", len(wanted), len(args.policies))
    if args.dry_run:
        return

    # Imported late: loading LeRobot and torch takes a while and a dry run needs neither.
    import gc

    import torch
    from lerobot.utils.io_utils import write_video

    from evalstats.harness.runner import PolicyRunner

    args.videos_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name in args.policies:
        runner = PolicyRunner(specs[name])
        for row in wanted.itertuples():
            key = key_of(name, row)
            new, frames = runner.rerun(key, record_frames=True)
            old = stored.loc[key.as_tuple()]
            video = args.videos_dir / (
                f"{name}__task{key.task_id:02d}_init{key.init_state_id:02d}_seed{key.seed}.mp4"
            )
            write_video(str(video), frames, FPS)
            results.append({
                "policy": name, "name": NAMES[name], "task_id": key.task_id,
                "task_name": new["task_name"], "init_state_id": key.init_state_id,
                "seed": key.seed, "stored_success": int(old["success"]),
                "rerun_success": new["success"], "stored_len": int(old["episode_len"]),
                "rerun_len": new["episode_len"], "same": (int(old["success"]), int(old["episode_len"])) == (new["success"], new["episode_len"]),
                "stored_run_id": old["run_id"], "rerun_host": socket.gethostname(),
                "video": str(video.relative_to(REPO_ROOT)),
            })  # fmt: skip
            logger.info(
                "%s task %d init %d seed %d: success %d (stored %d), len %d (stored %d) -> %s",
                name, key.task_id, key.init_state_id, key.seed, new["success"], old["success"],
                new["episode_len"], old["episode_len"], video.name,
            )  # fmt: skip
        del runner
        gc.collect()
        torch.cuda.empty_cache()

    out = pd.DataFrame(results)
    out.to_csv(args.out, index=False)
    logger.info("%d of %d re-runs identical to the stored episode", out["same"].sum(), len(out))


if __name__ == "__main__":
    main()
