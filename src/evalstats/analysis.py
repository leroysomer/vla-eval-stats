"""The main grid as arrays, for the analyses of Steps 6 to 8 (docs/plan.md)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from evalstats.harness.grid import load_grid, load_policies
from evalstats.harness.schema import KEY_COLUMNS
from evalstats.harness.store import load_episodes
from evalstats.stats.variance import episodes_to_array

# Names of docs/plan.md, Section 1, by entry of configs/policies.yaml.
NAMES = {"smolvla_n10": "A", "smolvla": "A′", "pi05": "B"}
LABELS = {
    "smolvla_n10": "A  (SmolVLA, 10 actions)",
    "smolvla": "A′ (SmolVLA, 50 actions)",
    "pi05": "B  (π0.5, 10 actions)",
}


def load_outcomes(grid_path: Path, policies_path: Path, raw_dir: Path) -> dict[str, np.ndarray]:
    """Success of each policy of the grid, as a (task, initial state, seed) array.

    Refuses to go on if any planned episode is missing: no policy is compared on an incomplete
    grid (docs/plan.md, Section 7). The arrays of all policies are on the same keys, in the same
    order, so they are paired.
    """
    grid = load_grid(grid_path)
    specs = load_policies(policies_path)
    episodes = load_episodes(raw_dir / grid.results_dir)
    stored = set(episodes[KEY_COLUMNS].itertuples(index=False, name=None))
    outcomes = {}
    for name in grid.policies:
        spec = specs[name]
        missing = [k for k in grid.keys(spec) if k.as_tuple() not in stored]
        if missing:
            raise SystemExit(f"{name}: {len(missing)} planned episodes missing, run Step 4 first")
        rows = episodes[
            (episodes["policy_id"] == spec.policy_id) & (episodes["policy_variant"] == spec.variant)
        ]
        outcomes[name] = episodes_to_array(rows).astype(np.int64)
    return outcomes
