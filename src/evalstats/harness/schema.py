"""Per-episode record: one parquet row per episode."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

# An episode is identified by these columns; the harness never runs the same key twice.
KEY_COLUMNS = [
    "policy_id",
    "policy_variant",
    "suite",
    "task_id",
    "init_state_id",
    "seed",
    "perturbation",
    "perturbation_level",
]

COLUMNS = [
    "policy_id",
    "policy_variant",
    "suite",
    "task_id",
    "task_name",
    "init_state_id",
    "seed",
    "rng_seed",
    "perturbation",
    "perturbation_level",
    "success",
    "episode_len",
    "wall_time_s",
    "lerobot_commit",
    "repo_commit",
    "config_hash",
    "run_id",
    "timestamp",
]

DTYPES = {
    "policy_id": "string",
    "policy_variant": "string",
    "suite": "string",
    "task_id": "int64",
    "task_name": "string",
    "init_state_id": "int64",
    "seed": "int64",
    "rng_seed": "int64",
    "perturbation": "string",
    "perturbation_level": "int64",
    "success": "int64",
    "episode_len": "int64",
    "wall_time_s": "float64",
    "lerobot_commit": "string",
    "repo_commit": "string",
    "config_hash": "string",
    "run_id": "string",
    "timestamp": "string",
}


@dataclass(frozen=True)
class EpisodeKey:
    policy_id: str
    policy_variant: str
    suite: str
    task_id: int
    init_state_id: int
    seed: int
    perturbation: str = "none"
    perturbation_level: int = 0

    def as_tuple(self) -> tuple:
        return tuple(asdict(self)[c] for c in KEY_COLUMNS)

    @property
    def shard(self) -> str:
        """File stem of the shard this episode belongs to: one policy variant x one task."""
        name = f"{self.policy_id.replace('/', '--')}__{self.policy_variant}__{self.suite}"
        if self.perturbation != "none":
            name += f"__{self.perturbation}-{self.perturbation_level}"
        return f"{name}__task{self.task_id:02d}"

    @property
    def rng_seed(self) -> int:
        """Seed actually given to torch, numpy and the simulator for this episode.

        It depends on the scenario (suite, task, initial state, seed, perturbation) and not on
        the policy, so two policies evaluated on the same scenario share it (paired design), and
        not on the order in which episodes run, so a resumed run reproduces a fresh one. Hashing
        avoids reusing one noise stream for every initial state that has the same `seed`.
        """
        scenario = (
            self.suite,
            self.task_id,
            self.init_state_id,
            self.seed,
            self.perturbation,
            self.perturbation_level,
        )
        digest = hashlib.sha256(json.dumps(scenario).encode()).digest()
        return int.from_bytes(digest[:4], "big") % (2**31)


def config_hash(config: dict) -> str:
    """Short stable hash of everything that defines how a policy variant is evaluated."""
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
