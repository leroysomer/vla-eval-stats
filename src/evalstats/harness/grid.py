"""Read the YAML configs and expand a grid into episode keys."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .schema import EpisodeKey

# Number of tasks per LIBERO suite, to expand `task_ids: all` without importing the simulator.
SUITE_N_TASKS = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}


@dataclass(frozen=True)
class PolicySpec:
    name: str
    policy_id: str
    variant: str
    revision: str | None = None
    policy_overrides: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Grid:
    name: str
    policies: list[str]
    suites: dict[str, list[int]]
    init_state_ids: list[int]
    seeds: list[int]
    perturbation: str = "none"
    perturbation_level: int = 0

    def keys(self, policy: PolicySpec) -> list[EpisodeKey]:
        """Planned episodes of one policy, ordered by shard (suite, task)."""
        return [
            EpisodeKey(
                policy_id=policy.policy_id,
                policy_variant=policy.variant,
                suite=suite,
                task_id=task_id,
                init_state_id=init_state_id,
                seed=seed,
                perturbation=self.perturbation,
                perturbation_level=self.perturbation_level,
            )
            for suite, task_ids in self.suites.items()
            for task_id in task_ids
            for init_state_id in self.init_state_ids
            for seed in self.seeds
        ]


def _as_ids(value, what: str) -> list[int]:
    ids = list(range(value)) if isinstance(value, int) else [int(v) for v in value]
    if not ids or len(set(ids)) != len(ids) or min(ids) < 0:
        raise ValueError(f"{what} must be a positive int or a list of distinct ids, got {value!r}")
    return ids


def load_policies(path: Path | str) -> dict[str, PolicySpec]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    specs = {
        name: PolicySpec(
            name=name,
            policy_id=entry["policy_id"],
            variant=entry["variant"],
            revision=entry.get("revision"),
            policy_overrides=entry.get("policy_overrides") or {},
            env=entry.get("env") or {},
        )
        for name, entry in raw.items()
    }
    pairs = [(s.policy_id, s.variant) for s in specs.values()]
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"{path}: two entries share the same (policy_id, variant)")
    return specs


def load_grid(path: Path | str) -> Grid:
    with open(path) as f:
        raw = yaml.safe_load(f)
    suites = {}
    for suite, spec in raw["suites"].items():
        if suite not in SUITE_N_TASKS:
            raise ValueError(f"unknown suite {suite!r}; known: {sorted(SUITE_N_TASKS)}")
        task_ids = (spec or {}).get("task_ids", "all")
        n_tasks = SUITE_N_TASKS[suite]
        suites[suite] = (
            list(range(n_tasks)) if task_ids == "all" else _as_ids(task_ids, f"{suite}.task_ids")
        )
        if max(suites[suite]) >= n_tasks:
            raise ValueError(f"{suite} has {n_tasks} tasks, got task_ids {suites[suite]}")
    return Grid(
        name=raw["name"],
        policies=list(raw["policies"]),
        suites=suites,
        init_state_ids=_as_ids(raw["init_state_ids"], "init_state_ids"),
        seeds=_as_ids(raw["seeds"], "seeds"),
        perturbation=raw.get("perturbation", "none"),
        perturbation_level=int(raw.get("perturbation_level", 0)),
    )
