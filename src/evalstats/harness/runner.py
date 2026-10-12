"""Run LIBERO episodes one at a time with LeRobot and log one record per episode.

A thin wrapper around `lerobot.scripts.lerobot_eval.rollout`, needed because `lerobot-eval`
(0.6.1) keeps per-episode success but drops the episode length, picks the initial state with a
hidden counter (episode i of a task gets initial state i), and seeds torch once per process, so
the policy noise of an episode depends on what ran before it. Here the initial state is an
explicit input and every episode is seeded on its own, which makes episodes independent of run
order: a resumed run reproduces a fresh one.
"""

from __future__ import annotations

import datetime as dt
import importlib.metadata
import json
import logging
import os
import socket
import subprocess
import time
import traceback
from contextlib import nullcontext
from itertools import groupby
from pathlib import Path

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs import make_env, make_env_config, make_env_pre_post_processors
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.scripts import lerobot_eval
from lerobot.utils.random_utils import set_seed

from .grid import PolicySpec
from .schema import EpisodeKey, config_hash
from .store import EpisodeStore, append_jsonl

logger = logging.getLogger(__name__)

# `rollout` draws a per-step progress bar unless it believes it runs under Slurm. One log line
# per episode is enough, and the bars flood the log files of long runs.
lerobot_eval.inside_slurm = lambda: True

REPO_ROOT = Path(__file__).resolve().parents[3]
# A crash is missing data, not a task failure: it goes to this file, never to the parquet rows.
FAILURES_FILE = "failures.jsonl"
RUNS_FILE = "runs.jsonl"
MAX_CONSECUTIVE_CRASHES = 3


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def repo_commit() -> str:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return git("rev-parse", "--short", "HEAD") + (
            "-dirty" if git("status", "--porcelain") else ""
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def lerobot_version() -> str:
    """LeRobot is installed from PyPI, so the release version stands in for a commit."""
    return importlib.metadata.version("lerobot")


def policy_config_hash(spec: PolicySpec) -> str:
    return config_hash(
        {
            "policy_id": spec.policy_id,
            "revision": spec.revision,
            "variant": spec.variant,
            "policy_overrides": spec.policy_overrides,
            "env": spec.env,
            "lerobot": lerobot_version(),
        }
    )


class PolicyRunner:
    """One loaded policy variant, able to run any LIBERO episode key."""

    def __init__(self, spec: PolicySpec):
        self.spec = spec
        self.config_hash = policy_config_hash(spec)
        self.run_id = f"{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%SZ}-{os.getpid()}-{spec.name}"

        # Same global settings as lerobot-eval.
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        start = time.perf_counter()
        # JSON spelling, so that YAML booleans reach draccus as true/false.
        overrides = [
            f"--{k}={v if isinstance(v, str) else json.dumps(v)}"
            for k, v in spec.policy_overrides.items()
        ]
        cfg = PreTrainedConfig.from_pretrained(
            spec.policy_id, revision=spec.revision, cli_overrides=overrides
        )
        cfg.pretrained_path = Path(spec.policy_id)
        cfg.pretrained_revision = spec.revision
        self.policy_cfg = cfg
        env_cfg = make_env_config("libero", **spec.env)
        self.policy = make_policy(cfg=cfg, env_cfg=env_cfg, rename_map={})
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=cfg.pretrained_path,
            pretrained_revision=spec.revision,
            preprocessor_overrides={
                "device_processor": {"device": str(self.policy.config.device)},
                "rename_observations_processor": {"rename_map": {}},
            },
        )
        self.env_preprocessor, self.env_postprocessor = make_env_pre_post_processors(
            env_cfg=env_cfg, policy_cfg=cfg
        )
        self.model_load_s = time.perf_counter() - start

    def _make_env(self, suite: str, task_id: int):
        env_cfg = make_env_config("libero", task=suite, task_ids=[task_id], **self.spec.env)
        return make_env(env_cfg, n_envs=1)[suite][task_id]

    def _run_episode(self, vec_env, key: EpisodeKey) -> dict:
        sub_env = vec_env.envs[0].unwrapped
        n_init_states = len(sub_env._init_states)
        if key.init_state_id >= n_init_states:
            raise ValueError(f"{key.suite} task {key.task_id} has {n_init_states} initial states")
        # LeRobot's LiberoEnv loads `_init_states[init_state_id]` on reset, then increments it.
        sub_env.init_state_id = key.init_state_id
        set_seed(key.rng_seed)

        start = time.perf_counter()
        amp = (
            torch.autocast(device_type=self.policy.config.device)
            if self.policy_cfg.use_amp
            else None
        )
        with torch.no_grad(), amp or nullcontext():
            data = lerobot_eval.rollout(
                env=vec_env,
                policy=self.policy,
                env_preprocessor=self.env_preprocessor,
                env_postprocessor=self.env_postprocessor,
                preprocessor=self.preprocessor,
                postprocessor=self.postprocessor,
                seeds=[key.rng_seed],
            )
        wall_time_s = time.perf_counter() - start

        if sub_env.init_state_id != key.init_state_id + 1:
            raise RuntimeError(
                "the env was reset more than once: the initial state is not the one asked"
            )
        # `done` latches at the first termination, or at the suite's step limit.
        done = data["done"][0]
        episode_len = int(torch.argmax(done.int())) + 1
        success = bool(data["success"][0, :episode_len].any())
        return {
            "policy_id": key.policy_id,
            "policy_variant": key.policy_variant,
            "suite": key.suite,
            "task_id": key.task_id,
            "task_name": sub_env.task,
            "init_state_id": key.init_state_id,
            "seed": key.seed,
            "rng_seed": key.rng_seed,
            "perturbation": key.perturbation,
            "perturbation_level": key.perturbation_level,
            "success": int(success),
            "episode_len": episode_len,
            "wall_time_s": wall_time_s,
            "lerobot_commit": lerobot_version(),
            "repo_commit": self._repo_commit,
            "config_hash": self.config_hash,
            "run_id": self.run_id,
            "timestamp": _now(),
        }

    def run(self, keys: list[EpisodeKey], store: EpisodeStore) -> dict:
        """Run the episodes of `keys` that `store` does not hold yet, saving after each one."""
        for key in keys:
            if (key.policy_id, key.policy_variant) != (self.spec.policy_id, self.spec.variant):
                raise ValueError(f"{key} is not an episode of policy {self.spec.name}")
            if key.perturbation != "none":
                raise NotImplementedError(
                    "perturbations are out of scope (docs/plan.md, Section 9)"
                )
        stored = store.config_hashes(self.spec.policy_id, self.spec.variant)
        if stored - {self.config_hash}:
            raise ValueError(
                f"{store.out_dir} holds episodes of {self.spec.policy_id} [{self.spec.variant}] run "
                f"with other settings (config_hash {sorted(stored)} != {self.config_hash}). "
                "Give the changed settings a new variant name, or use another output directory."
            )

        self._repo_commit = repo_commit()
        pending = [k for k in keys if k not in store]
        summary = {
            "run_id": self.run_id,
            "started": _now(),
            "host": socket.gethostname(),
            "policy": self.spec.name,
            "policy_id": self.spec.policy_id,
            "policy_variant": self.spec.variant,
            "config_hash": self.config_hash,
            "n_planned": len(keys),
            "n_already_done": len(keys) - len(pending),
            "n_run": 0,
            "n_crashed": 0,
            "model_load_s": self.model_load_s,
            "env_build_s": 0.0,
        }
        logger.info(
            "%s: %d planned, %d already done, %d to run",
            self.spec.name, len(keys), len(keys) - len(pending), len(pending),
        )  # fmt: skip
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        consecutive_crashes = 0
        try:
            for (suite, task_id), shard_keys in groupby(
                pending, key=lambda k: (k.suite, k.task_id)
            ):
                env_start = time.perf_counter()
                vec_env = self._make_env(suite, task_id)
                summary["env_build_s"] += time.perf_counter() - env_start
                try:
                    for key in shard_keys:
                        try:
                            row = self._run_episode(vec_env, key)
                        except Exception as exc:
                            summary["n_crashed"] += 1
                            consecutive_crashes += 1
                            logger.exception("episode crashed: %s", key)
                            append_jsonl(
                                store.out_dir / FAILURES_FILE,
                                {
                                    "run_id": self.run_id,
                                    "timestamp": _now(),
                                    "key": key.as_tuple(),
                                    "error": repr(exc),
                                    "traceback": traceback.format_exc(),
                                },
                            )
                            if consecutive_crashes >= MAX_CONSECUTIVE_CRASHES:
                                raise RuntimeError(
                                    f"{consecutive_crashes} episodes crashed in a row, stopping"
                                ) from exc
                            continue
                        consecutive_crashes = 0
                        store.add(key, row)
                        summary["n_run"] += 1
                        logger.info(
                            "[%d/%d] %s task %d init %d seed %d: success=%d len=%d %.1fs",
                            summary["n_run"] + summary["n_crashed"], len(pending), suite, task_id,
                            key.init_state_id, key.seed, row["success"], row["episode_len"],
                            row["wall_time_s"],
                        )  # fmt: skip
                finally:
                    vec_env.close()
        finally:
            summary["total_wall_s"] = time.perf_counter() - start
            summary["peak_gpu_mem_mib"] = (
                torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
            )
            summary["finished"] = _now()
            append_jsonl(store.out_dir / RUNS_FILE, summary)
        return summary
