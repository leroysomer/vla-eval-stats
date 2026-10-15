#!/usr/bin/env python
"""Step 4 completeness check: is every planned episode of a grid stored exactly once?

    uv run scripts/check_completeness.py

Compares the keys of configs/grid_main.yaml with the rows of results/raw/main/ and writes
results/tables/main_completeness.csv: one row per shard (policy x task) and one row per policy.
A key that is missing is written to results/tables/main_missing.csv with its reason: the error
of its last crash (failures*.jsonl), or "not run". Exits with status 1 if the grid is not
complete. No success rate is read here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evalstats.harness.grid import load_grid, load_policies
from evalstats.harness.schema import KEY_COLUMNS
from evalstats.harness.store import load_episodes

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(paths: list[Path]) -> list[dict]:
    return [json.loads(line) for p in paths for line in p.read_text().splitlines() if line]


def count(label: str, planned: set[tuple], stored: pd.DataFrame, crashes: dict, hosts: dict):
    """Completeness of one set of planned keys against the rows stored for them."""
    keys = list(stored[KEY_COLUMNS].itertuples(index=False, name=None))
    present = set(keys)
    return {
        "task": label,
        "n_planned": len(planned),
        "n_stored": len(present & planned),
        "n_missing": len(planned - present),
        "n_duplicated": len(keys) - len(present),
        "n_unplanned": len(present - planned),
        "n_crash_records": sum(k in planned for k in crashes),
        "hosts": " ".join(sorted({hosts.get(r, "unknown") for r in stored["run_id"]})),
        "first_episode": stored["timestamp"].min() if len(stored) else "",
        "last_episode": stored["timestamp"].max() if len(stored) else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--grid", type=Path, default=REPO_ROOT / "configs/grid_main.yaml")
    parser.add_argument("--policies-config", type=Path, default=REPO_ROOT / "configs/policies.yaml")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "results/raw")
    parser.add_argument("--tables-dir", type=Path, default=REPO_ROOT / "results/tables")
    args = parser.parse_args()

    grid = load_grid(args.grid)
    specs = load_policies(args.policies_config)
    out_dir = args.raw_dir / grid.results_dir
    episodes = load_episodes(out_dir)
    # Crashes are missing data, not task failures: they are logged apart from the episode rows.
    crashes = {tuple(r["key"]): r for r in read_jsonl(sorted(out_dir.glob("failures*.jsonl")))}
    hosts = {r["run_id"]: r["host"].split(".")[0] for r in read_jsonl(sorted(out_dir.glob("runs*.jsonl")))}  # fmt: skip

    rows, missing = [], []
    for name in grid.policies:
        spec = specs[name]
        keys = grid.keys(spec)
        stored = episodes[
            (episodes["policy_id"] == spec.policy_id) & (episodes["policy_variant"] == spec.variant)
        ]
        present = set(stored[KEY_COLUMNS].itertuples(index=False, name=None))
        by_task: dict[tuple, set[tuple]] = {}
        for k in keys:
            by_task.setdefault((k.suite, k.task_id), set()).add(k.as_tuple())
            if k.as_tuple() not in present:
                crash = crashes.get(k.as_tuple())
                reason = f"crashed: {crash['error']}" if crash else "not run"
                missing.append({"policy": name, **dict(zip(KEY_COLUMNS, k.as_tuple())), "reason": reason})  # fmt: skip
        table = [
            count(f"{suite} task {task_id}", planned, d, crashes, hosts)
            for (suite, task_id), planned in by_task.items()
            for d in [stored[(stored["suite"] == suite) & (stored["task_id"] == task_id)]]
        ]
        table.append(count("all", {k.as_tuple() for k in keys}, stored, crashes, hosts))
        for row in table:
            rows.append({"policy": name, "policy_variant": spec.variant, **row})

    out = pd.DataFrame(rows)
    out["complete"] = (out[["n_missing", "n_duplicated", "n_unplanned"]] == 0).all(axis=1)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.tables_dir / f"{grid.name}_completeness.csv", index=False)
    missing_file = args.tables_dir / f"{grid.name}_missing.csv"
    if missing:
        pd.DataFrame(missing).to_csv(missing_file, index=False)
    else:
        missing_file.unlink(missing_ok=True)

    totals = out[out["task"] == "all"]
    print(totals.drop(columns=["task", "first_episode", "last_episode"]).to_string(index=False))
    n_planned, n_stored = int(totals["n_planned"].sum()), int(totals["n_stored"].sum())
    print(f"{grid.name}: {n_stored} of {n_planned} planned episodes stored ({n_stored / n_planned:.1%})")  # fmt: skip
    if not out["complete"].all():
        raise SystemExit(f"{grid.name} is not complete, see {missing_file}")


if __name__ == "__main__":
    main()
