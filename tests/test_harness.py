"""Storage and grid logic of the harness (no simulator, no GPU)."""

from pathlib import Path

import pytest

from evalstats.harness.grid import load_grid, load_policies
from evalstats.harness.schema import COLUMNS, KEY_COLUMNS, EpisodeKey
from evalstats.harness.store import EpisodeStore, load_episodes

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_key(**changes) -> EpisodeKey:
    fields = {
        "policy_id": "org/policy",
        "policy_variant": "default",
        "suite": "libero_spatial",
        "task_id": 0,
        "init_state_id": 0,
        "seed": 0,
    }
    return EpisodeKey(**{**fields, **changes})


def make_row(key: EpisodeKey, cfg_hash: str = "abc") -> dict:
    row = dict.fromkeys(COLUMNS, "x")
    row.update(dict(zip(KEY_COLUMNS, key.as_tuple(), strict=True)))
    row.update(
        rng_seed=key.rng_seed, success=1, episode_len=100, wall_time_s=1.5, config_hash=cfg_hash
    )
    return row


def test_store_roundtrip_and_resume(tmp_path):
    keys = [make_key(task_id=t, init_state_id=i) for t in range(2) for i in range(3)]
    store = EpisodeStore(tmp_path)
    for key in keys[:4]:
        store.add(key, make_row(key))

    # A new process sees what the first one stored and only the rest is left to run.
    resumed = EpisodeStore(tmp_path)
    assert [k in resumed for k in keys] == [True] * 4 + [False] * 2
    for key in keys:
        if key not in resumed:
            resumed.add(key, make_row(key))

    df = load_episodes(tmp_path)
    assert list(df.columns) == COLUMNS
    assert len(df) == len(keys)
    assert not df.duplicated(KEY_COLUMNS).any()
    assert sorted(p.name for p in tmp_path.glob("*.parquet")) == [
        "org--policy__default__libero_spatial__task00.parquet",
        "org--policy__default__libero_spatial__task01.parquet",
    ]


def test_store_refuses_duplicates(tmp_path):
    store = EpisodeStore(tmp_path)
    key = make_key()
    store.add(key, make_row(key))
    with pytest.raises(ValueError, match="already stored"):
        store.add(key, make_row(key))
    with pytest.raises(ValueError, match="does not match"):
        store.add(make_key(seed=1), make_row(key))


def test_store_ignores_interrupted_write(tmp_path):
    store = EpisodeStore(tmp_path)
    key = make_key()
    store.add(key, make_row(key))
    (tmp_path / f"{key.shard}.parquet.tmp").write_bytes(b"half-written")
    assert len(EpisodeStore(tmp_path)) == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_store_config_hashes(tmp_path):
    store = EpisodeStore(tmp_path)
    for seed, cfg_hash in [(0, "aaa"), (1, "bbb")]:
        key = make_key(seed=seed)
        store.add(key, make_row(key, cfg_hash))
    assert store.config_hashes("org/policy", "default") == {"aaa", "bbb"}
    assert store.config_hashes("org/policy", "other") == set()


def test_rng_seed_is_shared_across_policies_only():
    key = make_key()
    assert key.rng_seed == make_key(policy_id="org/other", policy_variant="fast").rng_seed
    others = [
        make_key(task_id=1),
        make_key(init_state_id=1),
        make_key(seed=1),
        make_key(suite="libero_10"),
    ]
    assert len({key.rng_seed, *(k.rng_seed for k in others)}) == 5
    assert 0 <= key.rng_seed < 2**31


def test_repo_configs_expand():
    specs = load_policies(REPO_ROOT / "configs/policies.yaml")
    grid = load_grid(REPO_ROOT / "configs/grid_throughput.yaml")
    for name in grid.policies:
        keys = grid.keys(specs[name])
        assert len(keys) == 20
        assert len({k.as_tuple() for k in keys}) == 20
        assert {k.task_id for k in keys} == set(range(10))


def test_main_grid_is_paired_and_holds_the_sanity_check():
    specs = load_policies(REPO_ROOT / "configs/policies.yaml")
    main = load_grid(REPO_ROOT / "configs/grid_main.yaml")
    sanity = load_grid(REPO_ROOT / "configs/grid_sanity.yaml")

    # Every policy runs the same 2,000 (suite, task, initial state, seed) scenarios.
    scenarios = [{k.as_tuple()[2:] for k in main.keys(specs[name])} for name in main.policies]
    assert len(main.policies) == 3
    assert all(len(s) == 2000 and s == scenarios[0] for s in scenarios)

    # The sanity check is stored with the main grid and is part of it.
    assert sanity.results_dir == main.results_dir == "main"
    for name in sanity.policies:
        keys = {k.as_tuple() for k in sanity.keys(specs[name])}
        assert len(keys) == 100
        assert keys <= {k.as_tuple() for k in main.keys(specs[name])}
