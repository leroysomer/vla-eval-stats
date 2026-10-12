"""Parquet storage for episode records: one file per shard, rewritten atomically."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from .schema import COLUMNS, DTYPES, KEY_COLUMNS, EpisodeKey


def _to_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=COLUMNS).astype(DTYPES)


def load_episodes(out_dir: Path | str) -> pd.DataFrame:
    """All episode rows stored under `out_dir`, across shards."""
    files = sorted(Path(out_dir).glob("*.parquet"))
    if not files:
        return _to_frame([])
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df[COLUMNS].astype(DTYPES)


class EpisodeStore:
    """Episode rows of one output directory.

    Every `add` rewrites the episode's shard through a temporary file and an atomic rename, so
    killing the process at any point leaves each shard either without or with the new row,
    never half-written.
    """

    def __init__(self, out_dir: Path | str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        for leftover in self.out_dir.glob("*.parquet.tmp"):
            leftover.unlink()
        self._shards: dict[str, pd.DataFrame] = {
            f.stem: pd.read_parquet(f)[COLUMNS].astype(DTYPES)
            for f in sorted(self.out_dir.glob("*.parquet"))
        }
        self._done: set[tuple] = set()
        for df in self._shards.values():
            keys = list(df[KEY_COLUMNS].itertuples(index=False, name=None))
            if len(set(keys)) != len(keys) or self._done & set(keys):
                raise ValueError(f"duplicate episode keys in {self.out_dir}")
            self._done |= set(keys)

    def __len__(self) -> int:
        return len(self._done)

    def __contains__(self, key: EpisodeKey) -> bool:
        return key.as_tuple() in self._done

    def config_hashes(self, policy_id: str, policy_variant: str) -> set[str]:
        """Config hashes already stored for a policy variant (more than one means a mix-up)."""
        hashes: set[str] = set()
        for df in self._shards.values():
            sel = (df["policy_id"] == policy_id) & (df["policy_variant"] == policy_variant)
            hashes |= set(df.loc[sel, "config_hash"])
        return hashes

    def add(self, key: EpisodeKey, row: dict) -> None:
        if key in self:
            raise ValueError(f"episode already stored: {key}")
        if tuple(row[c] for c in KEY_COLUMNS) != key.as_tuple():
            raise ValueError(f"row does not match its key: {key}")
        new = _to_frame([row])
        old = self._shards.get(key.shard)
        df = new if old is None else pd.concat([old, new], ignore_index=True)
        path = self.out_dir / f"{key.shard}.parquet"
        tmp = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        self._shards[key.shard] = df
        self._done.add(key.as_tuple())


def append_jsonl(path: Path | str, record: dict) -> None:
    """Append one JSON line (crash log, run log) and flush it to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
