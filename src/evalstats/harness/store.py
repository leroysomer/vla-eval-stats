"""Parquet storage for episode records: one file per shard, rewritten atomically.

The output directory may sit on a network disk and be written by several machines at once, as
long as each policy variant is run by one process at a time (`RunLock`).
"""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import Self

import pandas as pd

from .schema import COLUMNS, DTYPES, KEY_COLUMNS, EpisodeKey

HOST = socket.gethostname().split(".")[0]
# A temporary file or a lock untouched for this long belongs to a process that died.
STALE_AFTER_S = 15 * 60


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
        # Temporary files of interrupted writes. Recent ones may belong to a live process on
        # another machine, so only old ones are removed.
        for leftover in self.out_dir.glob("*.tmp"):
            if time.time() - leftover.stat().st_mtime > STALE_AFTER_S:
                leftover.unlink(missing_ok=True)
        self.reload()

    def reload(self) -> None:
        """Read the directory again, to see rows written by other processes since."""
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
        tmp = path.with_name(f"{path.name}.{HOST}.{os.getpid()}.tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        self._shards[key.shard] = df
        self._done.add(key.as_tuple())


class RunLock:
    """Reserve one policy variant of an output directory for the current process.

    Two processes running the same policy variant, on one machine or two, would overwrite each
    other's shards. The lock is a file created atomically; `beat()` refreshes it after every
    episode. A lock whose process is gone (same machine) or that has not been refreshed for
    `STALE_AFTER_S` (another machine) is taken over.
    """

    def __init__(self, out_dir: Path | str, policy_id: str, policy_variant: str):
        name = f"{policy_id.replace('/', '--')}__{policy_variant}.lock"
        self.path = Path(out_dir) / name
        self.held = False

    def _holder_is_gone(self) -> bool:
        try:
            holder = json.loads(self.path.read_text())
            age = time.time() - self.path.stat().st_mtime
        except (OSError, ValueError):
            return False  # just created or just released by someone else: try again
        if holder.get("host") == HOST:
            try:
                os.kill(int(holder["pid"]), 0)
            except ProcessLookupError:
                return True
            except (PermissionError, KeyError, ValueError):
                pass
        return age > STALE_AFTER_S

    def acquire(self) -> None:
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if self._holder_is_gone():
                    self.path.unlink(missing_ok=True)
                    continue
                raise RuntimeError(
                    f"{self.path.name} is held by another run ({self.path.read_text().strip()}). "
                    "Wait for it, or delete the lock file if that run is dead."
                ) from None
            with os.fdopen(fd, "w") as f:
                json.dump({"host": HOST, "pid": os.getpid(), "started": time.time()}, f)
            self.held = True
            return
        raise RuntimeError(f"could not take over the stale lock {self.path}")

    def beat(self) -> None:
        os.utime(self.path)

    def release(self) -> None:
        if self.held:
            self.path.unlink(missing_ok=True)
            self.held = False

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def append_jsonl(path: Path | str, record: dict) -> None:
    """Append one JSON line (crash log, run log) and flush it to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
