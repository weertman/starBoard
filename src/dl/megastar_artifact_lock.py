from __future__ import annotations

import fcntl
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from src.dl.registry import DLRegistry


def lock_path() -> Path:
    return DLRegistry.get_precompute_root() / ".megastar_artifact.lock"


@contextmanager
def megastar_artifact_lock(timeout_seconds: float | None = None) -> Iterator[None]:
    path = lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
    with open(path, "a+", encoding="utf-8") as lock_file:
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for MegaStar artifact lock: {path}")
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
