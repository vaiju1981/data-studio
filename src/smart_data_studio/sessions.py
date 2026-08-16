"""Track live workspaces so the host cannot be filled by abandoned ones.

Every session holds its own in-memory DuckDB, which is the real concurrency
ceiling here: a 2.7GB file is a 2.7GB workspace, and two idle tabs cost as much
as two busy ones. So sessions are counted, idle ones are closed, and everything
is closed again on shutdown.
"""

from __future__ import annotations

import atexit
import contextlib
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from smart_data_studio import logs
from smart_data_studio.config import (
    MAX_ACTIVE_SESSIONS,
    SESSION_IDLE_SECONDS,
    temp_directory,
)
from smart_data_studio.dataset import Dataset


class TooManySessions(RuntimeError):
    """Raised when admitting another workspace would overcommit the host."""


@dataclass
class _Entry:
    dataset: Dataset
    touched: float = field(default_factory=time.monotonic)


_entries: dict[str, _Entry] = {}
_lock = threading.Lock()


def register(session_id: str, dataset: Dataset) -> None:
    """Adopt a workspace, evicting idle ones first and refusing if still full."""
    with _lock:
        _evict_idle_locked()
        existing = _entries.pop(session_id, None)
        if existing is not None and existing.dataset is not dataset:
            existing.dataset.close()
        if len(_entries) >= MAX_ACTIVE_SESSIONS:
            raise TooManySessions(
                f"{MAX_ACTIVE_SESSIONS} workspaces are already open on this host. "
                "Try again shortly — idle ones are released automatically."
            )
        _entries[session_id] = _Entry(dataset)
        logs.event("session.registered", active=len(_entries))


def touch(session_id: str) -> None:
    with _lock:
        entry = _entries.get(session_id)
        if entry is not None:
            entry.touched = time.monotonic()


def release(session_id: str) -> None:
    with _lock:
        entry = _entries.pop(session_id, None)
    if entry is not None:
        entry.dataset.close()
        logs.event("session.released")


def active() -> int:
    with _lock:
        return len(_entries)


def _evict_idle_locked() -> None:
    cutoff = time.monotonic() - SESSION_IDLE_SECONDS
    stale = [key for key, entry in _entries.items() if entry.touched < cutoff]
    for key in stale:
        _entries.pop(key).dataset.close()
        logs.event("session.evicted", reason="idle")


def shutdown() -> None:
    """Close every workspace and remove the spill directory.

    Registered with atexit so a container stop does not leave DuckDB temp files
    behind on a mounted volume.
    """
    with _lock:
        entries, _entries_cleared = list(_entries.values()), _entries.clear()
    for entry in entries:
        with contextlib.suppress(Exception):  # shutdown must not raise
            entry.dataset.close()
    shutil.rmtree(Path(temp_directory()), ignore_errors=True)
    if entries:
        logs.event("shutdown", closed=len(entries))


atexit.register(shutdown)
