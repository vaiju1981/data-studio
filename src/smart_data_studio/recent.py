"""Local CSV paths this machine has loaded before.

Paths only, never data. The workspace itself stays in memory and is discarded with
the session; this exists so a file loaded last week can be picked from a list
instead of retyped, and it is the one thing the app writes to disk.

A path that no longer exists is dropped on the way out rather than offered and
then failed on, which also means a mistyped path removes itself.

The list belongs to the machine, not to a person: one file, and every tab of a
hosted instance reads it. That is only ever reached where server paths are
enabled, which is the same setting the config already calls disqualifying for a
shared deployment and which the container turns off — so the two are tied
together here rather than left to line up by luck.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from smart_data_studio.config import ALLOW_LOCAL_PATHS

# Enough to cover the files someone actually works with, short enough to stay a
# list rather than a search problem.
MAX_RECENT = 12


def _store() -> Path:
    root = os.environ.get("SDS_STATE_DIR") or str(Path.home() / ".smart-data-studio")
    return Path(root) / "recent.json"


def recall() -> list[str]:
    """Remembered paths, newest first, minus any that have since gone away."""
    if not ALLOW_LOCAL_PATHS:
        return []
    try:
        saved = json.loads(_store().read_text())
    except (OSError, ValueError):
        return []
    if not isinstance(saved, list):
        return []
    return [item for item in saved if isinstance(item, str) and Path(item).is_file()]


def remember(paths: list[Path]) -> None:
    """Move these to the front of the list, keeping it deduplicated and bounded."""
    if not paths or not ALLOW_LOCAL_PATHS:
        return
    fresh = [str(Path(path).expanduser().resolve()) for path in paths]
    combined = fresh + [item for item in recall() if item not in fresh]
    try:
        store = _store()
        store.parent.mkdir(parents=True, exist_ok=True)
        store.write_text(json.dumps(combined[:MAX_RECENT], indent=2))
    except OSError:
        # Remembering is a convenience. A read-only home directory is not a reason
        # to fail a load that has already succeeded.
        pass


def forget() -> None:
    _store().unlink(missing_ok=True)
