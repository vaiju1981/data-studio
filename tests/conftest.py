"""Shared fixtures."""

from __future__ import annotations

import pytest

from smart_data_studio import sessions


@pytest.fixture(autouse=True)
def close_workspaces():
    """Release any workspace a test registered, before pytest closes its streams.

    Left alone, the atexit shutdown runs during interpreter teardown and logs to a
    stream pytest has already closed. Logging swallows that internally, so it
    cannot be caught at the call site — the fix is to leave it nothing to do.
    """
    yield
    sessions.shutdown()
