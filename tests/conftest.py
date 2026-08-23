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

    release_all rather than shutdown: shutdown also removes the DuckDB spill
    directory, which is shared. Doing that between cases pulled the spill file out
    from under the module-scoped bank datasets, and the trio bank errored four ways
    with "Cannot open file … No such file or directory" on a query that had been
    passing all day.
    """
    yield
    sessions.release_all()
