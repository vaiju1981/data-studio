"""Workspace accounting: the host holds one DuckDB per session, so they are counted."""

from __future__ import annotations

import importlib
from pathlib import Path

import duckdb
import pytest

from smart_data_studio import config, sessions
from smart_data_studio.dataset import CsvSource, Dataset


def make_dataset() -> Dataset:
    return Dataset.load([CsvSource.from_upload("s.csv", b"a\n1\n")])


@pytest.fixture(autouse=True)
def clean_registry():
    sessions.shutdown()
    yield
    sessions.shutdown()
    importlib.reload(sessions)


def test_workspaces_are_counted_and_released() -> None:
    first, second = make_dataset(), make_dataset()
    sessions.register("one", first)
    sessions.register("two", second)
    assert sessions.active() == 2

    sessions.release("one")
    assert sessions.active() == 1
    sessions.release("two")
    assert sessions.active() == 0


def test_a_full_host_refuses_rather_than_overcommitting(monkeypatch) -> None:
    monkeypatch.setattr(sessions, "MAX_ACTIVE_SESSIONS", 1)
    sessions.register("one", make_dataset())
    with pytest.raises(sessions.TooManySessions, match="already open"):
        sessions.register("two", make_dataset())


def test_idle_workspaces_are_evicted_to_make_room(monkeypatch) -> None:
    """An abandoned tab costs as much as a busy one, so it does not keep its slot."""
    monkeypatch.setattr(sessions, "MAX_ACTIVE_SESSIONS", 1)
    monkeypatch.setattr(sessions, "SESSION_IDLE_SECONDS", 0)
    sessions.register("stale", make_dataset())
    sessions.register("fresh", make_dataset())  # the idle one is reclaimed first
    assert sessions.active() == 1


def test_re_registering_a_session_replaces_its_workspace() -> None:
    first = make_dataset()
    sessions.register("one", first)
    sessions.register("one", make_dataset())
    assert sessions.active() == 1
    with pytest.raises(duckdb.Error):
        first.connection.execute("SELECT 1")  # the old workspace was closed


def test_shutdown_closes_everything_and_clears_the_spill_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sessions, "temp_directory", lambda: str(tmp_path / "spill"))
    (tmp_path / "spill").mkdir()
    (tmp_path / "spill" / "leftover.tmp").write_text("x")
    sessions.register("one", make_dataset())

    sessions.shutdown()
    assert sessions.active() == 0
    assert not (tmp_path / "spill").exists()


def test_a_full_host_is_refused_before_anything_is_loaded() -> None:
    """register() is the real gate but runs after the file is parsed and profiled,
    so a host with no room still spent a minute and several gigabytes finding out."""
    datasets = [make_dataset() for _ in range(sessions.MAX_ACTIVE_SESSIONS)]
    try:
        for index, dataset in enumerate(datasets):
            sessions.register(f"s{index}", dataset)
        with pytest.raises(sessions.TooManySessions):
            sessions.check_capacity("newcomer")
        # A session replacing its own workspace always has room.
        sessions.check_capacity("s0")
    finally:
        sessions.shutdown()


def test_the_spill_directory_is_always_one_we_own(monkeypatch, tmp_path) -> None:
    """Shutdown removes it recursively. Returning a configured path unchanged meant
    SDS_DUCKDB_TEMP_DIR=/data deleted /data and everything in it."""
    monkeypatch.setattr(config, "DUCKDB_TEMP_DIR", str(tmp_path))
    assert Path(config.temp_directory()).parent == tmp_path
    assert Path(config.temp_directory()).name == "smart-data-studio"


def test_an_evicted_session_says_so_rather_than_going_quiet() -> None:
    """touch() was silent about a session that no longer existed, so a tab went on
    rendering a workspace whose connection was already closed and only found out
    at the next question, as "Connection already closed"."""
    dataset = make_dataset()
    sessions.register("one", dataset)
    assert sessions.touch("one") is True

    sessions.release("one")
    assert sessions.touch("one") is False
    assert sessions.touch("never-registered") is False


def test_releasing_workspaces_leaves_the_shared_spill_directory_alone(tmp_path) -> None:
    """The trio bank errored four ways on "Cannot open file … No such file or
    directory", from a query that had been passing all day.

    The autouse fixture called shutdown() after every test, and shutdown removes
    the DuckDB spill directory — which every live connection shares. A
    module-scoped dataset spilling mid-query found its own temp file gone.
    """
    from pathlib import Path

    from smart_data_studio.config import temp_directory

    spill = Path(temp_directory())
    spill.mkdir(parents=True, exist_ok=True)
    marker = spill / "in-use.tmp"
    marker.write_bytes(b"a live connection's spill")

    dataset = make_dataset()
    sessions.register("one", dataset)
    assert sessions.release_all() == 1
    assert marker.exists(), "releasing a workspace deleted another connection's spill file"

    # Shutdown still clears it, because by then the process is going away.
    sessions.shutdown()
    assert not marker.exists()


def test_a_workspace_answering_a_question_is_not_evicted_as_idle(monkeypatch) -> None:
    """Idleness is measured between page runs, and a question runs inside one.

    An investigation longer than the idle window looked exactly like an abandoned
    tab, and eviction closes the connection from whichever thread noticed — so the
    query in flight failed with "Connection already closed" while its own tab was
    plainly in use.
    """
    monkeypatch.setattr(sessions, "SESSION_IDLE_SECONDS", 0)
    dataset = make_dataset()
    sessions.register("busy", dataset)

    with sessions.working("busy"):
        sessions.check_capacity("other")  # evicts what it can
        assert sessions.active() == 1, "the workspace was released mid-question"

    # Once the question is done it goes idle like any other.
    sessions.check_capacity("other")
    assert sessions.active() == 0


def test_the_lease_survives_a_workspace_that_is_not_registered() -> None:
    """The exploration takes the lease straight after registering, and a load that
    was refused has nothing to hold. It must not raise on the way past."""
    with sessions.working("never-registered"):
        pass
