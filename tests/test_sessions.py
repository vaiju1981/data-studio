"""Workspace accounting: the host holds one DuckDB per session, so they are counted."""

from __future__ import annotations

import importlib

import duckdb
import pytest

from smart_data_studio import sessions
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
