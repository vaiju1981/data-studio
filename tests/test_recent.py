"""Remembering which local CSVs were loaded, so they need not be retyped."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smart_data_studio import recent


@pytest.fixture(autouse=True)
def state_dir(tmp_path, monkeypatch) -> Path:
    """Never the real home directory: this module writes the one file the app keeps."""
    monkeypatch.setenv("SDS_STATE_DIR", str(tmp_path / "state"))
    return tmp_path


def csv(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_text("a,b\n1,2\n")
    return path


def test_nothing_is_remembered_before_anything_is_loaded() -> None:
    assert recent.recall() == []


def test_a_loaded_path_comes_back(state_dir) -> None:
    path = csv(state_dir, "sales.csv")
    recent.remember([path])
    assert recent.recall() == [str(path.resolve())]


def test_the_newest_comes_first_and_is_not_duplicated(state_dir) -> None:
    first, second = csv(state_dir, "a.csv"), csv(state_dir, "b.csv")
    recent.remember([first])
    recent.remember([second])
    recent.remember([first])
    assert recent.recall() == [str(first.resolve()), str(second.resolve())]


def test_the_list_is_bounded(state_dir) -> None:
    for index in range(recent.MAX_RECENT + 5):
        recent.remember([csv(state_dir, f"f{index}.csv")])
    assert len(recent.recall()) == recent.MAX_RECENT


def test_a_path_that_has_gone_away_is_not_offered(state_dir) -> None:
    """A stale entry that fails on load is worse than no entry, and this is also
    what quietly removes a path that was mistyped."""
    path = csv(state_dir, "gone.csv")
    recent.remember([path])
    path.unlink()
    assert recent.recall() == []


def test_a_corrupt_store_is_ignored_rather_than_fatal() -> None:
    store = Path(recent._store())
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text("{not json at all")
    assert recent.recall() == []


def test_forgetting_clears_the_file(state_dir) -> None:
    recent.remember([csv(state_dir, "a.csv")])
    recent.forget()
    assert recent.recall() == []
    recent.forget()  # twice is not an error


def test_remembering_survives_an_unwritable_directory(state_dir, monkeypatch) -> None:
    """A load that already succeeded must not fail over a convenience."""
    monkeypatch.setenv("SDS_STATE_DIR", "/proc/nonexistent/definitely-not-writable")
    recent.remember([csv(state_dir, "a.csv")])
    assert recent.recall() == []


def test_only_paths_are_written_never_contents(state_dir) -> None:
    path = csv(state_dir, "secret.csv")
    path.write_text("name,ssn\nAda,111-22-3333\n")
    recent.remember([path])
    written = Path(recent._store()).read_text()
    assert "111-22-3333" not in written and "Ada" not in written
    assert json.loads(written) == [str(path.resolve())]
