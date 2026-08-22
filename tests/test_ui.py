from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_starts_with_csv_empty_state() -> None:
    app_path = Path(__file__).parents[1] / "src/smart_data_studio/ui/app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=10)
    assert not app.exception
    assert app.title[0].value == "Smart Data Studio"
    assert app.file_uploader[0].label == "Upload CSV files"


def test_streamlit_is_confined_to_the_ui_package() -> None:
    """The glob was not recursive, so a Streamlit import under ui/ was invisible.

    The contract is the package boundary, not a single file: everything outside
    ui/ must stay free of Streamlit so the core can sit behind another front end.
    """
    package = Path("src/smart_data_studio")
    core = [path for path in package.rglob("*.py") if path.parent != package / "ui"]
    offenders = [str(path) for path in core if "import streamlit" in path.read_text()]
    assert not offenders, f"Streamlit imported outside the ui package: {offenders}"

    # And the ui package really is where it lives, so the rule is not vacuous.
    ui_files = [path for path in (package / "ui").glob("*.py")]
    assert any("import streamlit" in path.read_text() for path in ui_files)


def run_app(monkeypatch, tmp_path):
    monkeypatch.setenv("SDS_STATE_DIR", str(tmp_path / "state"))
    app_path = Path(__file__).parents[1] / "src/smart_data_studio/ui/app.py"
    return AppTest.from_file(str(app_path)).run(timeout=30)


def make_csv(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body)
    return path


def test_a_remembered_path_and_a_new_one_load_together(monkeypatch, tmp_path) -> None:
    """The point of the list: adding a second table to a set already in use must
    not mean retyping the first."""
    from smart_data_studio import recent

    sales = make_csv(tmp_path, "sales.csv", "region,amount\nNorth,10\n")
    regions = make_csv(tmp_path, "regions.csv", "region,manager\nNorth,Ada\n")

    monkeypatch.setenv("SDS_STATE_DIR", str(tmp_path / "state"))
    recent.remember([sales])

    app = run_app(monkeypatch, tmp_path)
    assert app.multiselect[0].label == "Files you have loaded before"
    # Shown as parent/name rather than the full path, which is unreadable in a
    # narrow sidebar and still tells two same-named files apart.
    label = f"{sales.parent.name}/{sales.name}"
    assert app.multiselect[0].options == [label]

    # Pick the remembered one, type the new one, load both.
    app.multiselect[0].select(label)
    app.text_area[0].set_value(str(regions))
    app.button[0].click().run(timeout=60)

    assert not app.exception
    assert sorted(app.session_state.dataset.tables) == ["regions", "sales"]


def test_the_list_is_absent_until_something_has_been_loaded(monkeypatch, tmp_path) -> None:
    app = run_app(monkeypatch, tmp_path)
    assert not app.multiselect
    assert app.text_area[0].label == "Or local CSV paths"


def test_the_selection_survives_a_rerun(monkeypatch, tmp_path) -> None:
    """The text box keeps its contents across a rerun. When the list did not, a
    second click quietly loaded only what had been typed."""
    from smart_data_studio import recent

    sales = make_csv(tmp_path, "sales.csv", "region,amount\nNorth,10\n")
    monkeypatch.setenv("SDS_STATE_DIR", str(tmp_path / "state"))
    recent.remember([sales])

    app = run_app(monkeypatch, tmp_path)
    app.multiselect[0].select(f"{sales.parent.name}/{sales.name}").run(timeout=30)
    assert app.session_state.chosen_paths == [str(sales.resolve())]
    app.run(timeout=30)
    assert app.session_state.chosen_paths == [str(sales.resolve())]


def test_a_selection_of_a_vanished_file_does_not_break_the_sidebar(monkeypatch, tmp_path) -> None:
    """Streamlit refuses a selection that is not among the options, and recall
    drops a path once its file is gone."""
    from smart_data_studio import recent

    sales = make_csv(tmp_path, "sales.csv", "region,amount\nNorth,10\n")
    monkeypatch.setenv("SDS_STATE_DIR", str(tmp_path / "state"))
    recent.remember([sales])

    app = run_app(monkeypatch, tmp_path)
    app.multiselect[0].select(f"{sales.parent.name}/{sales.name}").run(timeout=30)
    sales.unlink()
    app.run(timeout=30)
    assert not app.exception
    assert app.session_state.chosen_paths == []


def test_money_in_an_answer_is_not_rendered_as_mathematics() -> None:
    """Streamlit reads $...$ as LaTeX, so an answer quoting two amounts lost
    everything between them: the real one collapsed "$4.48M**, the monthly volume
    collapsed to a range between **$509k" into a single equation."""
    from smart_data_studio.ui.render import as_text

    written = "peaked at $4.48M**, collapsing to between **$509k and $676k"
    assert as_text(written) == r"peaked at \$4.48M**, collapsing to between **\$509k and \$676k"

    # Idempotent, so an answer that escaped its own dollars is left alone.
    assert as_text(as_text(written)) == as_text(written)
    # Nothing else is touched.
    assert as_text("plain **bold** text") == "plain **bold** text"


def test_an_evicted_workspace_clears_the_tab_instead_of_failing_later(monkeypatch, tmp_path):
    """Eviction closes the connection from whichever thread noticed. The tab went
    on rendering panels over a closed workspace and only found out at the next
    question, as "Connection already closed" — a database error where what happened
    was that the session had been released for sitting idle.

    A tab holding a dataset the registry has never heard of is exactly what an
    evicted one looks like from here, so that is what this sets up."""
    from smart_data_studio.dataset import CsvSource, Dataset

    app = run_app(monkeypatch, tmp_path)
    dataset = Dataset.load([CsvSource.from_upload("s.csv", b"a\n1\n")])
    try:
        app.session_state.dataset = dataset
        app.run(timeout=30)

        assert not app.exception
        assert app.session_state.dataset is None
        assert app.session_state.expired
        assert any("released after sitting idle" in item.value for item in app.warning)
    finally:
        dataset.close()
