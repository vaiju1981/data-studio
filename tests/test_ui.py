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
