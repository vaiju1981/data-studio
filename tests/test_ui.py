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
