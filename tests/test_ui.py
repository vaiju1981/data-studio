from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_starts_with_csv_empty_state() -> None:
    app_path = Path(__file__).parents[1] / "src/smart_data_studio/ui/app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=10)
    assert not app.exception
    assert app.title[0].value == "Smart Data Studio"
    assert app.file_uploader[0].label == "Upload CSV files"


def test_streamlit_is_confined_to_ui_layer() -> None:
    core_files = Path("src/smart_data_studio").glob("*.py")
    assert all("import streamlit" not in path.read_text() for path in core_files)
