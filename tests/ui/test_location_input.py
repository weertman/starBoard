import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QComboBox

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.location_registry import add_or_update_location
from src.data.vocabulary_store import get_vocabulary_store
from src.ui.location_input import LocationInputGroup


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_location_input_uses_editable_combo_and_populates_known_coordinates(tmp_path, monkeypatch):
    monkeypatch.setenv("STARBOARD_ARCHIVE_DIR", str(tmp_path / "archive"))
    get_vocabulary_store().reload()
    add_or_update_location("Eagle Point", 48.123456, -123.654321)
    _app()

    widget = LocationInputGroup()

    assert isinstance(widget._name_combo, QComboBox)
    assert widget._name_combo.isEditable()
    assert widget._name_combo.findText("Eagle Point") >= 0

    widget.set_location_name("Eagle Point")

    assert widget.get_values() == {
        "location": "Eagle Point",
        "latitude": "48.123456",
        "longitude": "-123.654321",
    }


def test_location_input_persists_new_typed_location_with_coordinates(tmp_path, monkeypatch):
    monkeypatch.setenv("STARBOARD_ARCHIVE_DIR", str(tmp_path / "archive"))
    get_vocabulary_store().reload()
    _app()

    widget = LocationInputGroup()
    widget.set_location("New Reef", 48.222222, -123.333333)
    widget.persist_current_location()

    second = LocationInputGroup()

    assert second._name_combo.findText("New Reef") >= 0
    second.set_location_name("New Reef")
    assert second.latitude() == 48.222222
    assert second.longitude() == -123.333333
