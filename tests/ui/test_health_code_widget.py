import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.annotation_schema import FIELD_BY_NAME
from src.ui.annotation_widgets import HealthCodeEditor, create_widget_for_field
from src.ui.metadata_form_v2 import MetadataFormV2


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_health_code_widget_round_trips_canonical_value():
    _app()
    widget = create_widget_for_field(FIELD_BY_NAME["health_codes"])

    assert isinstance(widget, HealthCodeEditor)

    widget.set_value("l(1+), c-(2), bt")

    assert widget.get_value() == "L(1)+, C-(2), BT"


def test_health_code_widget_adds_blank_row_until_code_is_selected():
    _app()
    widget = create_widget_for_field(FIELD_BY_NAME["health_codes"])

    widget.add_btn.click()
    row = widget._rows[0]

    assert row.code_combo.currentData() is None
    assert widget.get_value() == ""
    assert row.code_combo.itemData(1) == "X"
    assert row.code_combo.itemData(row.code_combo.count() - 1) == "RELEASED"

    row.code_combo.setCurrentIndex(row.code_combo.findData("L"))
    row.count_spin.setValue(2)
    row.plus_check.setChecked(True)

    assert widget.get_value() == "L(2)+"


def test_health_code_widget_user_action_adds_counted_plus_code():
    _app()
    widget = create_widget_for_field(FIELD_BY_NAME["health_codes"])

    widget.add_btn.click()
    row = widget._rows[0]
    row.code_combo.setCurrentIndex(row.code_combo.findData("L"))
    row.count_spin.setValue(2)
    row.plus_check.setChecked(True)

    assert widget.get_value() == "L(2)+"


def test_metadata_form_collects_health_codes_without_touching_data():
    _app()
    form = MetadataFormV2()
    form.set_target("Gallery")
    form.set_id_value("test-star")
    form.populate({"health_codes": "L(2)+, BT"})

    row = form.collect_row_dict()

    assert row["gallery_id"] == "test-star"
    assert row["health_codes"] == "L(2)+, BT"
    assert form.get_widget("health_codes") is not None
