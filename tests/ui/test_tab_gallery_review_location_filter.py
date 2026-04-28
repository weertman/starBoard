import importlib.util
import sys
from pathlib import Path

from ui_stub_helpers import install_src_stubs, restore_modules, stub_module

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "src/ui/tab_gallery_review.py"
def _load_gallery_review_module():
    module_name = "_tab_gallery_review_location_filter_under_test"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

    stubbed_modules = {
        "PySide6": stub_module("PySide6"),
        "PySide6.QtCore": stub_module("PySide6.QtCore", Qt=type("Qt", (), {})),
        "PySide6.QtWidgets": stub_module(
            "PySide6.QtWidgets",
            QWidget=type("QWidget", (), {}),
            QVBoxLayout=type("QVBoxLayout", (), {}),
            QHBoxLayout=type("QHBoxLayout", (), {}),
            QLabel=type("QLabel", (), {}),
            QComboBox=type("QComboBox", (), {"NoInsert": 0}),
            QPushButton=type("QPushButton", (), {}),
            QCompleter=type("QCompleter", (), {}),
            QMessageBox=type("QMessageBox", (), {}),
            QDialog=type("QDialog", (), {"Accepted": 1}),
        ),
        "src.data.id_registry": stub_module(
            "src.data.id_registry",
            list_ids=lambda *args, **kwargs: [],
        ),
        "src.data.image_index": stub_module(
            "src.data.image_index",
            list_image_files=lambda *args, **kwargs: [],
            invalidate_image_cache=lambda *args, **kwargs: None,
        ),
        "src.data.archive_paths": stub_module(
            "src.data.archive_paths",
            metadata_csv_paths_for_read=lambda *args, **kwargs: [],
            id_column_name=lambda *args, **kwargs: "gallery_id",
            root_for=lambda *args, **kwargs: Path("."),
        ),
        "src.data.best_photo": stub_module(
            "src.data.best_photo",
            reorder_files_with_best=lambda *args, **kwargs: [],
            save_best_for_id=lambda *args, **kwargs: None,
        ),
        "src.data.csv_io": stub_module(
            "src.data.csv_io",
            read_rows_multi=lambda *args, **kwargs: [],
            last_row_per_id=lambda *args, **kwargs: {},
            normalize_id_value=lambda s: "" if s is None else str(s).replace("\ufeff", "").strip(),
        ),
        "src.ui.help_button": stub_module(
            "src.ui.help_button",
            HelpButton=type("HelpButton", (), {}),
            HELP_TEXTS={},
        ),
        "src.ui.annotator_view_second": stub_module(
            "src.ui.annotator_view_second",
            AnnotatorViewSecond=type("AnnotatorViewSecond", (), {}),
        ),
        "src.ui.image_quality_panel": stub_module(
            "src.ui.image_quality_panel",
            ImageQualityPanel=type("ImageQualityPanel", (), {}),
        ),
        "src.ui.query_state_delegate": stub_module(
            "src.ui.query_state_delegate",
            QueryStateDelegate=type("QueryStateDelegate", (), {}),
            apply_quality_to_combobox=lambda *args, **kwargs: None,
        ),
        "src.ui.tab_first_order": stub_module(
            "src.ui.tab_first_order",
            _MetadataEditPopup=type("_MetadataEditPopup", (), {}),
        ),
        "src.ui.tab_setup": stub_module(
            "src.ui.tab_setup",
            _RenameIdDialog=type("_RenameIdDialog", (), {}),
        ),
        "src.data.rename_id": stub_module(
            "src.data.rename_id",
            rename_id=lambda *args, **kwargs: None,
        ),
        "src.data.encounter_info": stub_module(
            "src.data.encounter_info",
            get_encounter_date_from_path=lambda *args, **kwargs: None,
            format_encounter_date=lambda *args, **kwargs: "",
        ),
        "src.utils.interaction_logger": stub_module(
            "src.utils.interaction_logger",
            get_interaction_logger=lambda: None,
        ),
    }
    previous_modules = install_src_stubs(
        stubbed_modules,
        data_modules=(
            "id_registry", "image_index", "archive_paths", "best_photo",
            "csv_io", "rename_id", "encounter_info",
        ),
        ui_modules=(
            "help_button", "annotator_view_second", "image_quality_panel",
            "query_state_delegate", "tab_first_order", "tab_setup",
        ),
        utils_modules=("interaction_logger",),
    )

    try:
        spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        restore_modules(previous_modules)


tab_gallery_review = _load_gallery_review_module()


def _last_row_per_id(rows, id_col):
    latest = {}
    for row in rows:
        key = (row.get(id_col) or "").strip()
        if key:
            latest[key] = row
    return latest


class _DummyCombo:
    def __init__(self):
        self.items = []
        self.current_index = -1

    def blockSignals(self, _blocked):
        return None

    def clear(self):
        self.items.clear()
        self.current_index = -1

    def addItem(self, text, user_data=None):
        self.items.append((text, text if user_data is None else user_data))
        if self.current_index < 0:
            self.current_index = 0

    def addItems(self, texts):
        for text in texts:
            self.addItem(text)

    def setCurrentIndex(self, idx):
        self.current_index = idx

    def currentText(self):
        if 0 <= self.current_index < len(self.items):
            return self.items[self.current_index][0]
        return ""

    def currentData(self):
        if 0 <= self.current_index < len(self.items):
            return self.items[self.current_index][1]
        return None

    def findText(self, text):
        for idx, (item_text, _item_data) in enumerate(self.items):
            if item_text == text:
                return idx
        return -1

    def findData(self, data):
        for idx, (_item_text, item_data) in enumerate(self.items):
            if item_data == data:
                return idx
        return -1

    def texts(self):
        return [text for text, _data in self.items]


class _DummyLogger:
    def __init__(self):
        self.calls = []

    def log(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class _DummyGalleryReview:
    _current_location_filter = tab_gallery_review.TabGalleryReview._current_location_filter
    _load_gallery_locations = tab_gallery_review.TabGalleryReview._load_gallery_locations
    _refresh_location_filter_options = tab_gallery_review.TabGalleryReview._refresh_location_filter_options
    _refresh_ids = tab_gallery_review.TabGalleryReview._refresh_ids
    _on_location_filter_changed = tab_gallery_review.TabGalleryReview._on_location_filter_changed
    _on_rename_clicked = tab_gallery_review.TabGalleryReview._on_rename_clicked

    def __init__(self):
        self.cmb_gallery = _DummyCombo()
        self.cmb_last_location = _DummyCombo()
        self._ilog = _DummyLogger()
        self.gallery_changed_calls = 0

    def _on_gallery_changed(self):
        self.gallery_changed_calls += 1


def _install_gallery_state(monkeypatch, state):
    monkeypatch.setattr(tab_gallery_review, "list_ids", lambda target: list(state["ids"]))
    monkeypatch.setattr(tab_gallery_review.ap, "metadata_csv_paths_for_read", lambda target: ["gallery.csv"])
    monkeypatch.setattr(tab_gallery_review.ap, "id_column_name", lambda target: "gallery_id")
    monkeypatch.setattr(
        tab_gallery_review,
        "read_rows_multi",
        lambda _paths: [dict(row) for row in state["rows"]],
    )
    monkeypatch.setattr(tab_gallery_review, "last_row_per_id", _last_row_per_id)
    applied_ids = []
    monkeypatch.setattr(
        tab_gallery_review,
        "apply_quality_to_combobox",
        lambda _combo, gallery_ids, _target: applied_ids.append(list(gallery_ids)),
    )
    return applied_ids


def test_refresh_ids_filters_gallery_by_latest_location(monkeypatch):
    state = {
        "ids": ["apricot", "berry", "citrus"],
        "rows": [
            {"gallery_id": "apricot", "location": "Pier"},
            {"gallery_id": "berry", "location": "Cove"},
            {"gallery_id": "citrus", "location": ""},
        ],
    }
    applied_ids = _install_gallery_state(monkeypatch, state)
    review = _DummyGalleryReview()

    review._refresh_ids()
    assert review.cmb_last_location.texts() == ["All", "No location", "Cove", "Pier"]

    review.cmb_last_location.setCurrentIndex(review.cmb_last_location.findData("Pier"))
    review._on_location_filter_changed()

    assert review.cmb_gallery.texts() == ["apricot"]
    assert review.cmb_gallery.currentText() == "apricot"
    assert applied_ids[-1] == ["apricot"]


def test_refresh_ids_filters_gallery_with_no_location_option(monkeypatch):
    state = {
        "ids": ["apricot", "berry", "citrus"],
        "rows": [
            {"gallery_id": "apricot", "location": "Pier"},
            {"gallery_id": "berry", "location": ""},
            {"gallery_id": "citrus", "location": ""},
        ],
    }
    applied_ids = _install_gallery_state(monkeypatch, state)
    review = _DummyGalleryReview()

    review._refresh_ids()
    review.cmb_last_location.setCurrentIndex(
        review.cmb_last_location.findData(tab_gallery_review._NO_LOCATION_FILTER_DATA)
    )
    review._on_location_filter_changed()

    assert review.cmb_last_location.currentText() == "No location"
    assert review.cmb_gallery.texts() == ["berry", "citrus"]
    assert applied_ids[-1] == ["berry", "citrus"]


def test_refresh_ids_preserves_selected_gallery_with_active_location_filter(monkeypatch):
    state = {
        "ids": ["apricot", "berry", "citrus"],
        "rows": [
            {"gallery_id": "apricot", "location": "Pier"},
            {"gallery_id": "berry", "location": "Pier"},
            {"gallery_id": "citrus", "location": "Cove"},
        ],
    }
    _install_gallery_state(monkeypatch, state)
    review = _DummyGalleryReview()

    review._refresh_ids()
    review.cmb_last_location.setCurrentIndex(review.cmb_last_location.findData("Pier"))
    review._on_location_filter_changed()
    review.cmb_gallery.setCurrentIndex(review.cmb_gallery.findText("berry"))

    review._refresh_ids()

    assert review.cmb_last_location.currentText() == "Pier"
    assert review.cmb_gallery.texts() == ["apricot", "berry"]
    assert review.cmb_gallery.currentText() == "berry"


def test_refresh_ids_falls_back_to_all_when_filtered_location_disappears(monkeypatch):
    state = {
        "ids": ["apricot", "berry"],
        "rows": [
            {"gallery_id": "apricot", "location": "Pier"},
            {"gallery_id": "berry", "location": "Cove"},
        ],
    }
    _install_gallery_state(monkeypatch, state)
    review = _DummyGalleryReview()

    review._refresh_ids()
    review.cmb_last_location.setCurrentIndex(review.cmb_last_location.findData("Pier"))
    review._on_location_filter_changed()
    assert review.cmb_gallery.texts() == ["apricot"]

    state["rows"] = [
        {"gallery_id": "apricot", "location": "Harbor"},
        {"gallery_id": "berry", "location": "Cove"},
    ]
    review._refresh_ids()

    assert review.cmb_last_location.currentText() == "All"
    assert review.cmb_last_location.texts() == ["All", "Cove", "Harbor"]
    assert review.cmb_gallery.texts() == ["apricot", "berry"]


def test_rename_refresh_keeps_location_filter_and_selects_new_id(monkeypatch):
    state = {
        "ids": ["apricot", "berry"],
        "rows": [
            {"gallery_id": "apricot", "location": "Pier"},
            {"gallery_id": "berry", "location": "Cove"},
        ],
    }
    _install_gallery_state(monkeypatch, state)
    review = _DummyGalleryReview()

    review._refresh_ids()
    review.cmb_last_location.setCurrentIndex(review.cmb_last_location.findData("Pier"))
    review._on_location_filter_changed()

    class _FakeRenameDialog:
        def __init__(self, target, old_id, parent=None):
            self.target = target
            self.old_id = old_id
            self.parent = parent

        def exec(self):
            return tab_gallery_review.QDialog.Accepted

        def get_new_id(self):
            return "apricot-renamed"

    class _RenameReport:
        def __init__(self):
            self.errors = []

    def _rename_id(_target, old_id, new_id):
        assert old_id == "apricot"
        state["ids"] = ["apricot-renamed", "berry"]
        state["rows"] = [
            {"gallery_id": "apricot-renamed", "location": "Pier"},
            {"gallery_id": "berry", "location": "Cove"},
        ]
        return _RenameReport()

    monkeypatch.setattr(tab_gallery_review, "_RenameIdDialog", _FakeRenameDialog)
    monkeypatch.setattr(tab_gallery_review, "rename_id", _rename_id)
    monkeypatch.setattr(tab_gallery_review, "invalidate_image_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(tab_gallery_review.QMessageBox, "warning", staticmethod(lambda *args, **kwargs: None), raising=False)
    monkeypatch.setattr(tab_gallery_review.QMessageBox, "information", staticmethod(lambda *args, **kwargs: None), raising=False)

    review._on_rename_clicked()

    assert review.cmb_last_location.currentText() == "Pier"
    assert review.cmb_gallery.texts() == ["apricot-renamed"]
    assert review.cmb_gallery.currentText() == "apricot-renamed"
