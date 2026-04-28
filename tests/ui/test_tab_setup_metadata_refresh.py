import importlib.util
import sys
from pathlib import Path

from ui_stub_helpers import install_src_stubs, restore_modules, stub_class, stub_module

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "src/ui/tab_setup.py"
def _load_tab_setup_module():
    module_name = "src.ui._tab_setup_under_test"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

    signal_factory = lambda *args, **kwargs: object()
    stubbed_modules = {
        "PySide6": stub_module("PySide6"),
        "PySide6.QtCore": stub_module(
            "PySide6.QtCore",
            Qt=stub_class("Qt"),
            QDate=stub_class("QDate"),
            QTimer=stub_class("QTimer"),
            Signal=signal_factory,
        ),
        "PySide6.QtWidgets": stub_module(
            "PySide6.QtWidgets",
            QWidget=stub_class("QWidget"),
            QVBoxLayout=stub_class("QVBoxLayout"),
            QHBoxLayout=stub_class("QHBoxLayout"),
            QGroupBox=stub_class("QGroupBox"),
            QPushButton=stub_class("QPushButton"),
            QComboBox=stub_class("QComboBox", NoInsert=0),
            QLabel=stub_class("QLabel"),
            QFileDialog=stub_class("QFileDialog"),
            QListWidget=stub_class("QListWidget"),
            QListWidgetItem=stub_class("QListWidgetItem"),
            QLineEdit=stub_class("QLineEdit"),
            QDateEdit=stub_class("QDateEdit"),
            QMessageBox=stub_class("QMessageBox"),
            QCheckBox=stub_class("QCheckBox"),
            QPlainTextEdit=stub_class("QPlainTextEdit"),
            QScrollArea=stub_class("QScrollArea"),
            QSizePolicy=stub_class("QSizePolicy"),
            QTabWidget=stub_class("QTabWidget"),
            QCompleter=stub_class("QCompleter"),
            QDialog=stub_class("QDialog"),
            QDialogButtonBox=stub_class("QDialogButtonBox"),
            QRadioButton=stub_class("QRadioButton"),
            QButtonGroup=stub_class("QButtonGroup"),
            QTableWidget=stub_class("QTableWidget"),
            QTableWidgetItem=stub_class("QTableWidgetItem"),
            QHeaderView=stub_class("QHeaderView"),
            QAbstractItemView=stub_class("QAbstractItemView"),
            QSpinBox=stub_class("QSpinBox"),
        ),
        "src.ui.help_button": stub_module(
            "src.ui.help_button",
            HelpButton=type("HelpButton", (), {}),
            HELP_TEXTS={},
        ),
        "src.ui.collapsible": stub_module(
            "src.ui.collapsible",
            CollapsibleSection=stub_class("CollapsibleSection"),
        ),
        "src.data.archive_paths": stub_module(
            "src.data.archive_paths",
            last_observation_for_all=lambda *_args, **_kwargs: {},
        ),
        "src.data.csv_io": stub_module(
            "src.data.csv_io",
            append_row=lambda *_args, **_kwargs: None,
            read_rows_multi=lambda *_args, **_kwargs: [],
            last_row_per_id=lambda *_args, **_kwargs: {},
            normalize_id_value=lambda value: value,
        ),
        "src.data.id_registry": stub_module(
            "src.data.id_registry",
            list_ids=lambda *_args, **_kwargs: [],
            id_exists=lambda *_args, **_kwargs: False,
            invalidate_id_cache=lambda: None,
        ),
        "src.data.ingest": stub_module(
            "src.data.ingest",
            ensure_encounter_name=lambda *_args, **_kwargs: "",
            place_images=lambda *_args, **_kwargs: None,
            discover_ids_and_images=lambda *_args, **_kwargs: [],
            discover_ids_with_encounters=lambda *_args, **_kwargs: [],
            discover_grouped_ids_with_encounters=lambda *_args, **_kwargs: [],
            detect_folder_depth=lambda *_args, **_kwargs: "flat",
            _encounter_suffix=lambda *_args, **_kwargs: "",
            _parse_encounter_date=lambda *_args, **_kwargs: None,
            image_file_dialog_filter=lambda: "Images (*.jpg *.JPG);;All Files (*)",
        ),
        "src.data.batch_undo": stub_module(
            "src.data.batch_undo",
            generate_batch_id=lambda *_args, **_kwargs: "",
            record_batch_upload=lambda *_args, **_kwargs: None,
            list_batches=lambda *_args, **_kwargs: [],
            undo_batch=lambda *_args, **_kwargs: None,
            redo_batch=lambda *_args, **_kwargs: None,
            check_redo_sources=lambda *_args, **_kwargs: [],
            BatchInfo=stub_class("BatchInfo"),
        ),
        "src.data.metadata_history": stub_module(
            "src.data.metadata_history",
            record_bulk_update=lambda *_args, **_kwargs: None,
            get_current_metadata_for_gallery=lambda *_args, **_kwargs: {},
            SOURCE_UI="ui",
        ),
        "src.data.validators": stub_module(
            "src.data.validators",
            validate_id=lambda *_args, **_kwargs: (True, ""),
        ),
        "src.data.best_photo": stub_module(
            "src.data.best_photo",
            reorder_files_with_best=lambda *_args, **_kwargs: [],
            save_best_for_id=lambda *_args, **_kwargs: None,
        ),
        "src.data.image_index": stub_module(
            "src.data.image_index",
            list_image_files=lambda *_args, **_kwargs: [],
        ),
        "src.data.encounter_info": stub_module(
            "src.data.encounter_info",
            list_encounters_for_id=lambda *_args, **_kwargs: [],
            get_encounter_date=lambda *_args, **_kwargs: None,
            set_encounter_date=lambda *_args, **_kwargs: None,
        ),
        "src.data.field_visits": stub_module(
            "src.data.field_visits",
            append_field_visit=lambda *_args, **_kwargs: None,
            get_field_visit_locations=lambda *_args, **_kwargs: [],
            read_field_visits=lambda *_args, **_kwargs: [],
            delete_field_visit=lambda *_args, **_kwargs: False,
        ),
        "src.data.archive_merge": stub_module(
            "src.data.archive_merge",
            scan_external_archive=lambda *_args, **_kwargs: [],
            build_merge_plan=lambda *_args, **_kwargs: None,
            execute_merge=lambda *_args, **_kwargs: None,
            MergeItem=stub_class("MergeItem"),
            MergePlan=stub_class("MergePlan"),
            MergeReport=stub_class("MergeReport"),
        ),
        "src.ui.metadata_form_v2": stub_module(
            "src.ui.metadata_form_v2",
            MetadataForm=stub_class("MetadataForm"),
        ),
        "src.utils.interaction_logger": stub_module(
            "src.utils.interaction_logger",
            get_interaction_logger=lambda: None,
        ),
    }

    previous_modules = install_src_stubs(
        stubbed_modules,
        data_modules=(
            "archive_paths", "csv_io", "id_registry", "ingest", "batch_undo",
            "metadata_history", "validators", "best_photo", "image_index",
            "encounter_info", "field_visits", "archive_merge",
        ),
        ui_modules=("help_button", "collapsible", "metadata_form_v2"),
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


tab_setup = _load_tab_setup_module()


class _DummyCombo:
    def __init__(self, items=None, current_index=-1):
        self.items = list(items or [])
        self.current_index = current_index
        if self.items and self.current_index < 0:
            self.current_index = 0

    def blockSignals(self, _blocked):
        return None

    def clear(self):
        self.items.clear()
        self.current_index = -1

    def addItems(self, texts):
        self.items.extend(texts)
        if self.items and self.current_index < 0:
            self.current_index = 0

    def setCurrentIndex(self, idx):
        self.current_index = idx

    def currentText(self):
        if 0 <= self.current_index < len(self.items):
            return self.items[self.current_index]
        return ""

    def findText(self, text):
        for idx, item in enumerate(self.items):
            if item == text:
                return idx
        return -1

    def texts(self):
        return list(self.items)


class _DummyButton:
    def __init__(self):
        self.enabled = None

    def setEnabled(self, enabled):
        self.enabled = enabled


class _DummyMetaForm:
    def __init__(self):
        self.targets = []

    def set_target(self, target):
        self.targets.append(target)


class _DummyLogger:
    def __init__(self):
        self.calls = []

    def log(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class _DummySetupTab:
    _refresh_id_list_edit = tab_setup.TabSetup._refresh_id_list_edit
    _on_refresh_id_edit_clicked = tab_setup.TabSetup._on_refresh_id_edit_clicked

    def __init__(self, *, target="Gallery", ids=None, current_index=-1, last_edit_id=""):
        self.cmb_target_edit = _DummyCombo([target], current_index=0)
        self.cmb_id_edit = _DummyCombo(ids, current_index=current_index)
        self.meta_form_edit = _DummyMetaForm()
        self.btn_save_only = _DummyButton()
        self.btn_save_edit = _DummyButton()
        self._last_edit_id = last_edit_id
        self._ilog = _DummyLogger()
        self.edit_change_calls = 0

    def _on_edit_id_changed(self):
        self.edit_change_calls += 1


def test_refresh_button_invalidates_cache_and_preserves_current_id(monkeypatch):
    dummy = _DummySetupTab(ids=["alpha", "beta"], current_index=1, last_edit_id="beta")
    invalidations = []
    monkeypatch.setattr(tab_setup, "invalidate_id_cache", lambda: invalidations.append(True))
    monkeypatch.setattr(tab_setup, "list_ids", lambda _target: ["alpha", "beta", "delta"])

    dummy._on_refresh_id_edit_clicked()

    assert invalidations == [True]
    assert dummy.cmb_id_edit.texts() == ["alpha", "beta", "delta"]
    assert dummy.cmb_id_edit.currentText() == "beta"
    assert dummy.edit_change_calls == 0
    assert dummy.btn_save_only.enabled is True
    assert dummy.btn_save_edit.enabled is True
    assert dummy.meta_form_edit.targets == ["Gallery"]
    assert dummy._ilog.calls[0][0][:2] == ("button_click", "btn_refresh_id_edit")


def test_refresh_id_list_edit_reloads_when_previous_id_is_missing(monkeypatch):
    dummy = _DummySetupTab(ids=["beta"], current_index=0, last_edit_id="beta")
    monkeypatch.setattr(tab_setup, "list_ids", lambda _target: ["aardvark", "gamma"])

    dummy._refresh_id_list_edit()

    assert dummy.cmb_id_edit.texts() == ["aardvark", "gamma"]
    assert dummy.cmb_id_edit.currentText() == "aardvark"
    assert dummy.edit_change_calls == 1
    assert dummy.meta_form_edit.targets == ["Gallery"]


def test_refresh_id_list_edit_force_reload_rehydrates_current_selection(monkeypatch):
    dummy = _DummySetupTab(ids=["beta"], current_index=0, last_edit_id="beta")
    monkeypatch.setattr(tab_setup, "list_ids", lambda _target: ["alpha", "beta"])

    dummy._refresh_id_list_edit(force_reload=True)

    assert dummy.cmb_id_edit.currentText() == "beta"
    assert dummy.edit_change_calls == 1
