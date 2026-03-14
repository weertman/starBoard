import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "src/ui/tab_setup.py"
_MISSING = object()


def _stub_class(name, **attrs):
    return type(name, (), attrs)


def _stub_module(name, **attrs):
    module = types.ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    return module


def _load_tab_setup_module():
    module_name = "src.ui._tab_setup_under_test"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

    signal_factory = lambda *args, **kwargs: object()
    stubbed_modules = {
        "PySide6": types.ModuleType("PySide6"),
        "PySide6.QtCore": _stub_module(
            "PySide6.QtCore",
            Qt=_stub_class("Qt"),
            QDate=_stub_class("QDate"),
            QTimer=_stub_class("QTimer"),
            Signal=signal_factory,
        ),
        "PySide6.QtWidgets": _stub_module(
            "PySide6.QtWidgets",
            QWidget=_stub_class("QWidget"),
            QVBoxLayout=_stub_class("QVBoxLayout"),
            QHBoxLayout=_stub_class("QHBoxLayout"),
            QGroupBox=_stub_class("QGroupBox"),
            QPushButton=_stub_class("QPushButton"),
            QComboBox=_stub_class("QComboBox", NoInsert=0),
            QLabel=_stub_class("QLabel"),
            QFileDialog=_stub_class("QFileDialog"),
            QListWidget=_stub_class("QListWidget"),
            QListWidgetItem=_stub_class("QListWidgetItem"),
            QLineEdit=_stub_class("QLineEdit"),
            QDateEdit=_stub_class("QDateEdit"),
            QMessageBox=_stub_class("QMessageBox"),
            QCheckBox=_stub_class("QCheckBox"),
            QPlainTextEdit=_stub_class("QPlainTextEdit"),
            QScrollArea=_stub_class("QScrollArea"),
            QSizePolicy=_stub_class("QSizePolicy"),
            QTabWidget=_stub_class("QTabWidget"),
            QCompleter=_stub_class("QCompleter"),
            QDialog=_stub_class("QDialog"),
            QDialogButtonBox=_stub_class("QDialogButtonBox"),
            QRadioButton=_stub_class("QRadioButton"),
            QButtonGroup=_stub_class("QButtonGroup"),
            QTableWidget=_stub_class("QTableWidget"),
            QTableWidgetItem=_stub_class("QTableWidgetItem"),
            QHeaderView=_stub_class("QHeaderView"),
            QAbstractItemView=_stub_class("QAbstractItemView"),
            QSpinBox=_stub_class("QSpinBox"),
        ),
        "src.ui.collapsible": _stub_module(
            "src.ui.collapsible",
            CollapsibleSection=_stub_class("CollapsibleSection"),
        ),
        "src.data.archive_paths": _stub_module(
            "src.data.archive_paths",
            last_observation_for_all=lambda *_args, **_kwargs: {},
        ),
        "src.data.csv_io": _stub_module(
            "src.data.csv_io",
            append_row=lambda *_args, **_kwargs: None,
            read_rows_multi=lambda *_args, **_kwargs: [],
            last_row_per_id=lambda *_args, **_kwargs: {},
            normalize_id_value=lambda value: value,
        ),
        "src.data.id_registry": _stub_module(
            "src.data.id_registry",
            list_ids=lambda *_args, **_kwargs: [],
            id_exists=lambda *_args, **_kwargs: False,
            invalidate_id_cache=lambda: None,
        ),
        "src.data.ingest": _stub_module(
            "src.data.ingest",
            ensure_encounter_name=lambda *_args, **_kwargs: "",
            place_images=lambda *_args, **_kwargs: None,
            discover_ids_and_images=lambda *_args, **_kwargs: [],
        ),
        "src.data.batch_undo": _stub_module(
            "src.data.batch_undo",
            generate_batch_id=lambda *_args, **_kwargs: "",
            record_batch_upload=lambda *_args, **_kwargs: None,
            list_batches=lambda *_args, **_kwargs: [],
            undo_batch=lambda *_args, **_kwargs: None,
            redo_batch=lambda *_args, **_kwargs: None,
            check_redo_sources=lambda *_args, **_kwargs: [],
            BatchInfo=_stub_class("BatchInfo"),
        ),
        "src.data.metadata_history": _stub_module(
            "src.data.metadata_history",
            record_bulk_update=lambda *_args, **_kwargs: None,
            get_current_metadata_for_gallery=lambda *_args, **_kwargs: {},
            SOURCE_UI="ui",
        ),
        "src.data.validators": _stub_module(
            "src.data.validators",
            validate_id=lambda *_args, **_kwargs: (True, ""),
        ),
        "src.data.best_photo": _stub_module(
            "src.data.best_photo",
            reorder_files_with_best=lambda *_args, **_kwargs: [],
            save_best_for_id=lambda *_args, **_kwargs: None,
        ),
        "src.data.image_index": _stub_module(
            "src.data.image_index",
            list_image_files=lambda *_args, **_kwargs: [],
        ),
        "src.data.encounter_info": _stub_module(
            "src.data.encounter_info",
            list_encounters_for_id=lambda *_args, **_kwargs: [],
            get_encounter_date=lambda *_args, **_kwargs: None,
            set_encounter_date=lambda *_args, **_kwargs: None,
        ),
        "src.data.negative_outings": _stub_module(
            "src.data.negative_outings",
            append_negative_outing=lambda *_args, **_kwargs: None,
            get_negative_outing_locations=lambda *_args, **_kwargs: [],
            read_negative_outings=lambda *_args, **_kwargs: [],
        ),
        "src.data.archive_merge": _stub_module(
            "src.data.archive_merge",
            scan_external_archive=lambda *_args, **_kwargs: [],
            build_merge_plan=lambda *_args, **_kwargs: None,
            execute_merge=lambda *_args, **_kwargs: None,
            MergeItem=_stub_class("MergeItem"),
            MergePlan=_stub_class("MergePlan"),
            MergeReport=_stub_class("MergeReport"),
        ),
        "src.ui.metadata_form_v2": _stub_module(
            "src.ui.metadata_form_v2",
            MetadataForm=_stub_class("MetadataForm"),
        ),
        "src.utils.interaction_logger": _stub_module(
            "src.utils.interaction_logger",
            get_interaction_logger=lambda: None,
        ),
    }

    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = []
    data_pkg = types.ModuleType("src.data")
    data_pkg.__path__ = []
    ui_pkg = types.ModuleType("src.ui")
    ui_pkg.__path__ = []
    utils_pkg = types.ModuleType("src.utils")
    utils_pkg.__path__ = []

    src_pkg.data = data_pkg
    src_pkg.ui = ui_pkg
    src_pkg.utils = utils_pkg
    data_pkg.archive_paths = stubbed_modules["src.data.archive_paths"]
    data_pkg.csv_io = stubbed_modules["src.data.csv_io"]
    data_pkg.id_registry = stubbed_modules["src.data.id_registry"]
    data_pkg.ingest = stubbed_modules["src.data.ingest"]
    data_pkg.batch_undo = stubbed_modules["src.data.batch_undo"]
    data_pkg.metadata_history = stubbed_modules["src.data.metadata_history"]
    data_pkg.validators = stubbed_modules["src.data.validators"]
    data_pkg.best_photo = stubbed_modules["src.data.best_photo"]
    data_pkg.image_index = stubbed_modules["src.data.image_index"]
    data_pkg.encounter_info = stubbed_modules["src.data.encounter_info"]
    data_pkg.negative_outings = stubbed_modules["src.data.negative_outings"]
    data_pkg.archive_merge = stubbed_modules["src.data.archive_merge"]
    ui_pkg.collapsible = stubbed_modules["src.ui.collapsible"]
    ui_pkg.metadata_form_v2 = stubbed_modules["src.ui.metadata_form_v2"]
    utils_pkg.interaction_logger = stubbed_modules["src.utils.interaction_logger"]

    stubbed_modules.update({
        "src": src_pkg,
        "src.data": data_pkg,
        "src.ui": ui_pkg,
        "src.utils": utils_pkg,
    })
    previous_modules = {name: sys.modules.get(name, _MISSING) for name in stubbed_modules}

    try:
        sys.modules.update(stubbed_modules)
        spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in previous_modules.items():
            if previous is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


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
