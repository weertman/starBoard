import importlib.util
import sys
from pathlib import Path

from ui_stub_helpers import install_src_stubs, restore_modules, stub_module

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "src/ui/tab_gallery_review.py"
def _load_gallery_review_module():
    module_name = "_tab_gallery_review_under_test"
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
            QComboBox=type("QComboBox", (), {}),
            QPushButton=type("QPushButton", (), {}),
            QCompleter=type("QCompleter", (), {}),
            QMessageBox=type("QMessageBox", (), {}),
            QDialog=type("QDialog", (), {}),
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
        "src.data.archive_paths": stub_module("src.data.archive_paths"),
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


class _FakeSignal:
    def __init__(self):
        self._callbacks = []

    def connect(self, callback):
        self._callbacks.append(callback)

    def emit(self, *args, **kwargs):
        for callback in list(self._callbacks):
            callback(*args, **kwargs)


class _FakePopup:
    instances = []

    def __init__(self, target, id_value, parent=None):
        self._target = target
        self._id_value = id_value
        self._parent = parent
        self.saved = _FakeSignal()
        self.destroyed = _FakeSignal()
        self.visible = False
        self.deleted = False
        self.close_result = True
        self.close_calls = 0
        self.show_calls = 0
        self.raise_calls = 0
        self.activate_calls = 0
        _FakePopup.instances.append(self)

    def isVisible(self):
        if self.deleted:
            raise RuntimeError("Internal C++ object (_MetadataEditPopup) already deleted.")
        return self.visible

    def close(self):
        if self.deleted:
            raise RuntimeError("Internal C++ object (_MetadataEditPopup) already deleted.")
        self.close_calls += 1
        if not self.close_result:
            return False
        self.visible = False
        return True

    def show(self):
        if self.deleted:
            raise RuntimeError("Internal C++ object (_MetadataEditPopup) already deleted.")
        self.visible = True
        self.show_calls += 1

    def raise_(self):
        if self.deleted:
            raise RuntimeError("Internal C++ object (_MetadataEditPopup) already deleted.")
        self.raise_calls += 1

    def activateWindow(self):
        if self.deleted:
            raise RuntimeError("Internal C++ object (_MetadataEditPopup) already deleted.")
        self.activate_calls += 1

    def emit_destroyed(self):
        self.deleted = True
        self.visible = False
        self.destroyed.emit()


class _DummyCombo:
    def __init__(self, text):
        self._text = text

    def currentText(self):
        return self._text


class _DummyLogger:
    def __init__(self):
        self.calls = []

    def log(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class _DummyGalleryReview:
    _active_meta_edit_popup = tab_gallery_review.TabGalleryReview._active_meta_edit_popup
    _on_meta_edit_popup_destroyed = tab_gallery_review.TabGalleryReview._on_meta_edit_popup_destroyed
    _show_meta_edit_popup = tab_gallery_review.TabGalleryReview._show_meta_edit_popup
    _on_edit_metadata_clicked = tab_gallery_review.TabGalleryReview._on_edit_metadata_clicked

    def __init__(self, gid):
        self.cmb_gallery = _DummyCombo(gid)
        self._ilog = _DummyLogger()
        self._meta_edit_popup = None
        self.metadata_saved_calls = 0

    def _on_metadata_saved(self):
        self.metadata_saved_calls += 1


def _patch_popup(monkeypatch):
    _FakePopup.instances.clear()
    monkeypatch.setattr(tab_gallery_review, "_MetadataEditPopup", _FakePopup)


def test_reopen_after_popup_destroy_clears_cached_reference(monkeypatch):
    _patch_popup(monkeypatch)
    review = _DummyGalleryReview("apricot")

    review._on_edit_metadata_clicked()
    first_popup = review._meta_edit_popup

    first_popup.emit_destroyed()
    assert review._meta_edit_popup is None

    review.cmb_gallery._text = "baby_carrot"
    review._on_edit_metadata_clicked()

    second_popup = review._meta_edit_popup
    assert second_popup is not None
    assert second_popup is not first_popup
    assert second_popup._id_value == "baby_carrot"
    assert first_popup.close_calls == 0


def test_switching_gallery_respects_close_veto(monkeypatch):
    _patch_popup(monkeypatch)
    review = _DummyGalleryReview("baby_carrot")

    existing_popup = _FakePopup("Gallery", "apricot", review)
    existing_popup.visible = True
    existing_popup.close_result = False
    review._meta_edit_popup = existing_popup

    review._on_edit_metadata_clicked()

    assert review._meta_edit_popup is existing_popup
    assert len(_FakePopup.instances) == 1
    assert existing_popup.close_calls == 1
    assert existing_popup.show_calls == 1
    assert existing_popup.raise_calls == 1
    assert existing_popup.activate_calls == 1


def test_destroyed_old_popup_does_not_clear_new_popup_reference(monkeypatch):
    _patch_popup(monkeypatch)
    review = _DummyGalleryReview("apricot")

    review._on_edit_metadata_clicked()
    first_popup = review._meta_edit_popup

    review.cmb_gallery._text = "baby_carrot"
    review._on_edit_metadata_clicked()
    second_popup = review._meta_edit_popup

    assert second_popup is not first_popup

    first_popup.emit_destroyed()

    assert review._meta_edit_popup is second_popup
