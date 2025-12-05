from __future__ import annotations

from typing import Dict, List
from PySide6.QtGui import QDoubleValidator, QIntValidator
from PySide6.QtWidgets import QWidget, QFormLayout, QLineEdit, QTextEdit, QLabel

from src.data.archive_paths import GALLERY_HEADER, QUERIES_HEADER

class MetadataForm(QWidget):
    """
    Renders the exact metadata fields (EXCEPT the ID field, which is controlled
    by the top-level selector). The ID is injected at save time via set_id_value().
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._target = "Gallery"
        self._id_value = ""
        self._loaded_snapshot: Dict[str, str] = {}

        self.form = QFormLayout(self)
        self.widgets: Dict[str, QWidget] = {}

        self._build_form(GALLERY_HEADER)
        self._loaded_snapshot = self.collect_row_dict()   # prime

    def set_target(self, target: str) -> None:
        if target == self._target:
            return
        self._target = target
        header = GALLERY_HEADER if target == "Gallery" else QUERIES_HEADER
        self._rebuild(header)

    def set_id_value(self, value: str) -> None:
        self._id_value = value or ""

    def populate(self, values: Dict[str, str]) -> None:
        header = self._header()
        id_col = self._id_col()
        for col in header:
            if col == id_col:
                continue  # hidden in UI
            w = self.widgets.get(col)
            if isinstance(w, QLineEdit):
                w.setText(values.get(col, ""))
            elif isinstance(w, QTextEdit):
                w.setPlainText(values.get(col, ""))
        self._loaded_snapshot = self.collect_row_dict()

    def apply_values(self, values: Dict[str, str]) -> None:
        """
        Set current widget values for provided fields without touching the
        internal 'loaded snapshot'. This keeps is_dirty() = True so the user
        can save carried-over values on the new ID.
        """
        header = self._header()
        id_col = self._id_col()
        for col in header:
            if col == id_col:
                continue
            if col not in values:
                continue
            w = self.widgets.get(col)
            val = values.get(col, "")
            try:
                if hasattr(w, "setText"):
                    w.setText(val)  # QLineEdit
                elif hasattr(w, "setPlainText"):
                    w.setPlainText(val)  # QTextEdit
            except Exception:
                pass

    def collect_row(self) -> Dict[str, str]:
        return self.collect_row_dict()

    def collect_row_dict(self) -> Dict[str, str]:
        header = self._header()
        id_col = self._id_col()
        out: Dict[str, str] = {k: "" for k in header}
        out[id_col] = self._id_value  # inject ID
        for col in header:
            if col == id_col:
                continue
            w = self.widgets.get(col)
            if hasattr(w, "text"):
                out[col] = w.text()  # QLineEdit
            if hasattr(w, "toPlainText"):
                # QLineEdit also has toPlainText? no; guard with hasattr
                try:
                    out[col] = w.toPlainText()
                except Exception:
                    pass
        return out

    def is_dirty(self) -> bool:
        return self.collect_row_dict() != (self._loaded_snapshot or {})

    def revert_to_loaded(self) -> None:
        if self._loaded_snapshot:
            self.populate(self._loaded_snapshot)

    def _header(self) -> List[str]:
        return GALLERY_HEADER if self._target == "Gallery" else QUERIES_HEADER

    def _id_col(self) -> str:
        return "gallery_id" if self._target == "Gallery" else "query_id"

    def _rebuild(self, header: List[str]) -> None:
        while self.form.count():
            item = self.form.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.widgets.clear()
        self._build_form(header)
        self._loaded_snapshot = self.collect_row_dict()

    def _build_form(self, header: List[str]) -> None:
        id_col = "gallery_id" if self._target == "Gallery" else "query_id"
        for col in header:
            if col == id_col:
                continue  # HIDE the ID row from the UI
            if col in ("stripe_descriptions", "reticulation_descriptions", "rosette_descriptions",
                       "madreporite_descriptions", "Other_descriptions"):
                w = QTextEdit(); w.setObjectName(col); w.setMinimumHeight(60)
            else:
                w = QLineEdit(); w.setObjectName(col)
                if col in ("diameter_cm", "volume_ml"):
                    w.setValidator(QDoubleValidator())
                if col in ("num_apparent_arms", "num_arms"):
                    w.setValidator(QIntValidator())
            self.widgets[col] = w
            self.form.addRow(QLabel(col), w)
