from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List
import logging

from PySide6.QtCore import Qt, QDate
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QComboBox, QLabel,
    QFileDialog, QListWidget, QListWidgetItem, QLineEdit, QDateEdit,
    QMessageBox, QCheckBox, QPlainTextEdit, QScrollArea, QSizePolicy
)

from PySide6.QtGui import QPixmap

from src.data import archive_paths as ap
from src.data.csv_io import (
    append_row, read_rows_multi, last_row_per_id, normalize_id_value
)
from src.data.id_registry import list_ids, id_exists
from src.data.ingest import ensure_encounter_name, place_images, discover_ids_and_images
from src.data.validators import validate_id
from .metadata_form import MetadataForm

logger = logging.getLogger("starBoard.ui.setup")

# ---------- helpers ----------

def qdate_to_ymd(q) -> Tuple[int, int, int]:
    """Accept either a QDate or a QDateEdit."""
    if hasattr(q, 'date'):
        q = q.date()
    return q.year(), q.month(), q.day()

def info(msg: str, parent: Optional[QWidget] = None) -> None:
    QMessageBox.information(parent, "starBoard", msg)

def warn(msg: str, parent: Optional[QWidget] = None) -> None:
    QMessageBox.warning(parent, "starBoard", msg)

def _csv_paths_for_read(target: str) -> List[Path]:
    """
    Get all plausible metadata CSVs to READ from.
    Prefer ap.metadata_csv_paths_for_read if present; otherwise fall back.
    """
    try:
        return ap.metadata_csv_paths_for_read(target)  # type: ignore[attr-defined]
    except Exception:
        root = ap.archive_root()
        if target.lower() == "gallery":
            return [ap.gallery_root() / "gallery_metadata.csv"]
        candidates = [
            root / "querries" / "querries_metadata.csv",  # legacy
            root / "queries" / "queries_metadata.csv",    # new
        ]
        return [p for p in candidates if p.exists()]

# ---------- minimal image viewer for editing mode ----------
class ImageViewer(QWidget):
    """A minimal image viewer with Prev/Next and Zoom/Fit controls.
    Designed to be lightweight and embedded beside the metadata form.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._images: List[Path] = []
        self._idx: int = -1
        self._scale: float = 1.0
        self._fit: bool = True

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        # Controls
        ctrl = QHBoxLayout()
        self.btn_prev = QPushButton("◀ Prev")
        self.btn_next = QPushButton("Next ▶")
        self.btn_zoom_out = QPushButton("− Zoom")
        self.btn_zoom_in = QPushButton("+ Zoom")
        self.btn_fit = QCheckBox("Fit to window")
        self.btn_fit.setChecked(True)

        for w in (self.btn_prev, self.btn_next, self.btn_zoom_out, self.btn_zoom_in, self.btn_fit):
            ctrl.addWidget(w)
        ctrl.addStretch(1)
        self.lbl_counter = QLabel("0/0")
        ctrl.addWidget(self.lbl_counter)
        outer.addLayout(ctrl)

        # Scroller + label
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.label = QLabel("No image", alignment=Qt.AlignCenter)
        self.label.setWordWrap(True)
        self.scroll.setMinimumSize(320, 240)
        self.scroll.setWidget(self.label)
        outer.addWidget(self.scroll, 1)

        # Connections
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next.clicked.connect(self.next_image)
        self.btn_zoom_in.clicked.connect(lambda: self._set_scale(self._scale * 1.25))
        self.btn_zoom_out.clicked.connect(lambda: self._set_scale(self._scale / 1.25))
        self.btn_fit.toggled.connect(self._on_fit_toggled)

        self._update_ui()

    # ----- public API -----
    def set_images(self, paths: List[Path]):
        self._images = [Path(p) for p in paths if Path(p).exists()]
        self._idx = 0 if self._images else -1
        self._scale = 1.0
        self.btn_fit.setChecked(True)
        self._fit = True
        self._update_ui()
        self._render_current()

    def clear(self):
        self._images = []
        self._idx = -1
        self._scale = 1.0
        self.label.setText("No image")
        self._update_ui()

    # ----- internal helpers -----
    def _update_ui(self):
        n = len(self._images)
        self.lbl_counter.setText(f"{self._idx+1 if self._idx>=0 else 0}/{n}")
        has = (self._idx >= 0) and (self._idx < n)
        self.btn_prev.setEnabled(self._idx > 0)
        self.btn_next.setEnabled(self._idx+1 < n)
        self.btn_zoom_in.setEnabled(has and not self._fit)
        self.btn_zoom_out.setEnabled(has and not self._fit)

    def _on_fit_toggled(self, checked: bool):
        self._fit = checked
        self._render_current()
        self._update_ui()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._fit:
            self._render_current()

    def _set_scale(self, value: float):
        self._scale = max(0.05, min(10.0, value))
        self._render_current()
        self._update_ui()

    def prev_image(self):
        if self._idx > 0:
            self._idx -= 1
            self._render_current()
            self._update_ui()

    def next_image(self):
        if self._idx + 1 < len(self._images):
            self._idx += 1
            self._render_current()
            self._update_ui()

    def _render_current(self):
        if not (0 <= self._idx < len(self._images)):
            # nothing to show
            self.label.setText("No image")
            return

        pm = QPixmap(str(self._images[self._idx]))
        if pm.isNull():
            self.label.setText("Unable to load image.")
            return

        if self._fit:
            avail = self.scroll.viewport().size()
            scaled = pm.scaled(avail, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            w = int(pm.width() * self._scale)
            h = int(pm.height() * self._scale)
            scaled = pm.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.label.setPixmap(scaled)
        self.label.resize(scaled.size())

# ---------- main tab ----------

class TabSetup(QWidget):
    """
    Setup tab with Single Upload, Batch Upload IDs, and Metadata Editing Mode.
    All content is inside a QScrollArea to remain usable at small sizes.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TabSetup")
        self._edit_loaded_once = False  # don't prompt at startup

        # Scroll container
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        content = QWidget()
        content.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        lay = QVBoxLayout(content)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(12)
        self._content_layout = lay

        # Panels
        self.gb_single = self._build_single_upload_group()
        self.gb_batch = self._build_batch_upload_group()
        self.gb_edit  = self._build_editing_group()

        for gb in (self.gb_single, self.gb_batch, self.gb_edit):
            gb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
            lay.addWidget(gb)
        lay.addStretch(1)

        scroll.setWidget(content)
        outer.addWidget(scroll)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # -------------------- Single Upload --------------------
    def _build_single_upload_group(self) -> QGroupBox:
        gb = QGroupBox("Single Upload")
        lay = QVBoxLayout(gb)

        # File picker
        row0 = QHBoxLayout()
        self.btn_choose_files = QPushButton("Choose images…")
        self.btn_choose_files.clicked.connect(self._on_choose_files)
        self.list_files = QListWidget()
        self.list_files.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self.list_files.setMinimumHeight(100)
        self.chk_move = QCheckBox("Move files (instead of copy)")
        row0.addWidget(self.btn_choose_files)
        row0.addWidget(self.chk_move)
        lay.addLayout(row0)
        lay.addWidget(self.list_files)

        # Metadata-only toggle
        self.chk_metadata_only = QCheckBox("Save metadata only (no images)")
        self.chk_metadata_only.setToolTip("Append a metadata row even if no images are selected.")
        self.chk_metadata_only.toggled.connect(self._on_metadata_only_toggled)
        lay.addWidget(self.chk_metadata_only)

        # Target + ID
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Archive:"))
        self.cmb_target = QComboBox()
        self.cmb_target.addItems(["Gallery", "Queries"])
        self.cmb_target.currentIndexChanged.connect(self._refresh_id_list_single)
        row1.addWidget(self.cmb_target, 1)

        row1.addWidget(QLabel("ID:"))
        self.cmb_id = QComboBox()
        self.cmb_id.currentIndexChanged.connect(self._on_id_selection_changed_single)
        row1.addWidget(self.cmb_id, 1)

        self.edit_new_id = QLineEdit()
        self.edit_new_id.setPlaceholderText("New ID")
        self.edit_new_id.setVisible(False)
        row1.addWidget(self.edit_new_id, 1)
        lay.addLayout(row1)

        # Encounter date + suffix (used when files are present)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Encounter date:"))
        self.date_encounter = QDateEdit()
        self.date_encounter.setCalendarPopup(True)
        self.date_encounter.setDate(QDate.currentDate())
        self.date_encounter.dateChanged.connect(lambda _:
            self._update_encounter_preview())
        row2.addWidget(self.date_encounter)

        row2.addWidget(QLabel("Suffix (optional):"))
        self.edit_suffix = QLineEdit()
        self.edit_suffix.setPlaceholderText("e.g., 'pm2'")
        self.edit_suffix.textChanged.connect(lambda _:
            self._update_encounter_preview())
        row2.addWidget(self.edit_suffix)

        row2.addWidget(QLabel("Encounter preview:"))
        self.lbl_preview = QLabel("")
        row2.addWidget(self.lbl_preview, 1)
        lay.addLayout(row2)

        # Metadata form
        self.meta_form = MetadataForm()
        self.meta_form.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        lay.addWidget(self.meta_form)

        # Buttons row
        row3 = QHBoxLayout()
        row3.addStretch(1)
        self.btn_save_single = QPushButton("Save")
        self.btn_save_single.clicked.connect(self._on_save_single)
        row3.addWidget(self.btn_save_single)
        lay.addLayout(row3)

        # Log
        self.log_single = QPlainTextEdit()
        self.log_single.setReadOnly(True)
        self.log_single.setMaximumHeight(120)
        lay.addWidget(self.log_single)

        self._refresh_id_list_single()
        self._update_encounter_preview()
        self._set_encounter_controls_enabled(True)
        return gb

    def _on_metadata_only_toggled(self, checked: bool) -> None:
        # Disable encounter controls when metadata-only is selected
        self._set_encounter_controls_enabled(not checked)

    def _set_encounter_controls_enabled(self, enabled: bool) -> None:
        self.date_encounter.setEnabled(enabled)
        self.edit_suffix.setEnabled(enabled)
        self.lbl_preview.setEnabled(enabled)

    def _on_choose_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select images", "", "Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp)"
        )
        self.list_files.clear()
        for f in files:
            self.list_files.addItem(QListWidgetItem(f))
        logger.info("Selected %d files", len(files))
        self._log_single(f"Selected {len(files)} files.")

    def _refresh_id_list_single(self):
        target = self.cmb_target.currentText()
        self.meta_form.set_target(target)
        self.cmb_id.blockSignals(True)
        self.cmb_id.clear()
        ids = list_ids(target)
        self.cmb_id.addItems(["➕ New ID…"] + ids)
        self.cmb_id.blockSignals(False)
        logger.info("Refreshed ID list for target=%s (n=%d)", target, len(ids))
        self._on_id_selection_changed_single()

    def _on_id_selection_changed_single(self):
        use_new = (self.cmb_id.currentIndex() == 0)
        self.edit_new_id.setVisible(use_new)
        id_val = self.edit_new_id.text().strip() if use_new else self.cmb_id.currentText()
        self.meta_form.set_id_value("" if id_val == "➕ New ID…" else id_val)
        if not use_new and id_val:
            self._populate_metadata_from_csv(self.cmb_target.currentText(), id_val)
        else:
            self.meta_form.populate({})

    def _update_encounter_preview(self):
        y, m, d = qdate_to_ymd(self.date_encounter)
        name = ensure_encounter_name(y, m, d, suffix=self.edit_suffix.text().strip())
        self.lbl_preview.setText(name)
        logger.debug("Encounter preview set to %s", name)

    def _populate_metadata_from_csv(self, target: str, id_val: str):
        # Load prior values for this ID from CSV(s) if present
        id_col = ap.id_column_name(target)
        csv_paths = _csv_paths_for_read(target)
        rows = read_rows_multi(csv_paths)
        latest_map = last_row_per_id(rows, id_col)
        data = latest_map.get(normalize_id_value(id_val), {})
        data[id_col] = id_val
        self.meta_form.populate(data)

    def _on_save_single(self):
        target = self.cmb_target.currentText()

        # Use either the existing ID or a new one
        if self.cmb_id.currentIndex() == 0:
            id_val = self.edit_new_id.text().strip()
            if not id_val:
                warn("Please type a new ID or pick an existing one.", self)
                return
        else:
            id_val = self.cmb_id.currentText()

        # Validate ID
        v = validate_id(target, id_val)
        if not v.ok:
            warn(v.message, self)
            return

        files = [self.list_files.item(i).text() for i in range(self.list_files.count())]
        metadata_only = self.chk_metadata_only.isChecked()

        if not files and not metadata_only:
            warn("Please choose at least one image or check 'Save metadata only (no images)'.", self)
            return

        # Always append metadata (append-only)
        csv_path, header = ap.metadata_csv_for(target)
        row = self.meta_form.collect_row()
        row[ap.id_column_name(target)] = id_val
        append_row(csv_path, header, row)

        if not metadata_only:
            # Place images into archive
            y, m, d = qdate_to_ymd(self.date_encounter)
            enc_name = ensure_encounter_name(y, m, d, self.edit_suffix.text().strip())
            root = ap.root_for(target)
            report = place_images(root, id_val, enc_name, [Path(f) for f in files], move=self.chk_move.isChecked())
            n_ops = len(report.ops)
            n_renamed = sum(1 for op in report.ops if op.renamed)
            self._log_single(f"Saved: {n_ops} files to {root / id_val / enc_name} ({n_renamed} renamed).")
            if report.errors:
                self._log_single("Errors:\n - " + "\n - ".join(report.errors))
        else:
            logger.info("Metadata-only save: target=%s id=%s (no images)", target, id_val)
            self._log_single(f"Metadata saved for {target} ID '{id_val}' (no images).")

        self._refresh_id_list_single()

    def _log_single(self, msg: str):
        logger.info(msg)
        self.log_single.appendPlainText(msg)

    # -------------------- Batch Upload IDs --------------------
    def _build_batch_upload_group(self) -> QGroupBox:
        gb = QGroupBox("Batch Upload IDs")
        lay = QVBoxLayout(gb)

        # Pick base folder to scan
        row = QHBoxLayout()
        row.addWidget(QLabel("Archive:"))
        self.cmb_target_batch = QComboBox()
        self.cmb_target_batch.addItems(["Gallery", "Queries"])
        row.addWidget(self.cmb_target_batch)
        self.btn_discover = QPushButton("Discover IDs…")
        self.btn_discover.clicked.connect(self._on_discover)
        row.addWidget(self.btn_discover)
        lay.addLayout(row)

        # Results list
        self.list_discovered = QListWidget()
        self.list_discovered.setMinimumHeight(120)
        lay.addWidget(self.list_discovered)

        # Encounter & suffix for batch placement
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Encounter date:"))
        self.date_batch = QDateEdit()
        self.date_batch.setCalendarPopup(True)
        self.date_batch.setDate(QDate.currentDate())
        row2.addWidget(self.date_batch)
        row2.addWidget(QLabel("Suffix (optional):"))
        self.edit_suffix_batch = QLineEdit()
        self.edit_suffix_batch.setPlaceholderText("e.g., 'am2'")
        row2.addWidget(self.edit_suffix_batch)
        lay.addLayout(row2)

        # Buttons
        row3 = QHBoxLayout()
        row3.addStretch(1)
        self.btn_start_batch = QPushButton("Start batch")
        self.btn_start_batch.clicked.connect(self._on_start_batch)
        row3.addWidget(self.btn_start_batch)
        lay.addLayout(row3)

        # Log
        self.log_batch = QPlainTextEdit()
        self.log_batch.setReadOnly(True)
        self.log_batch.setMaximumHeight(120)
        lay.addWidget(self.log_batch)
        return gb

    def _on_discover(self):
        parent = QFileDialog.getExistingDirectory(self, "Select base folder to scan")
        if not parent:
            return
        self.list_discovered.clear()
        items = discover_ids_and_images(parent)
        for id_str, files in items:
            it = QListWidgetItem(f"{id_str}  —  {len(files)} images")
            it.setData(Qt.UserRole, (id_str, [str(p) for p in files]))
            self.list_discovered.addItem(it)
        logger.info("Discovered %d IDs under %s", len(items), str(parent))
        self._log_batch(f"Discovered {len(items)} IDs.")

    def _on_start_batch(self):
        if self.list_discovered.count() == 0:
            warn("Nothing discovered to ingest.", self)
            return
        target = self.cmb_target_batch.currentText()
        y, m, d = qdate_to_ymd(self.date_batch)
        enc = ensure_encounter_name(y, m, d, self.edit_suffix_batch.text().strip())
        root = ap.root_for(target)
        csv_path, header = ap.metadata_csv_for(target)
        id_col = ap.id_column_name(target)

        logger.info("Batch start: target=%s enc=%s count=%d", target, enc, self.list_discovered.count())

        for i in range(self.list_discovered.count()):
            item = self.list_discovered.item(i)
            id_str, files = item.data(Qt.UserRole)
            exists = id_exists(target, id_str)
            rep = place_images(root, id_str, enc, [Path(f) for f in files], move=False)
            if not exists:
                row = {col: "" for col in header}
                row[id_col] = id_str
                append_row(csv_path, header, row)
                self._log_batch(f"Created new ID {id_str}: {len(rep.ops)} images.")
            else:
                self._log_batch(f"Appended to existing ID {id_str}: {len(rep.ops)} images.")
            if rep.errors:
                self._log_batch("Errors:\n - " + "\n - ".join(rep.errors))

        info("Batch complete.", self)

    def _log_batch(self, msg: str):
        logger.info(msg)
        self.log_batch.appendPlainText(msg)

    # -------------------- Metadata Editing Mode --------------------
    def _build_editing_group(self) -> QGroupBox:
        gb = QGroupBox("Metadata Editing Mode")
        lay = QVBoxLayout(gb)

        row = QHBoxLayout()
        row.addWidget(QLabel("Archive:"))
        self.cmb_target_edit = QComboBox()
        self.cmb_target_edit.addItems(["Gallery", "Queries"])
        self.cmb_target_edit.currentIndexChanged.connect(self._refresh_id_list_edit)
        row.addWidget(self.cmb_target_edit)

        row.addWidget(QLabel("ID:"))
        self.cmb_id_edit = QComboBox()
        self.cmb_id_edit.currentIndexChanged.connect(self._on_edit_id_changed)
        row.addWidget(self.cmb_id_edit, 1)
        lay.addLayout(row)

        # Split pane: metadata form (left) + image viewer (right)
        self.meta_form_edit = MetadataForm()
        self.meta_form_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.viewer_edit = ImageViewer()
        self.viewer_edit.setObjectName("edit_image_viewer")
        self.viewer_edit.setMinimumSize(320, 240)
        split = QHBoxLayout()
        split.setContentsMargins(0, 0, 0, 0)
        split.setSpacing(12)
        split.addWidget(self.meta_form_edit, 2)
        split.addWidget(self.viewer_edit, 3)
        lay.addLayout(split)

        row2 = QHBoxLayout()
        self.btn_save_edit = QPushButton("Save edits")
        self.btn_save_edit.clicked.connect(self._on_save_edits)
        row2.addStretch(1)
        self.btn_save_edit.setDefault(True)
        row2.addWidget(self.btn_save_edit)
        lay.addLayout(row2)

        self._refresh_id_list_edit()
        return gb

    def _refresh_id_list_edit(self):
        target = self.cmb_target_edit.currentText()
        self.meta_form_edit.set_target(target)
        ids = list_ids(target)
        self.cmb_id_edit.blockSignals(True)
        self.cmb_id_edit.clear()
        self.cmb_id_edit.addItems(ids)
        self.cmb_id_edit.blockSignals(False)
        self.btn_save_edit.setEnabled(bool(ids))
        self._on_edit_id_changed()

    def _on_edit_id_changed(self):
        target = self.cmb_target_edit.currentText()
        next_id = self.cmb_id_edit.currentText()
        if not next_id:
            self.meta_form_edit.populate({})
            if hasattr(self, 'viewer_edit'):
                self.viewer_edit.clear()
            return

        # Only prompt after first successful load
        if self._edit_loaded_once and self.meta_form_edit.is_dirty():
            res = QMessageBox.question(
                self, "Save before changing?", "Save before changing?",
                QMessageBox.Yes | QMessageBox.No
            )
            if res == QMessageBox.Yes:
                self._on_save_edits()

        # Load values
        self.meta_form_edit.set_id_value(next_id)
        self._populate_metadata_from_csv_edit(target, next_id)
        # Update image viewer for this ID
        self.viewer_edit.set_images(self._gather_images_for_id(target, next_id))

        if not self._edit_loaded_once:
            self._edit_loaded_once = True

    def _populate_metadata_from_csv_edit(self, target: str, id_val: str):
        id_col = ap.id_column_name(target)
        csv_paths = _csv_paths_for_read(target)
        rows = read_rows_multi(csv_paths)
        latest_map = last_row_per_id(rows, id_col)
        data = latest_map.get(normalize_id_value(id_val), {})
        data[id_col] = id_val
        self.meta_form_edit.populate(data)

    def _gather_images_for_id(self, target: str, id_val: str) -> List[Path]:
        """Return a list of image files for *id_val* under the archive *target*.
        The expected layout is <root>/<ID>/<encounter>/*.ext (as produced by place_images).
        If both the raw id and its normalized form exist as directories, both are searched.
        """
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"}
        root = ap.root_for(target)
        candidates = [root / id_val, root / normalize_id_value(id_val)]
        out: List[Path] = []
        seen = set()
        for base in candidates:
            if base.exists() and base.is_dir():
                # Sort encounters lexicographically so recent ones (YYYY-MM-DD*) come last
                for p in sorted(base.rglob("*")):
                    if p.is_file() and p.suffix.lower() in exts:
                        rp = p.resolve()
                        if rp not in seen:
                            out.append(rp)
                            seen.add(rp)
        return out

    def _on_save_edits(self):
        target = self.cmb_target_edit.currentText()
        id_val = self.cmb_id_edit.currentText()
        if not id_val:
            warn("No ID selected.", self)
            return
        csv_path, header = ap.metadata_csv_for(target)
        row = self.meta_form_edit.collect_row()
        row[ap.id_column_name(target)] = id_val
        append_row(csv_path, header, row)
        logger.info("Saved edits for target=%s id=%s", target, id_val)
        info("Edits saved.", self)
