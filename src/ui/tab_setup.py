# src/ui/tab_setup.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List, Dict
import logging
from datetime import date as _date
from PySide6.QtCore import Qt, QDate, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QComboBox, QLabel,
    QFileDialog, QListWidget, QListWidgetItem, QLineEdit, QDateEdit,
    QMessageBox, QCheckBox, QPlainTextEdit, QScrollArea, QSizePolicy,
    QTabWidget, QCompleter,
)

from src.ui.collapsible import CollapsibleSection  # existing utility for collapsible panels
from src.data import archive_paths as ap
from src.data.csv_io import (
    append_row, read_rows_multi, last_row_per_id, normalize_id_value
)
from src.data.id_registry import list_ids, id_exists
from src.data.ingest import ensure_encounter_name, place_images, discover_ids_and_images
from src.data.batch_undo import (
    generate_batch_id, record_batch_upload, list_batches,
    undo_batch, redo_batch, check_redo_sources, BatchInfo,
)
from src.data.validators import validate_id
from src.data.archive_paths import last_observation_for_all
from src.data.best_photo import reorder_files_with_best, save_best_for_id
from src.data.image_index import list_image_files
from src.data.encounter_info import list_encounters_for_id, get_encounter_date, set_encounter_date
from .metadata_form_v2 import MetadataForm
from src.utils.interaction_logger import get_interaction_logger

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


# ---------- interactive image viewer (uses ImageStrip for consistent UX) ----------

from PySide6.QtCore import Signal


class ImageViewer(QWidget):
    """
    Image viewer for the *Metadata Editing Mode*.

    This is a thin wrapper around ImageStrip to provide:
      - Identical user interactions to First-order/Second-order tabs
      - Mouse wheel zoom (always enabled, anchored at cursor)
      - Click-and-drag panning
      - Hold R + drag rotation
      - Auto-upgrade to full resolution when zoomed in
      - Eyedropper support for color picking
      - "Best" button for marking the current image as best

    The wrapper maintains API compatibility with TabSetup's usage.
    """

    # Signal emitted when Best button is clicked
    bestRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ImageViewer")
        self._ilog = get_interaction_logger()

        # Import ImageStrip here to avoid circular imports
        from .image_strip import ImageStrip

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        # Embed the ImageStrip - it provides all the rendering and interaction
        self._strip = ImageStrip(files=[], long_edge=768)
        outer.addWidget(self._strip, 1)

        # Footer row with Best button
        footer = QHBoxLayout()
        footer.setContentsMargins(4, 0, 4, 0)
        self.btn_best = QPushButton("Best")
        self.btn_best.setToolTip("Mark current image as the 'best' for this ID (appears first)")
        self.btn_best.setEnabled(False)
        self.btn_best.clicked.connect(self._on_best_clicked)
        footer.addStretch(1)
        footer.addWidget(self.btn_best)
        outer.addLayout(footer)

        # Internal reference to images for API compatibility
        self._images: List[Path] = []

    def _on_best_clicked(self):
        """Emit signal when Best button is clicked."""
        self._ilog.log("button_click", "btn_best_viewer", value="clicked")
        self.bestRequested.emit()

    # ----- public API (compatible with TabSetup usage) -----
    def clear(self):
        """Clear all images."""
        self._images = []
        self._strip.set_files([])
        self.btn_best.setEnabled(False)

    def set_images(self, paths: List[Path]):
        """Set the list of image paths to display."""
        self._images = [Path(p) for p in paths if Path(p).exists()]
        self._strip.set_files(self._images)
        self.btn_best.setEnabled(bool(self._images))

    def set_index(self, idx: int):
        """Set the current image index."""
        if 0 <= idx < len(self._images):
            self._strip.idx = idx
            self._strip._show_current(reset_view=True)

    def current_index(self) -> int:
        """Return the current zero-based index, or -1 when empty."""
        if not self._images:
            return -1
        return self._strip.idx

    def current_path(self) -> Optional[Path]:
        """Return the current image Path (or None if empty)."""
        if not self._images:
            return None
        idx = self._strip.idx
        if 0 <= idx < len(self._images):
            return self._images[idx]
        return None

    # ----- ImageStrip delegation for eyedropper compatibility -----
    @property
    def files(self) -> List[Path]:
        """Expose files for compatibility with eyedropper search."""
        return self._images

    @property
    def view(self):
        """Expose the QGraphicsView for eyedropper compatibility."""
        return self._strip.view

    def start_eyedropper(self):
        """Start eyedropper mode (delegated to ImageStrip)."""
        self._strip.start_eyedropper()

    def stop_eyedropper(self):
        """Stop eyedropper mode (delegated to ImageStrip)."""
        self._strip.stop_eyedropper()

    @property
    def eyedropperColorPicked(self):
        """Expose signal for eyedropper color picked."""
        return self._strip.eyedropperColorPicked

    @property
    def eyedropperCancelled(self):
        """Expose signal for eyedropper cancelled."""
        return self._strip.eyedropperCancelled


# ---------- Batch Undo/Redo Dialogs ----------

class UndoBatchDialog(QMessageBox):
    """Confirmation dialog for undoing a batch upload."""
    
    def __init__(self, batch: BatchInfo, parent=None):
        super().__init__(parent)
        self.batch = batch
        self.permanent = False
        
        self.setWindowTitle("Undo Batch Upload?")
        self.setIcon(QMessageBox.Icon.Warning)
        
        # Build message
        ids_preview = ", ".join(batch.new_ids[:5])
        if len(batch.new_ids) > 5:
            ids_preview += f"… (+{len(batch.new_ids) - 5} more)"
        
        msg = (
            f"This will remove from the archive:\n"
            f"  • {batch.file_count} image files\n"
            f"  • {len(batch.new_ids)} new ID entries\n\n"
            f"Batch: {batch.timestamp.strftime('%m/%d/%y %I:%M %p')}\n"
            f"Encounter: {batch.encounter_name}\n"
        )
        if batch.new_ids:
            msg += f"New IDs: {ids_preview}\n"
        
        self.setText(msg)
        
        # Add checkbox for permanent delete
        self.chk_permanent = QCheckBox("Permanently delete (cannot be undone)")
        self.chk_permanent.toggled.connect(self._on_permanent_toggled)
        self.setCheckBox(self.chk_permanent)
        
        # Buttons
        self.setStandardButtons(QMessageBox.StandardButton.Cancel)
        self.btn_confirm = self.addButton("Undo", QMessageBox.ButtonRole.AcceptRole)
        self.setDefaultButton(QMessageBox.StandardButton.Cancel)
    
    def _on_permanent_toggled(self, checked: bool):
        self.permanent = checked
        if checked:
            self.btn_confirm.setText("Delete Permanently")
            self.btn_confirm.setStyleSheet("background-color: #c62828; color: white;")
        else:
            self.btn_confirm.setText("Undo")
            self.btn_confirm.setStyleSheet("")
    
    def exec(self) -> int:
        result = super().exec()
        # Map AcceptRole to Accepted
        if self.clickedButton() == self.btn_confirm:
            return QMessageBox.DialogCode.Accepted
        return QMessageBox.DialogCode.Rejected


class RedoBatchDialog(QMessageBox):
    """Confirmation dialog for redoing a batch upload."""
    
    def __init__(self, batch: BatchInfo, missing_count: int = 0, parent=None):
        super().__init__(parent)
        self.batch = batch
        
        self.setWindowTitle("Redo Batch Upload?")
        self.setIcon(QMessageBox.Icon.Question)
        
        msg = (
            f"This will restore:\n"
            f"  • {batch.file_count} image files (re-copied from source)\n"
            f"  • {len(batch.new_ids)} ID entries\n"
        )
        
        if missing_count > 0:
            msg += f"\n⚠️ {missing_count} source files no longer exist and will be skipped."
        
        self.setText(msg)
        
        # Buttons
        self.setStandardButtons(QMessageBox.StandardButton.Cancel)
        self.btn_confirm = self.addButton("Redo", QMessageBox.ButtonRole.AcceptRole)
        self.setDefaultButton(self.btn_confirm)
    
    def exec(self) -> int:
        result = super().exec()
        if self.clickedButton() == self.btn_confirm:
            return QMessageBox.DialogCode.Accepted
        return QMessageBox.DialogCode.Rejected


# ---------- main tab ----------

class TabSetup(QWidget):
    """
    Setup tab with Single Upload, Batch Upload IDs, and Metadata Editing Mode.
    All content is inside a QScrollArea to remain usable at small sizes.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TabSetup")

        # state for edit-mode prompts & carry-over
        self._edit_loaded_once = False
        self._last_edit_id: str = ""
        self._carry_over: Dict[str, str] = {}
        self._ilog = get_interaction_logger()

        # Scroll container
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        content = QWidget()
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay = QVBoxLayout(content)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(12)

        # Build inner groups (title-less QGroupBox used for visual framing)
        gb_single = self._build_single_upload_group()   # title-less
        gb_batch  = self._build_batch_upload_group()    # title-less
        gb_query_manage = self._build_query_management_group()  # title-less
        gb_edit   = self._build_editing_group()         # title-less

        # Wrap each group with an expandable panel (collapsed by default)
        sec_single = CollapsibleSection("Single Upload", start_collapsed=True)
        sec_single.setContent(gb_single)
        sec_batch = CollapsibleSection("Batch Upload IDs", start_collapsed=True)
        sec_batch.setContent(gb_batch)
        sec_query_manage = CollapsibleSection("Query Management", start_collapsed=True)
        sec_query_manage.setContent(gb_query_manage)
        sec_edit = CollapsibleSection("Metadata Editing Mode", start_collapsed=True)
        sec_edit.setContent(gb_edit)

        # Single/Batch/Query Management sections stay compact; Metadata Editing expands to fill space
        for sec in (sec_single, sec_batch, sec_query_manage):
            sec.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
            lay.addWidget(sec)
        
        # Metadata Editing Mode should expand vertically to fill available space
        sec_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(sec_edit, 1)  # stretch factor 1 to take remaining space

        scroll.setWidget(content)
        outer.addWidget(scroll)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # -------------------- Single Upload --------------------
    def _build_single_upload_group(self) -> QGroupBox:
        gb = QGroupBox("")  # title provided by CollapsibleSection
        lay = QVBoxLayout(gb)

        # File picker
        row0 = QHBoxLayout()
        self.btn_choose_files = QPushButton("Choose images…")
        self.btn_choose_files.clicked.connect(self._on_choose_files)
        self.list_files = QListWidget()
        self.list_files.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self.list_files.setMinimumHeight(100)
        self.chk_move = QCheckBox("Move files (instead of copy)")
        self.chk_move.toggled.connect(
            lambda checked: self._ilog.log("checkbox_toggle", "chk_move", value=str(checked)))
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
        self.cmb_target.currentIndexChanged.connect(
            lambda: self._ilog.log("combo_change", "cmb_target_single", value=self.cmb_target.currentText()))
        row1.addWidget(self.cmb_target, 1)

        row1.addWidget(QLabel("ID:"))
        self.cmb_id = QComboBox()
        # Make combo editable for type-to-search functionality
        self.cmb_id.setEditable(True)
        self.cmb_id.setInsertPolicy(QComboBox.NoInsert)
        self.cmb_id.completer().setFilterMode(Qt.MatchContains)
        self.cmb_id.completer().setCompletionMode(QCompleter.PopupCompletion)
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
        self.date_encounter.dateChanged.connect(lambda _: self._update_encounter_preview())
        row2.addWidget(self.date_encounter)

        row2.addWidget(QLabel("Suffix (optional):"))
        self.edit_suffix = QLineEdit()
        self.edit_suffix.setPlaceholderText("e.g. 'pm2'")
        self.edit_suffix.textChanged.connect(lambda _: self._update_encounter_preview())
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
        self._ilog.log("checkbox_toggle", "chk_metadata_only", value=str(checked))
        self._set_encounter_controls_enabled(not checked)

    def _set_encounter_controls_enabled(self, enabled: bool) -> None:
        self.date_encounter.setEnabled(enabled)
        self.edit_suffix.setEnabled(enabled)
        self.lbl_preview.setEnabled(enabled)

    def _on_choose_files(self):
        self._ilog.log("button_click", "btn_choose_files", value="clicked")
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
        if target == "Queries":
            last_obs = last_observation_for_all("Queries")
            ids = sorted(ids,
                         key=lambda qid: ((last_obs.get(qid) is None), (last_obs.get(qid) or _date.max), qid.lower()))
        self.cmb_id.addItems(["➕ New ID…"] + ids)
        self.cmb_id.blockSignals(False)
        logger.info("Refreshed ID list for target=%s (n=%d)", target, len(ids))
        self._on_id_selection_changed_single()

    def _on_id_selection_changed_single(self):
        use_new = (self.cmb_id.currentIndex() == 0)
        self.edit_new_id.setVisible(use_new)
        id_val = self.edit_new_id.text().strip() if use_new else self.cmb_id.currentText()
        self._ilog.log("combo_change", "cmb_id_single", value=id_val if not use_new else "new")
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
        id_col = ap.id_column_name(target)
        csv_paths = _csv_paths_for_read(target)
        rows = read_rows_multi(csv_paths)
        latest_map = last_row_per_id(rows, id_col)
        data = latest_map.get(normalize_id_value(id_val), {})
        data[id_col] = id_val
        self.meta_form.populate(data)

    def _on_save_single(self):
        target = self.cmb_target.currentText()
        self._ilog.log("button_click", "btn_save_single", value=target)

        # Use either the existing ID or a new one
        if self.cmb_id.currentIndex() == 0:
            id_val = self.edit_new_id.text().strip()
            if not id_val:
                warn("Please type a new ID or pick an existing one.", self)
                return
        else:
            id_val = self.cmb_id.currentText()

        # Validate ID
        v = validate_id(target)
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

        # Notify First‑order so it reflects new metadata immediately
        self._notify_first_order_refresh()
        self._refresh_id_list_single()

    def _log_single(self, msg: str):
        logger.info(msg)
        self.log_single.appendPlainText(msg)

    # -------------------- Batch Upload IDs --------------------
    def _build_batch_upload_group(self) -> QGroupBox:
        gb = QGroupBox("")  # title provided by CollapsibleSection
        lay = QVBoxLayout(gb)

        # Pick base folder to scan
        row = QHBoxLayout()
        row.addWidget(QLabel("Archive:"))
        self.cmb_target_batch = QComboBox()
        self.cmb_target_batch.addItems(["Gallery", "Queries"])
        self.cmb_target_batch.currentIndexChanged.connect(
            lambda: self._ilog.log("combo_change", "cmb_target_batch", value=self.cmb_target_batch.currentText()))
        row.addWidget(self.cmb_target_batch)
        self.btn_discover = QPushButton("Discover IDs…")
        self.btn_discover.clicked.connect(self._on_discover)
        row.addWidget(self.btn_discover)
        lay.addLayout(row)

        # Results list
        self.list_discovered = QListWidget()
        self.list_discovered.setMinimumHeight(120)
        lay.addWidget(self.list_discovered)

        # ID prefix/suffix for batch naming
        id_ext_row = QHBoxLayout()
        id_ext_row.addWidget(QLabel("ID prefix:"))
        self.edit_id_prefix = QLineEdit()
        self.edit_id_prefix.setPlaceholderText("e.g. 'dec22_trip1_'")
        self.edit_id_prefix.textChanged.connect(self._update_id_preview)
        id_ext_row.addWidget(self.edit_id_prefix)
        
        id_ext_row.addWidget(QLabel("ID suffix:"))
        self.edit_id_suffix = QLineEdit()
        self.edit_id_suffix.setPlaceholderText("e.g. '_obs1'")
        self.edit_id_suffix.textChanged.connect(self._update_id_preview)
        id_ext_row.addWidget(self.edit_id_suffix)
        lay.addLayout(id_ext_row)
        
        # ID preview label
        self.lbl_id_preview = QLabel("")
        self.lbl_id_preview.setStyleSheet("color: gray; font-style: italic;")
        lay.addWidget(self.lbl_id_preview)

        # Encounter & suffix for batch placement
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Encounter date:"))
        self.date_batch = QDateEdit()
        self.date_batch.setCalendarPopup(True)
        self.date_batch.setDate(QDate.currentDate())
        row2.addWidget(self.date_batch)
        row2.addWidget(QLabel("Suffix (optional):"))
        self.edit_suffix_batch = QLineEdit()
        self.edit_suffix_batch.setPlaceholderText("e.g. 'am2'")
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

        # Undo/Redo controls
        undo_row = QHBoxLayout()
        
        self.btn_undo_batch = QPushButton("Undo")
        self.btn_undo_batch.setFixedWidth(70)
        self.btn_undo_batch.clicked.connect(self._on_undo_batch)
        self.btn_undo_batch.setEnabled(False)
        undo_row.addWidget(self.btn_undo_batch)
        
        self.btn_redo_batch = QPushButton("Redo")
        self.btn_redo_batch.setFixedWidth(70)
        self.btn_redo_batch.clicked.connect(self._on_redo_batch)
        self.btn_redo_batch.setEnabled(False)
        undo_row.addWidget(self.btn_redo_batch)
        
        self.cmb_batch_history = QComboBox()
        self.cmb_batch_history.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cmb_batch_history.currentIndexChanged.connect(self._on_batch_selection_changed)
        undo_row.addWidget(self.cmb_batch_history)
        
        lay.addLayout(undo_row)
        
        # Connect target change to refresh batch history and ID preview
        self.cmb_target_batch.currentIndexChanged.connect(self._refresh_batch_history)
        self.cmb_target_batch.currentIndexChanged.connect(self._update_id_preview)
        
        # Initial load of batch history
        QTimer.singleShot(100, self._refresh_batch_history)
        
        return gb

    def _on_discover(self):
        self._ilog.log("button_click", "btn_discover", value="clicked")
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
        self._update_id_preview()

    def _transform_id(self, original_id: str) -> str:
        """Apply prefix and suffix to an ID."""
        prefix = self.edit_id_prefix.text()
        suffix = self.edit_id_suffix.text()
        return f"{prefix}{original_id}{suffix}"

    def _update_id_preview(self):
        """Update the ID transformation preview and conflict indicators."""
        if self.list_discovered.count() == 0:
            self.lbl_id_preview.setText("")
            return
        
        # Show preview of first ID transformation
        first_item = self.list_discovered.item(0)
        original_id, _ = first_item.data(Qt.UserRole)
        transformed = self._transform_id(original_id)
        
        # Count conflicts
        existing_count, new_count = 0, 0
        target = self.cmb_target_batch.currentText()
        for i in range(self.list_discovered.count()):
            item = self.list_discovered.item(i)
            orig_id, _ = item.data(Qt.UserRole)
            trans_id = self._transform_id(orig_id)
            if id_exists(target, trans_id):
                existing_count += 1
            else:
                new_count += 1
        
        # Build preview text
        preview_parts = []
        if original_id != transformed:
            preview_parts.append(f'"{original_id}" → "{transformed}"')
        
        if existing_count > 0:
            preview_parts.append(f"⚠️ {existing_count} already exist")
        
        self.lbl_id_preview.setText("  |  ".join(preview_parts) if preview_parts else "")

    def _check_id_conflicts(self) -> tuple[list[str], list[str]]:
        """
        Check which transformed IDs already exist.
        
        Returns:
            (existing_ids, new_ids)
        """
        target = self.cmb_target_batch.currentText()
        existing_ids = []
        new_ids = []
        
        for i in range(self.list_discovered.count()):
            item = self.list_discovered.item(i)
            original_id, _ = item.data(Qt.UserRole)
            transformed_id = self._transform_id(original_id)
            
            if id_exists(target, transformed_id):
                existing_ids.append(transformed_id)
            else:
                new_ids.append(transformed_id)
        
        return existing_ids, new_ids

    def _on_start_batch(self):
        if self.list_discovered.count() == 0:
            warn("Nothing discovered to ingest.", self)
            return
        target = self.cmb_target_batch.currentText()
        self._ilog.log("button_click", "btn_start_batch", value=target,
                      context={"count": self.list_discovered.count()})
        
        # Check for ID conflicts
        existing_ids, new_id_list = self._check_id_conflicts()
        if existing_ids:
            msg = f"The following IDs already exist in {target}:\n\n"
            msg += "\n".join(f"  • {id_}" for id_ in existing_ids[:10])
            if len(existing_ids) > 10:
                msg += f"\n  ... and {len(existing_ids) - 10} more"
            msg += f"\n\nContinuing will ADD images to these existing IDs.\n"
            msg += f"New IDs will be created for the rest.\n\n"
            msg += f"Total: {len(existing_ids)} existing, {len(new_id_list)} new"
            
            reply = QMessageBox.warning(
                self,
                "Some IDs already exist",
                msg,
                QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Ok,
                QMessageBox.StandardButton.Cancel
            )
            if reply != QMessageBox.StandardButton.Ok:
                self._log_batch("Batch cancelled due to ID conflicts.")
                return
        
        y, m, d = qdate_to_ymd(self.date_batch)
        enc = ensure_encounter_name(y, m, d, self.edit_suffix_batch.text().strip())
        root = ap.root_for(target)
        csv_path, header = ap.metadata_csv_for(target)
        id_col = ap.id_column_name(target)

        # Generate batch ID for undo/redo tracking
        batch_id = generate_batch_id()
        all_file_ops: list[tuple[Path, Path]] = []
        new_ids: set[str] = set()

        logger.info("Batch start: target=%s enc=%s count=%d batch_id=%s", 
                    target, enc, self.list_discovered.count(), batch_id)

        for i in range(self.list_discovered.count()):
            item = self.list_discovered.item(i)
            original_id, files = item.data(Qt.UserRole)
            id_str = self._transform_id(original_id)  # Apply prefix/suffix
            exists = id_exists(target, id_str)
            rep = place_images(root, id_str, enc, [Path(f) for f in files], move=False)
            
            # Track file operations for batch history
            for op in rep.ops:
                all_file_ops.append((op.src, op.dest))
            
            if not exists:
                new_ids.add(id_str)
                row = {col: "" for col in header}
                row[id_col] = id_str
                append_row(csv_path, header, row)
                self._log_batch(f"Created new ID {id_str}: {len(rep.ops)} images.")
            else:
                self._log_batch(f"Appended to existing ID {id_str}: {len(rep.ops)} images.")
            if rep.errors:
                self._log_batch("Errors:\n - " + "\n - ".join(rep.errors))

        # Record batch for undo/redo
        if all_file_ops:
            record_batch_upload(target, batch_id, all_file_ops, new_ids, enc)
            self._log_batch(f"Batch recorded (ID: {batch_id[:20]}…)")

        info("Batch complete.", self)
        # First‑order refresh: new IDs/rows may be available
        self._notify_first_order_refresh()
        # Refresh batch history combo
        self._refresh_batch_history()

    def _log_batch(self, msg: str):
        logger.info(msg)
        self.log_batch.appendPlainText(msg)

    # -------------------- Batch Undo/Redo --------------------
    def _refresh_batch_history(self):
        """Reload the batch history combo box."""
        target = self.cmb_target_batch.currentText()
        batches = list_batches(target)
        
        self.cmb_batch_history.blockSignals(True)
        self.cmb_batch_history.clear()
        
        for b in batches:
            # Skip purged batches
            if b.state == "purged":
                continue
            ts_str = b.timestamp.strftime("%m/%d/%y %I:%M%p").lower()
            label = f"{ts_str} — {b.id_count} IDs, {b.file_count} files ({b.state})"
            self.cmb_batch_history.addItem(label, userData=b)
        
        self.cmb_batch_history.blockSignals(False)
        self._on_batch_selection_changed()

    def _on_batch_selection_changed(self):
        """Enable/disable undo/redo buttons based on selected batch state."""
        if self.cmb_batch_history.count() == 0:
            self.btn_undo_batch.setEnabled(False)
            self.btn_redo_batch.setEnabled(False)
            return
        
        batch: BatchInfo = self.cmb_batch_history.currentData()
        if batch is None:
            self.btn_undo_batch.setEnabled(False)
            self.btn_redo_batch.setEnabled(False)
            return
        
        self.btn_undo_batch.setEnabled(batch.state == "active")
        self.btn_redo_batch.setEnabled(batch.state == "undone")

    def _on_undo_batch(self):
        """Show undo confirmation dialog and process."""
        batch: BatchInfo = self.cmb_batch_history.currentData()
        if batch is None:
            return
        
        self._ilog.log("button_click", "btn_undo_batch", value=batch.batch_id)
        
        dialog = UndoBatchDialog(batch, parent=self)
        if dialog.exec() != QMessageBox.DialogCode.Accepted:
            return
        
        permanent = dialog.permanent
        target = self.cmb_target_batch.currentText()
        
        report = undo_batch(target, batch.batch_id, permanent=permanent)
        
        if report.errors:
            msg = (f"Undo completed with errors.\n"
                   f"Files removed: {report.files_removed}\n"
                   f"Files already missing: {report.files_missing}\n"
                   f"CSV rows removed: {report.csv_rows_removed}\n\n"
                   f"Errors:\n• " + "\n• ".join(report.errors[:5]))
            QMessageBox.warning(self, "starBoard", msg)
        else:
            action = "permanently deleted" if permanent else "undone"
            msg = (f"Batch {action}.\n"
                   f"Files removed: {report.files_removed}\n"
                   f"CSV rows removed: {report.csv_rows_removed}")
            if not permanent:
                msg += "\n\nYou can redo this batch if needed."
            QMessageBox.information(self, "starBoard", msg)
        
        self._log_batch(f"Batch {'purged' if permanent else 'undone'}: {report.files_removed} files removed.")
        self._refresh_batch_history()
        self._notify_first_order_refresh()

    def _on_redo_batch(self):
        """Show redo confirmation dialog and process."""
        batch: BatchInfo = self.cmb_batch_history.currentData()
        if batch is None:
            return
        
        self._ilog.log("button_click", "btn_redo_batch", value=batch.batch_id)
        
        target = self.cmb_target_batch.currentText()
        
        # Check source availability
        available, missing, missing_paths = check_redo_sources(target, batch.batch_id)
        
        dialog = RedoBatchDialog(batch, missing_count=missing, parent=self)
        if dialog.exec() != QMessageBox.DialogCode.Accepted:
            return
        
        report = redo_batch(target, batch.batch_id)
        
        if report.errors:
            msg = (f"Redo completed with errors.\n"
                   f"Files restored: {report.files_restored}\n"
                   f"Files failed: {report.files_failed}\n"
                   f"CSV rows restored: {report.csv_rows_restored}\n\n"
                   f"Errors:\n• " + "\n• ".join(report.errors[:5]))
            QMessageBox.warning(self, "starBoard", msg)
        else:
            msg = (f"Batch restored.\n"
                   f"Files restored: {report.files_restored}\n"
                   f"CSV rows restored: {report.csv_rows_restored}")
            QMessageBox.information(self, "starBoard", msg)
        
        self._log_batch(f"Batch redone: {report.files_restored} files restored.")
        self._refresh_batch_history()
        self._notify_first_order_refresh()

    # -------------------- Query Management --------------------
    def _build_query_management_group(self) -> QGroupBox:
        """Build the Query Management section for inspecting and deleting queries."""
        import platform
        import subprocess
        import os
        import shutil
        
        gb = QGroupBox("")  # title provided by CollapsibleSection
        lay = QVBoxLayout(gb)

        # Top row: Query combo + Open Folder button
        row = QHBoxLayout()
        row.addWidget(QLabel("Query:"))
        self.cmb_query_manage = QComboBox()
        self.cmb_query_manage.setEditable(True)
        self.cmb_query_manage.setInsertPolicy(QComboBox.NoInsert)
        self.cmb_query_manage.completer().setFilterMode(Qt.MatchContains)
        self.cmb_query_manage.completer().setCompletionMode(QCompleter.PopupCompletion)
        self.cmb_query_manage.currentIndexChanged.connect(self._on_query_manage_changed)
        row.addWidget(self.cmb_query_manage, 1)

        self.btn_open_query_folder = QPushButton("Open Folder")
        self.btn_open_query_folder.setToolTip("Open this query's folder in file explorer")
        self.btn_open_query_folder.clicked.connect(self._on_open_query_manage_folder)
        row.addWidget(self.btn_open_query_folder)
        
        self.btn_refresh_query_manage = QPushButton("↻")
        self.btn_refresh_query_manage.setFixedWidth(28)
        self.btn_refresh_query_manage.setToolTip("Refresh query list")
        self.btn_refresh_query_manage.clicked.connect(self._refresh_query_manage_combo)
        row.addWidget(self.btn_refresh_query_manage)
        
        lay.addLayout(row)

        # Image viewer for query inspection
        self.viewer_query_manage = ImageViewer()
        self.viewer_query_manage.setObjectName("query_manage_viewer")
        self.viewer_query_manage.setMinimumSize(320, 200)
        self.viewer_query_manage.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self.viewer_query_manage, 1)

        # Metadata display (read-only) in a grid
        self.grp_query_metadata = QGroupBox("Metadata")
        meta_lay = QHBoxLayout(self.grp_query_metadata)
        meta_lay.setContentsMargins(8, 8, 8, 8)
        
        self.lbl_query_meta_left = QLabel("")
        self.lbl_query_meta_left.setWordWrap(True)
        meta_lay.addWidget(self.lbl_query_meta_left, 1)
        
        self.lbl_query_meta_right = QLabel("")
        self.lbl_query_meta_right.setWordWrap(True)
        meta_lay.addWidget(self.lbl_query_meta_right, 1)
        
        lay.addWidget(self.grp_query_metadata)

        # Bottom row: Status + Delete button
        bottom_row = QHBoxLayout()
        self.lbl_query_status = QLabel("")
        self.lbl_query_status.setStyleSheet("color: gray; font-style: italic;")
        bottom_row.addWidget(self.lbl_query_status, 1)
        
        self.btn_delete_query = QPushButton("Delete Query")
        self.btn_delete_query.setStyleSheet("background-color: #c62828; color: white;")
        self.btn_delete_query.setToolTip("Delete this query (can be undone via Batch Upload section)")
        self.btn_delete_query.clicked.connect(self._on_delete_query)
        bottom_row.addWidget(self.btn_delete_query)
        
        lay.addLayout(bottom_row)

        # Initial population
        QTimer.singleShot(100, self._refresh_query_manage_combo)
        
        return gb

    def _refresh_query_manage_combo(self):
        """Refresh the query management combo box with imageless queries first."""
        self._ilog.log("button_click", "btn_refresh_query_manage", value="refresh")
        
        ids = list_ids("Queries")
        
        # Partition by image count - imageless queries first
        imageless = []
        with_images = []
        for qid in ids:
            img_count = len(list_image_files("Queries", qid))
            if img_count == 0:
                imageless.append(qid)
            else:
                with_images.append(qid)
        
        # Sort each group alphabetically, imageless first
        sorted_ids = sorted(imageless) + sorted(with_images)
        
        self.cmb_query_manage.blockSignals(True)
        prev_text = self.cmb_query_manage.currentText()
        self.cmb_query_manage.clear()
        self.cmb_query_manage.addItems(sorted_ids)
        
        # Try to restore previous selection
        if prev_text:
            idx = self.cmb_query_manage.findText(prev_text)
            if idx >= 0:
                self.cmb_query_manage.setCurrentIndex(idx)
        
        self.cmb_query_manage.blockSignals(False)
        
        # Update UI state
        has_queries = len(sorted_ids) > 0
        self.btn_delete_query.setEnabled(has_queries)
        self.btn_open_query_folder.setEnabled(has_queries)
        
        if has_queries:
            self._on_query_manage_changed()
        else:
            self.viewer_query_manage.clear()
            self.lbl_query_meta_left.setText("")
            self.lbl_query_meta_right.setText("")
            self.lbl_query_status.setText("No queries in archive")

    def _on_query_manage_changed(self):
        """Handle query selection change - load images and metadata."""
        qid = self.cmb_query_manage.currentText()
        self._ilog.log("combo_change", "cmb_query_manage", value=qid)
        
        if not qid:
            self.viewer_query_manage.clear()
            self.lbl_query_meta_left.setText("")
            self.lbl_query_meta_right.setText("")
            self.lbl_query_status.setText("")
            return
        
        # Load images
        files = list_image_files("Queries", qid)
        try:
            files = reorder_files_with_best("Queries", qid, files)
        except Exception:
            pass
        self.viewer_query_manage.set_images(files)
        
        # Count encounters (subdirectories with images)
        encounter_count = 0
        try:
            from src.data.archive_paths import roots_for_read
            for root in roots_for_read("Queries"):
                qdir = root / qid
                if qdir.exists():
                    for subdir in qdir.iterdir():
                        if subdir.is_dir() and any(subdir.glob("*")):
                            encounter_count += 1
        except Exception:
            pass
        
        # Load metadata
        id_col = ap.id_column_name("Queries")
        csv_paths = _csv_paths_for_read("Queries")
        rows = read_rows_multi(csv_paths)
        latest_map = last_row_per_id(rows, id_col)
        meta = latest_map.get(normalize_id_value(qid), {})
        
        # Display metadata in two columns
        keys = [k for k in meta.keys() if k != id_col and meta.get(k)]
        mid = (len(keys) + 1) // 2
        
        left_text = "\n".join(f"<b>{k}:</b> {meta[k]}" for k in keys[:mid])
        right_text = "\n".join(f"<b>{k}:</b> {meta[k]}" for k in keys[mid:])
        
        self.lbl_query_meta_left.setText(left_text or "<i>No metadata</i>")
        self.lbl_query_meta_right.setText(right_text)
        
        # Update status
        img_text = f"{len(files)} image{'s' if len(files) != 1 else ''}"
        enc_text = f"{encounter_count} encounter{'s' if encounter_count != 1 else ''}"
        self.lbl_query_status.setText(f"{img_text}, {enc_text}")

    def _on_open_query_manage_folder(self):
        """Open the selected query's folder in the file explorer."""
        import platform
        import subprocess
        import os
        
        qid = self.cmb_query_manage.currentText()
        self._ilog.log("button_click", "btn_open_query_folder", value=qid)
        
        if not qid:
            return
        
        folder = ap.queries_root(prefer_new=True) / qid
        if not folder.exists():
            warn(f"Query folder does not exist:\n{folder}", self)
            return
        
        try:
            if platform.system() == "Windows":
                os.startfile(str(folder))
            elif platform.system() == "Darwin":
                subprocess.call(["open", str(folder)])
            else:
                subprocess.call(["xdg-open", str(folder)])
        except Exception as e:
            warn(f"Could not open folder: {e}", self)

    def _on_delete_query(self):
        """Delete the selected query with confirmation and batch undo support."""
        import shutil
        
        qid = self.cmb_query_manage.currentText()
        self._ilog.log("button_click", "btn_delete_query", value=qid)
        
        if not qid:
            return
        
        # Get query info for confirmation
        files = list_image_files("Queries", qid)
        
        # Confirmation dialog
        reply = QMessageBox.warning(
            self,
            "Delete Query?",
            f"Are you sure you want to delete query '{qid}'?\n\n"
            f"This will remove:\n"
            f"  • {len(files)} image file{'s' if len(files) != 1 else ''}\n"
            f"  • The query folder and metadata\n\n"
            f"This can be undone via the Batch Upload section.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        try:
            # Generate batch ID for tracking
            batch_id = generate_batch_id()
            
            # Find query folder(s)
            from src.data.archive_paths import roots_for_read
            query_folders = []
            for root in roots_for_read("Queries"):
                qdir = root / qid
                if qdir.exists():
                    query_folders.append(qdir)
            
            if not query_folders:
                warn(f"Query folder not found for '{qid}'", self)
                return
            
            # Record file operations for undo
            file_ops: list[tuple[Path, Path]] = []
            for qdir in query_folders:
                for img_file in qdir.rglob("*"):
                    if img_file.is_file():
                        # src = original location (for redo), dest = where it was
                        file_ops.append((img_file, img_file))
            
            # Record the batch (as a deletion batch)
            record_batch_upload("Queries", batch_id, file_ops, {qid}, f"DELETE_{qid}")
            
            # Delete the query folder(s)
            for qdir in query_folders:
                shutil.rmtree(qdir)
            
            # Remove CSV rows for this query
            from src.data.batch_undo import _remove_csv_rows
            _remove_csv_rows("Queries", {qid})
            
            # Success message
            info(f"Query '{qid}' deleted.\n\nYou can undo this via the Batch Upload section.", self)
            
            # Refresh UI
            self._refresh_query_manage_combo()
            self._refresh_batch_history()
            self._notify_first_order_refresh()
            
        except Exception as e:
            logger.error("Failed to delete query %s: %s", qid, e)
            warn(f"Failed to delete query: {e}", self)

    # -------------------- Metadata Editing Mode --------------------
    def _build_editing_group(self) -> QGroupBox:
        gb = QGroupBox("")  # title provided by CollapsibleSection
        gb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay = QVBoxLayout(gb)

        # --- top row: target + ID ---
        row = QHBoxLayout()
        row.addWidget(QLabel("Archive:"))
        self.cmb_target_edit = QComboBox()
        self.cmb_target_edit.addItems(["Gallery", "Queries"])
        self.cmb_target_edit.currentIndexChanged.connect(self._refresh_id_list_edit)
        self.cmb_target_edit.currentIndexChanged.connect(
            lambda: self._ilog.log("combo_change", "cmb_target_edit", value=self.cmb_target_edit.currentText()))
        row.addWidget(self.cmb_target_edit)

        row.addWidget(QLabel("ID:"))
        self.cmb_id_edit = QComboBox()
        # Make combo editable for type-to-search functionality
        self.cmb_id_edit.setEditable(True)
        self.cmb_id_edit.setInsertPolicy(QComboBox.NoInsert)
        self.cmb_id_edit.completer().setFilterMode(Qt.MatchContains)
        self.cmb_id_edit.completer().setCompletionMode(QCompleter.PopupCompletion)
        self.cmb_id_edit.currentIndexChanged.connect(self._on_edit_id_changed)
        row.addWidget(self.cmb_id_edit, 1)

        # ID navigation buttons
        self.btn_prev_id_edit = QPushButton("◀")
        self.btn_prev_id_edit.setFixedWidth(28)
        self.btn_prev_id_edit.setToolTip("Previous ID in list")
        self.btn_prev_id_edit.clicked.connect(self._on_prev_id_edit_clicked)
        row.addWidget(self.btn_prev_id_edit)

        self.btn_next_id_edit = QPushButton("▶")
        self.btn_next_id_edit.setFixedWidth(28)
        self.btn_next_id_edit.setToolTip("Next ID in list")
        self.btn_next_id_edit.clicked.connect(self._on_next_id_edit_clicked)
        row.addWidget(self.btn_next_id_edit)

        lay.addLayout(row)

        # --- encounter date row ---
        enc_row = QHBoxLayout()
        enc_row.addWidget(QLabel("Encounter:"))
        self.cmb_encounter_edit = QComboBox()
        self.cmb_encounter_edit.setMinimumWidth(140)
        self.cmb_encounter_edit.currentIndexChanged.connect(self._on_encounter_edit_changed)
        enc_row.addWidget(self.cmb_encounter_edit)

        enc_row.addWidget(QLabel("Date:"))
        self.date_encounter_edit = QDateEdit()
        self.date_encounter_edit.setCalendarPopup(True)
        self.date_encounter_edit.setDisplayFormat("yyyy-MM-dd")
        enc_row.addWidget(self.date_encounter_edit)

        self.btn_save_encounter_date = QPushButton("Save Date")
        self.btn_save_encounter_date.setToolTip("Save encounter date override")
        self.btn_save_encounter_date.clicked.connect(self._on_save_encounter_date)
        enc_row.addWidget(self.btn_save_encounter_date)
        enc_row.addStretch(1)
        lay.addLayout(enc_row)

        # --- split: form (left) + image viewer (right) ---
        self.meta_form_edit = MetadataForm()
        self.meta_form_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.viewer_edit = ImageViewer()
        self.viewer_edit.setObjectName("edit_image_viewer")
        self.viewer_edit.setMinimumSize(320, 240)
        self.viewer_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.viewer_edit.bestRequested.connect(self._on_set_best_clicked)

        split = QHBoxLayout()
        split.setContentsMargins(0, 0, 0, 0)
        split.setSpacing(12)
        split.addWidget(self.meta_form_edit, 2)
        split.addWidget(self.viewer_edit, 3)
        lay.addLayout(split, 1)  # stretch factor 1 to expand vertically

        # --- bottom row: Save buttons ---
        row2 = QHBoxLayout()
        row2.addStretch(1)

        # Save button (saves without carry-over)
        self.btn_save_only = QPushButton("Save")
        self.btn_save_only.setToolTip("Save current metadata without carrying values to next ID")
        self.btn_save_only.clicked.connect(self._on_save_only)
        row2.addWidget(self.btn_save_only)

        # Save & Carry Over button
        self.btn_save_edit = QPushButton("Save & Carry Over")
        self.btn_save_edit.setToolTip("Save and carry values to the next ID")
        self.btn_save_edit.clicked.connect(self._on_save_edits)
        self.btn_save_edit.setDefault(True)
        row2.addWidget(self.btn_save_edit)
        lay.addLayout(row2)

        self._refresh_id_list_edit()
        return gb

    def _refresh_id_list_edit(self):
        target = self.cmb_target_edit.currentText()
        self.meta_form_edit.set_target(target)
        ids = list_ids(target)
        if target == "Queries":
            last_obs = last_observation_for_all("Queries")
            ids = sorted(
                ids,
                key=lambda qid: (
                    (last_obs.get(qid) is None),
                    (last_obs.get(qid) or _date.max),
                    qid.lower(),
                )
            )

        self.cmb_id_edit.blockSignals(True)
        self.cmb_id_edit.clear()
        self.cmb_id_edit.addItems(ids)
        self.cmb_id_edit.blockSignals(False)
        self.btn_save_only.setEnabled(bool(ids))
        self.btn_save_edit.setEnabled(bool(ids))
        # reset last_edit_id baseline to current (or "")
        self._last_edit_id = self.cmb_id_edit.currentText() if ids else ""
        self._on_edit_id_changed()

    def _apply_carry_over_to_form(self, form: MetadataForm, carry: Dict[str, str]) -> None:
        """For any empty field in the current form, apply a non-empty value from carry-over."""
        try:
            id_col = form._id_col()
            for col, val in carry.items():
                if col == id_col:
                    continue
                if not (val or "").strip():
                    continue
                # Get the widget for this field (v2 uses _widgets dict with AnnotationWidget)
                w = form.get_widget(col) if hasattr(form, "get_widget") else form.widgets.get(col)
                if not w:
                    continue
                # Read current value using the v2 API
                if hasattr(w, "get_value"):
                    # v2 AnnotationWidget
                    current = w.get_value()
                    if not (current or "").strip():
                        w.set_value(val)
                else:
                    # Legacy v1 fallback
                    current = ""
                    if hasattr(w, "text"):
                        current = w.text()
                    elif hasattr(w, "toPlainText"):
                        try:
                            current = w.toPlainText()
                        except Exception:
                            current = ""
                    if not (current or "").strip():
                        if hasattr(w, "setText"):
                            w.setText(val)
                        elif hasattr(w, "setPlainText"):
                            w.setPlainText(val)
        except Exception as e:
            logger.debug("carry_over apply skipped due to: %s", e)

    def _on_edit_id_changed(self):
        target = self.cmb_target_edit.currentText()
        next_id = self.cmb_id_edit.currentText()
        self._ilog.log("combo_change", "cmb_id_edit", value=next_id,
                      context={"target": target})

        # nothing selected
        if not next_id:
            self.meta_form_edit.populate({})
            if hasattr(self, 'viewer_edit'):
                self.viewer_edit.clear()
            self._last_edit_id = ""
            return

        # If user has unsaved edits, offer Save & Carry Over / Discard / Cancel
        if self._edit_loaded_once and self.meta_form_edit.is_dirty():
            box = QMessageBox(self)
            box.setWindowTitle("Save & carry over?")
            box.setText("You have unsaved edits. Save and carry them over to the next ID?")
            btn_save = box.addButton("Save & Carry Over", QMessageBox.AcceptRole)
            btn_discard = box.addButton("Discard", QMessageBox.DestructiveRole)
            btn_cancel = box.addButton("Cancel", QMessageBox.RejectRole)
            box.setDefaultButton(btn_save)
            box.exec()

            clicked = box.clickedButton()
            if clicked is btn_cancel:
                # revert selection to previous one
                self.cmb_id_edit.blockSignals(True)
                if self._last_edit_id:
                    idx = self.cmb_id_edit.findText(self._last_edit_id)
                    if idx >= 0:
                        self.cmb_id_edit.setCurrentIndex(idx)
                self.cmb_id_edit.blockSignals(False)
                return
            elif clicked is btn_save:
                self._on_save_edits()
            else:
                # Discard: keep working buffer but don't change it
                pass

        # Load values
        self.meta_form_edit.set_id_value(next_id)
        self._populate_metadata_from_csv_edit(target, next_id)

        # Apply carry-over defaults (only to empty fields)
        if self._carry_over:
            self._apply_carry_over_to_form(self.meta_form_edit, self._carry_over)

        # Update image viewer for this ID — with BEST-first reordering
        from src.data.best_photo import reorder_files_with_best  # local import to avoid touching module imports
        files = self._gather_images_for_id(target, next_id)
        try:
            files = reorder_files_with_best(target, next_id, files)
        except Exception:
            # any I/O problem should not break the UI
            pass
        self.viewer_edit.set_images(files)

        self._last_edit_id = next_id
        if not self._edit_loaded_once:
            self._edit_loaded_once = True

        # Refresh encounter list for this ID
        self._refresh_encounter_list_edit()

    def _refresh_encounter_list_edit(self) -> None:
        """Populate encounter combo for the current ID."""
        target = self.cmb_target_edit.currentText()
        id_val = self.cmb_id_edit.currentText()
        self.cmb_encounter_edit.blockSignals(True)
        self.cmb_encounter_edit.clear()
        if id_val:
            encounters = list_encounters_for_id(target, id_val)
            self.cmb_encounter_edit.addItems(encounters)
        self.cmb_encounter_edit.blockSignals(False)
        self._on_encounter_edit_changed()

    def _on_encounter_edit_changed(self) -> None:
        """Update date picker when encounter selection changes."""
        target = self.cmb_target_edit.currentText()
        id_val = self.cmb_id_edit.currentText()
        enc_name = self.cmb_encounter_edit.currentText()
        if not enc_name:
            self.date_encounter_edit.setDate(QDate.currentDate())
            self.btn_save_encounter_date.setEnabled(False)
            return
        self.btn_save_encounter_date.setEnabled(True)
        d = get_encounter_date(target, id_val, enc_name)
        if d:
            self.date_encounter_edit.setDate(QDate(d.year, d.month, d.day))
        else:
            self.date_encounter_edit.setDate(QDate.currentDate())

    def _on_save_encounter_date(self) -> None:
        """Save the encounter date override."""
        target = self.cmb_target_edit.currentText()
        id_val = self.cmb_id_edit.currentText()
        enc_name = self.cmb_encounter_edit.currentText()
        if not id_val or not enc_name:
            return
        qd = self.date_encounter_edit.date()
        d = _date(qd.year(), qd.month(), qd.day())
        set_encounter_date(target, id_val, enc_name, d)
        info(f"Saved encounter date: {d.isoformat()}", self)

    def _on_prev_id_edit_clicked(self) -> None:
        """Navigate to the previous ID in the combo box list."""
        self._ilog.log("button_click", "btn_prev_id_edit", value="clicked")
        current_idx = self.cmb_id_edit.currentIndex()
        if current_idx > 0:
            self.cmb_id_edit.setCurrentIndex(current_idx - 1)

    def _on_next_id_edit_clicked(self) -> None:
        """Navigate to the next ID in the combo box list."""
        self._ilog.log("button_click", "btn_next_id_edit", value="clicked")
        current_idx = self.cmb_id_edit.currentIndex()
        max_idx = self.cmb_id_edit.count() - 1
        if current_idx < max_idx:
            self.cmb_id_edit.setCurrentIndex(current_idx + 1)

    def _on_set_best_edit(self):
        target = self.cmb_target_edit.currentText()
        id_val = self.cmb_id_edit.currentText()
        self._ilog.log("button_click", "btn_set_best_edit", value=id_val,
                      context={"target": target})
        if not id_val:
            warn("No ID selected.", self)
            return
        cur = self.viewer_edit.current_path()
        if not cur:
            warn("There is no image to pin for this ID.", self)
            return
        try:
            # Persist sidecar
            save_best_for_id(target, id_val, cur)
            # Reload ordered images (pinned first) and show first
            files = self._gather_images_for_id(target, id_val)
            files = reorder_files_with_best(target, id_val, files)
            self.viewer_edit.set_images(files)
            self.viewer_edit.set_images(files)
            self.viewer_edit.set_index(0)
            info("Saved 'best' photo for this ID.", self)
        except Exception as e:
            warn(f"Couldn't save 'best' photo: {e}", self)

    def _on_set_best_clicked(self) -> None:
        """Mark the currently shown image as the 'best' for this ID and refresh the viewer."""
        target = self.cmb_target_edit.currentText() or ""
        id_val = self.cmb_id_edit.currentText() or ""
        self._ilog.log("button_click", "btn_set_best_clicked", value=id_val,
                      context={"target": target})
        if not id_val:
            return

        # Get the current image from the Metadata tab's viewer
        cur = None
        try:
            cur = self.viewer_edit.current_path()
        except Exception:
            # Fallback for older ImageViewer without accessors
            try:
                if getattr(self.viewer_edit, "_images", None):
                    idx = getattr(self.viewer_edit, "_idx", -1)
                    if 0 <= idx < len(self.viewer_edit._images):
                        cur = self.viewer_edit._images[idx]
            except Exception:
                cur = None

        if not cur:
            return

        # Persist and then reorder currently loaded list to put best first
        try:
            from src.data.best_photo import save_best_for_id, reorder_files_with_best
            save_best_for_id(target, id_val, cur)
            files = list(getattr(self.viewer_edit, "_images", []))
            files = reorder_files_with_best(target, id_val, files)
            self.viewer_edit.set_images(files)
        except Exception:
            # No user-facing error; this feature is best-effort and should never break the app
            pass

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
                for p in sorted(base.rglob("*")):
                    if p.is_file() and p.suffix.lower() in exts:
                        rp = p.resolve()
                        if rp not in seen:
                            out.append(rp)
                            seen.add(rp)
        return out

    def _on_save_only(self):
        """Save current metadata without updating carry-over buffer."""
        target = self.cmb_target_edit.currentText()
        id_val = self.cmb_id_edit.currentText()
        self._ilog.log("button_click", "btn_save_only", value=id_val,
                      context={"target": target})
        if not id_val:
            warn("No ID selected.", self)
            return
        csv_path, header = ap.metadata_csv_for(target)
        row = self.meta_form_edit.collect_row()
        row[ap.id_column_name(target)] = id_val
        append_row(csv_path, header, row)

        # Mark form as clean so is_dirty() returns False
        self.meta_form_edit.mark_clean()

        # Do NOT update carry-over buffer

        logger.info("Saved edits (no carry-over) for target=%s id=%s", target, id_val)
        info("Edits saved.", self)

        # First‑order should reflect edits immediately
        self._notify_first_order_refresh()

    def _on_save_edits(self):
        """Save current metadata and update carry-over buffer."""
        target = self.cmb_target_edit.currentText()
        id_val = self.cmb_id_edit.currentText()
        self._ilog.log("button_click", "btn_save_edits", value=id_val,
                      context={"target": target})
        if not id_val:
            warn("No ID selected.", self)
            return
        csv_path, header = ap.metadata_csv_for(target)
        row = self.meta_form_edit.collect_row()
        row[ap.id_column_name(target)] = id_val
        append_row(csv_path, header, row)

        # Mark form as clean so is_dirty() returns False
        self.meta_form_edit.mark_clean()

        # Update carry-over buffer: keep only non-empty, non-ID fields
        id_col = ap.id_column_name(target)
        self._carry_over = {k: v for k, v in row.items() if k != id_col and (v or "").strip()}

        logger.info("Saved edits for target=%s id=%s", target, id_val)
        info("Edits saved. Carry-over is active for the next ID.", self)

        # First‑order should reflect edits immediately
        self._notify_first_order_refresh()

    # -------------------- cross-tab refresh --------------------
    def _notify_first_order_refresh(self) -> None:
        """
        Best-effort: find a TabFirstOrder in the same QTabWidget and tell it to rebuild/refresh.
        Works without importing TabFirstOrder directly.
        """
        try:
            tabs = None
            for w in self.window().findChildren(QTabWidget):
                tabs = w
                break
            if not tabs:
                return
            for i in range(tabs.count()):
                wid = tabs.widget(i)
                if wid.__class__.__name__ == "TabFirstOrder":
                    # gentle rebuild + refresh
                    if hasattr(wid, "engine") and hasattr(wid.engine, "rebuild"):
                        try:
                            wid.engine.rebuild()
                        except Exception:
                            pass
                    for name in ("_refresh_query_ids", "_refresh_results"):
                        if hasattr(wid, name):
                            try:
                                getattr(wid, name)()
                            except Exception:
                                pass
                    break
        except Exception as e:
            logger.debug("notify_first_order_refresh skipped: %s", e)