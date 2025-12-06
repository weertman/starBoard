# src/ui/tab_setup.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List, Dict
import logging
from datetime import date as _date
from PySide6.QtCore import Qt, QDate, QPoint, QEvent, QPointF, QSize, Signal, QObject, QThreadPool, QRunnable, QTimer
from PySide6.QtGui import QPixmap, QCursor, QImage, QImageReader
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QComboBox, QLabel,
    QFileDialog, QListWidget, QListWidgetItem, QLineEdit, QDateEdit,
    QMessageBox, QCheckBox, QPlainTextEdit, QScrollArea, QSizePolicy,
    QScrollBar, QTabWidget, QCompleter,
)

from src.ui.collapsible import CollapsibleSection  # existing utility for collapsible panels
from src.data import archive_paths as ap
from src.data.csv_io import (
    append_row, read_rows_multi, last_row_per_id, normalize_id_value
)
from src.data.id_registry import list_ids, id_exists
from src.data.ingest import ensure_encounter_name, place_images, discover_ids_and_images
from src.data.validators import validate_id
from src.data.archive_paths import last_observation_for_all
from src.data.best_photo import reorder_files_with_best, save_best_for_id
from .metadata_form_v2 import MetadataForm

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


# ---------- interactive image viewer (now supports mouse pan + wheel zoom) ----------


# ---------- interactive image viewer (async loader, mouse pan + wheel zoom) ----------

class ImageViewer(QWidget):
    """
    Lightweight image viewer dedicated to the *Metadata Editing Mode*.

    Improvements over the previous version:
      - Non-blocking image decoding with a private QThreadPool (max 2 threads).
      - On-demand prefetch of previous/next images.
      - Small LRU cache of decoded QImages (CPU-only), conversion to QPixmap happens on the GUI thread.
      - Mouse wheel zoom anchored at the cursor; click-and-drag to pan.
      - 'Fit to window' toggle and zoom buttons.

    NOTE: We intentionally keep the implementation CPU-only and cap the pool to two threads,
    per requirements. No OpenGL or GPU-specific paths are used.
    """
    class _TaskSignals(QObject):
        finished = Signal(str, object)  # path, QImage (or None on failure)

    class _LoadTask(QRunnable):
        def __init__(self, path:str, signals:'ImageViewer._TaskSignals'):
            super().__init__()
            self.path = path
            self.signals = signals

        def run(self):
            img = None
            try:
                reader = QImageReader(self.path)
                reader.setAutoTransform(True)  # honor EXIF orientation
                # Decode the image. If it fails, img will remain None.
                qimg = reader.read()
                if not qimg.isNull():
                    img = qimg
            except Exception:
                img = None
            # deliver back to the GUI thread
            self.signals.finished.emit(self.path, img)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ImageViewer")
        self._images: List[Path] = []
        self._idx: int = -1
        self._fit: bool = True
        self._scale: float = 1.0

        # async loader state
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(2)  # up to two threads max
        from collections import OrderedDict
        self._cache = OrderedDict()  # key: str(path) -> QImage
        self._cache_cap = 12
        self._inflight: Dict[str, ImageViewer._TaskSignals] = {}

        # pan/zoom state
        self._dragging = False
        self._last_mouse_pos = QPoint()
        self._last_pixmap_size = None  # QSize of last rendered pixmap, for scroll anchoring

        # toolbar
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        bar = QHBoxLayout()
        self.btn_prev = QPushButton("◀ Prev")
        self.btn_next = QPushButton("Next ▶")
        self.btn_zoom_out = QPushButton("– Zoom")
        self.btn_zoom_in = QPushButton("+ Zoom")
        self.btn_fit = QCheckBox("Fit to window")
        self.btn_fit.setChecked(True)
        self.lbl_counter = QLabel("0/0")
        bar.addWidget(self.btn_prev)
        bar.addWidget(self.btn_next)
        bar.addSpacing(12)
        bar.addWidget(self.btn_zoom_out)
        bar.addWidget(self.btn_zoom_in)
        bar.addSpacing(12)
        bar.addWidget(self.btn_fit)
        bar.addStretch(1)
        bar.addWidget(self.lbl_counter)
        outer.addLayout(bar)

        # Scroller + label
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.label = QLabel("No image", alignment=Qt.AlignCenter)
        self.label.setWordWrap(True)
        self.scroll.setMinimumSize(320, 240)
        self.scroll.setWidget(self.label)
        outer.addWidget(self.scroll, 1)

        # Install event filter on the viewport for mouse pan/zoom
        self.scroll.viewport().installEventFilter(self)

        # Connections
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next.clicked.connect(self.next_image)
        self.btn_zoom_in.clicked.connect(lambda: self._set_scale(self._scale * 1.25))
        self.btn_zoom_out.clicked.connect(lambda: self._set_scale(self._scale / 1.25))
        self.btn_fit.toggled.connect(self._on_fit_toggled)

        self._update_ui()

    # ----- public API -----
    def clear(self):
        self._images = []
        self._idx = -1
        self._scale = 1.0
        self.btn_fit.setChecked(True)
        self._fit = True
        self.label.setText('No image')
        self._update_ui()

    def set_images(self, paths: List[Path]):
        self._images = [Path(p) for p in paths if Path(p).exists()]
        self._idx = 0 if self._images else -1
        self._scale = 1.0
        self.btn_fit.setChecked(True)
        self._fit = True
        self._render_current_async(prefetch=True)
        self._update_ui()

    def set_index(self, idx: int):
        if 0 <= idx < len(self._images):
            self._idx = idx
            self._render_current_async(prefetch=True)
            self._update_ui()

    def current_index(self) -> int:
        """Return the current zero-based index, or -1 when empty."""
        return self._idx

    def current_path(self):
        """Return the current image Path (or None if empty)."""
        if 0 <= self._idx < len(self._images):
            return self._images[self._idx]
        return None

    # ----- cache / loader helpers -----
    def _touch_cache(self, key: str, img: QImage):
        from collections import OrderedDict
        # move-to-end behavior
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            self._cache[key] = img
            while len(self._cache) > self._cache_cap:
                self._cache.popitem(last=False)

    def _get_cached(self, key: str):
        if key in self._cache:
            img = self._cache[key]
            # touch
            self._cache.move_to_end(key)
            return img
        return None

    def _request_image(self, path: Path):
        key = str(path)
        cached = self._get_cached(key)
        if cached is not None:
            return cached
        # Already loading?
        if key in self._inflight:
            return None

        sig = ImageViewer._TaskSignals()
        sig.finished.connect(self._on_loader_finished)
        task = ImageViewer._LoadTask(str(path), sig)
        self._inflight[key] = sig
        self._pool.start(task)
        return None

    def _on_loader_finished(self, path: str, img: object):
        # task finished in background; register & refresh if relevant
        self._inflight.pop(path, None)
        if isinstance(img, QImage) and not img.isNull():
            self._touch_cache(path, img)
        # If the finished image is the one we need (or neighbor), trigger re-render
        if self._idx >= 0 and self._idx < len(self._images):
            cur = str(self._images[self._idx])
            if path == cur:
                self._render_current_async(prefetch=True)
        # No else: neighbor images will be used when navigated to.

    # ----- UI helpers -----
    def _update_ui(self):
        n = len(self._images)
        self.lbl_counter.setText(f"{self._idx+1 if self._idx>=0 else 0}/{n}")
        has = (self._idx >= 0) and (self._idx < n)
        self.btn_prev.setEnabled(self._idx > 0)
        self.btn_next.setEnabled(self._idx + 1 < n)
        self.btn_zoom_in.setEnabled(has and not self._fit)
        self.btn_zoom_out.setEnabled(has and not self._fit)

    def _on_fit_toggled(self, checked: bool):
        self._fit = checked
        if checked:
            # Reset pan state when switching to fit
            self._dragging = False
            self.scroll.viewport().unsetCursor()
            # reset scale when switching to fit
            self._scale = 1.0
        self._render_current_async(prefetch=True)
        self._update_ui()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._fit:
            self._render_current_async(prefetch=False)

    # ----- navigation -----
    def prev_image(self):
        if self._idx > 0:
            self._idx -= 1
            self._render_current_async(prefetch=True)
            self._update_ui()

    def next_image(self):
        if self._idx + 1 < len(self._images):
            self._idx += 1
            self._render_current_async(prefetch=True)
            self._update_ui()

    def current_path(self) -> Optional[Path]:
        """Return absolute Path of the image currently shown, or None if empty."""
        if 0 <= self._idx < len(self._images):
            return self._images[self._idx]
        return None
    # ----- events -----
    def eventFilter(self, obj, ev):
        if obj is self.scroll.viewport():
            et = ev.type()
            if et == QEvent.Wheel and not self._fit and (0 <= self._idx < len(self._images)):
                # Zoom around cursor
                delta = ev.angleDelta().y()
                if delta == 0:
                    return True
                old_scale = self._scale
                factor = 1.25 if delta > 0 else 1.0 / 1.25
                new_scale = max(0.05, min(10.0, old_scale * factor))

                # Cursor position relative to viewport
                vp_pos = ev.position().toPoint()
                hbar: QScrollBar = self.scroll.horizontalScrollBar()
                vbar: QScrollBar = self.scroll.verticalScrollBar()

                # Content coords under cursor before zoom
                content_x = hbar.value() + vp_pos.x()
                content_y = vbar.value() + vp_pos.y()
                content_w = max(1, self.label.width())
                content_h = max(1, self.label.height())
                rel_x = content_x / content_w
                rel_y = content_y / content_h

                # Apply zoom
                self._scale = new_scale
                self._render_current_async(prefetch=False)

                # Keep the same content point under cursor
                new_w = max(1, self.label.width())
                new_h = max(1, self.label.height())
                hbar.setValue(int(rel_x * new_w - vp_pos.x()))
                vbar.setValue(int(rel_y * new_h - vp_pos.y()))
                return True

            if et == QEvent.MouseButtonPress and (0 <= self._idx < len(self._images)):
                if ev.button() == Qt.LeftButton:
                    self._dragging = True
                    self._last_mouse_pos = ev.pos()
                    self.scroll.viewport().setCursor(QCursor(Qt.ClosedHandCursor))
                    return True

            if et == QEvent.MouseMove and self._dragging:
                pos = ev.pos()
                dx = pos.x() - self._last_mouse_pos.x()
                dy = pos.y() - self._last_mouse_pos.y()
                self._last_mouse_pos = pos
                self.scroll.horizontalScrollBar().setValue(self.scroll.horizontalScrollBar().value() - dx)
                self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().value() - dy)
                return True

            if et == QEvent.MouseButtonRelease and self._dragging:
                self._dragging = False
                self.scroll.viewport().unsetCursor()
                return True
        return super().eventFilter(obj, ev)

    # ----- rendering -----
    def _set_scale(self, value: float):
        self._scale = max(0.05, min(10.0, value))
        self._render_current_async(prefetch=False)
        self._update_ui()

    def _render_current_async(self, prefetch: bool):
        if not (0 <= self._idx < len(self._images)):
            self.label.setText("No image")
            return

        cur_path = self._images[self._idx]
        # request/obtain the original image (decoded)
        img = self._request_image(cur_path)

        if img is None:
            # show a quick placeholder while loading
            self.label.setText("Loading…")
        else:
            # compute target size
            view_size = self.scroll.viewport().size()
            iw, ih = img.width(), img.height()
            if iw <= 0 or ih <= 0:
                self.label.setText("Unable to load image.")
                return

            if self._fit:
                if view_size.width() > 0 and view_size.height() > 0:
                    ratio = min(view_size.width() / iw, view_size.height() / ih)
                else:
                    ratio = 1.0
            else:
                ratio = self._scale

            w = max(1, int(iw * ratio))
            h = max(1, int(ih * ratio))

            # scale in GUI thread (CPU), no GPU-specific path
            scaled_img = img.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            pm = QPixmap.fromImage(scaled_img)
            self.label.setPixmap(pm)
            self.label.resize(pm.size())

        # prefetch neighbors
        if prefetch:
            for j in (self._idx - 1, self._idx + 1):
                if 0 <= j < len(self._images):
                    self._request_image(self._images[j])


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
        gb_edit   = self._build_editing_group()         # title-less

        # Wrap each group with an expandable panel (collapsed by default)
        sec_single = CollapsibleSection("Single Upload", start_collapsed=True)
        sec_single.setContent(gb_single)
        sec_batch = CollapsibleSection("Batch Upload IDs", start_collapsed=True)
        sec_batch.setContent(gb_batch)
        sec_edit = CollapsibleSection("Metadata Editing Mode", start_collapsed=True)
        sec_edit.setContent(gb_edit)

        # Single/Batch sections stay compact; Metadata Editing expands to fill space
        for sec in (sec_single, sec_batch):
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
        # First‑order refresh: new IDs/rows may be available
        self._notify_first_order_refresh()

    def _log_batch(self, msg: str):
        logger.info(msg)
        self.log_batch.appendPlainText(msg)

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
        lay.addLayout(row)

        # --- split: form (left) + image viewer (right) ---
        self.meta_form_edit = MetadataForm()
        self.meta_form_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.viewer_edit = ImageViewer()
        self.viewer_edit.setObjectName("edit_image_viewer")
        self.viewer_edit.setMinimumSize(320, 240)
        self.viewer_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        split = QHBoxLayout()
        split.setContentsMargins(0, 0, 0, 0)
        split.setSpacing(12)
        split.addWidget(self.meta_form_edit, 2)
        split.addWidget(self.viewer_edit, 3)
        lay.addLayout(split, 1)  # stretch factor 1 to expand vertically

        # --- bottom row: Set Best + Save ---
        row2 = QHBoxLayout()

        # NEW: 'Set Best' pin action for the current image/ID
        self.btn_set_best_edit = QPushButton("Set Best")
        self.btn_set_best_edit.setToolTip("Mark the currently shown image as the 'best' for this ID")
        self.btn_set_best_edit.clicked.connect(self._on_set_best_clicked)
        self.btn_set_best_edit.setEnabled(False)  # enabled when an image is present
        row2.addWidget(self.btn_set_best_edit)

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

        # nothing selected
        if not next_id:
            self.meta_form_edit.populate({})
            if hasattr(self, 'viewer_edit'):
                self.viewer_edit.clear()
            self._last_edit_id = ""
            if hasattr(self, 'btn_set_best_edit'):
                self.btn_set_best_edit.setEnabled(False)
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

        # Enable/disable the Set Best button based on availability
        if hasattr(self, 'btn_set_best_edit'):
            self.btn_set_best_edit.setEnabled(bool(files))

        self._last_edit_id = next_id
        if not self._edit_loaded_once:
            self._edit_loaded_once = True

    def _on_set_best_edit(self):
        target = self.cmb_target_edit.currentText()
        id_val = self.cmb_id_edit.currentText()
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
        if not id_val:
            warn("No ID selected.", self)
            return
        csv_path, header = ap.metadata_csv_for(target)
        row = self.meta_form_edit.collect_row()
        row[ap.id_column_name(target)] = id_val
        append_row(csv_path, header, row)

        # Do NOT update carry-over buffer

        logger.info("Saved edits (no carry-over) for target=%s id=%s", target, id_val)
        info("Edits saved.", self)

        # First‑order should reflect edits immediately
        self._notify_first_order_refresh()

    def _on_save_edits(self):
        """Save current metadata and update carry-over buffer."""
        target = self.cmb_target_edit.currentText()
        id_val = self.cmb_id_edit.currentText()
        if not id_val:
            warn("No ID selected.", self)
            return
        csv_path, header = ap.metadata_csv_for(target)
        row = self.meta_form_edit.collect_row()
        row[ap.id_column_name(target)] = id_val
        append_row(csv_path, header, row)

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