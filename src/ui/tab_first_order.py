# src/ui/tab_first_order.py
from __future__ import annotations

import json
import os
import platform
import subprocess
from typing import Dict, List, Set, Tuple, Optional
from datetime import date as _date

from PySide6.QtCore import Qt, QEvent, QTimer, Signal, QDate, QFileSystemWatcher
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QGroupBox, QCheckBox, QScrollArea, QSizePolicy, QSpinBox, QDoubleSpinBox, QGridLayout,
    QSplitter, QDialog, QTableWidget, QTableWidgetItem, QHeaderView, QWidget as _QW, QVBoxLayout as _QVL,
    QListWidget, QListWidgetItem, QLineEdit, QDialogButtonBox, QToolButton, QFrame, QDateEdit,
    QCompleter, QSlider,
)

from src.search.engine import (
    FirstOrderSearchEngine, ALL_FIELDS,
    TEXT_FIELDS, NUMERIC_FIELDS, ORDINAL_FIELDS, COLOR_FIELDS, SET_FIELDS
)
from src.data.fields_config import get_fields_config
from src.data import archive_paths as ap
from src.data.id_registry import list_ids
from src.data.image_index import list_image_files
from src.data.compare_labels import load_latest_map_for_query
from src.data.matches_matrix import load_match_matrix
from src.data.observation_dates import last_observation_for_all
from src.data.merge_yes import is_query_silent
from .image_strip import ImageStrip
from .lineup_card import LineupCard
from .fields_config_dialog import FieldsConfigDialog
from .query_state_delegate import (
    QueryStateDelegate, QueryState, QUERY_STATE_ROLE,
    get_query_state, apply_query_states_to_combobox, apply_quality_to_combobox,
    get_quality_for_ids, QUALITY_MADREPORITE_ROLE, QUALITY_ANUS_ROLE, QUALITY_POSTURE_ROLE,
)
from src.data.best_photo import reorder_files_with_best, save_best_for_id
from src.data.archive_paths import last_observation_for_all
from src.data.csv_io import read_rows_multi, last_row_per_id, append_row
from src.data.metadata_history import (
    record_bulk_update, get_current_metadata_for_gallery,
    SOURCE_UI,
)
from src.ui.image_quality_panel import ImageQualityPanel
from src.ui.metadata_form_v2 import MetadataFormV2
from src.utils.interaction_logger import get_interaction_logger
from src.dl.verification_lookup import get_verification_lookup, get_active_verification_lookup
from src.dl.registry import DLRegistry

# ---- Field groupings for the checkbox panel (V2 schema)
FIELD_GROUPS = [
    ("Numeric", NUMERIC_FIELDS),
    ("Ordinal", ORDINAL_FIELDS),
    ("Colors", COLOR_FIELDS),
    ("Codes", SET_FIELDS),
    ("Text", TEXT_FIELDS),
]

# Special preset name that reads from fields_config.yaml
PRESET_CONFIG = "Config"
# Special preset name that uses all fields (hardcoded default)
PRESET_DEFAULT = "Default (All)"

PRESETS: Dict[str, Set[str]] = {
    PRESET_CONFIG: set(),  # Placeholder - handled dynamically in _apply_preset
    PRESET_DEFAULT: set(ALL_FIELDS),
    # Arms + short arm codes + ordinal patterns
    "Arms & Patterns": set([
        "num_apparent_arms", "num_total_arms", "short_arm_code",
        "stripe_order", "stripe_prominence", "reticulation_order",
        "rosette_prominence", "arm_thickness",
    ]),
    # All color fields
    "Colors only": set(COLOR_FIELDS),
    # Size measurement
    "Size only": set(["tip_to_tip_size_cm"]),
    # Text descriptions
    "Text only": set(TEXT_FIELDS),
    # Stripe-related fields
    "Stripes": set([
        "stripe_color", "stripe_order", "stripe_prominence", "stripe_extent",
    ]),
}


# ---------------- metadata pop-out (non-modal) ----------------
class _MetadataPopup(QDialog):
    def __init__(self, title: str, parent=None, gallery_id: str = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)                     # non-blocking
        self.setAttribute(Qt.WA_DeleteOnClose)   # close frees memory
        self.resize(540, 600)
        self._gallery_id = gallery_id  # For location history display

        lay = QVBoxLayout(self)
        
        # Metadata table
        lay.addWidget(QLabel("<b>Metadata</b>"))
        self.table = QTableWidget(0, 2, self)
        self.table.setHorizontalHeaderLabels(["Field", "Value"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setWordWrap(True)
        lay.addWidget(self.table)
        
        # Location history section (only for gallery)
        if gallery_id:
            lay.addWidget(QLabel("<b>Location History</b>"))
            self.history_table = QTableWidget(0, 3, self)
            self.history_table.setHorizontalHeaderLabels(["Date", "Location", "Source Query"])
            self.history_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            self.history_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
            self.history_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
            self.history_table.verticalHeader().setVisible(False)
            self.history_table.setMaximumHeight(150)
            lay.addWidget(self.history_table)
            self._load_location_history()
        else:
            self.history_table = None

    def _load_location_history(self):
        """Load and display location history for gallery individual."""
        if not self._gallery_id or not self.history_table:
            return
        try:
            from src.data.location_history import get_location_history
            sightings = get_location_history(self._gallery_id)
            self.history_table.setRowCount(len(sightings))
            for i, s in enumerate(sightings):
                date_str = s.observation_date.strftime("%Y-%m-%d") if s.observation_date else ""
                self.history_table.setItem(i, 0, QTableWidgetItem(date_str))
                self.history_table.setItem(i, 1, QTableWidgetItem(s.location))
                self.history_table.setItem(i, 2, QTableWidgetItem(s.query_id))
            if not sightings:
                self.history_table.setRowCount(1)
                self.history_table.setItem(0, 0, QTableWidgetItem(""))
                self.history_table.setItem(0, 1, QTableWidgetItem("(no location history)"))
                self.history_table.setItem(0, 2, QTableWidgetItem(""))
        except Exception:
            pass

    def populate(self, row: Dict[str, str]):
        fields = [k for k in row.keys() if k]
        self.table.setRowCount(len(fields))
        for i, k in enumerate(fields):
            self.table.setItem(i, 0, QTableWidgetItem(k))
            self.table.setItem(i, 1, QTableWidgetItem(row.get(k, "")))


# ---------------- metadata edit pop-out (non-modal, editable) ----------------
class _MetadataEditPopup(QDialog):
    """
    Editable metadata popup dialog with full MetadataFormV2.
    Allows in-place editing of metadata from the First-order tab.
    """
    saved = Signal()  # Emitted when metadata is saved

    def __init__(self, target: str, id_value: str, parent=None):
        super().__init__(parent)
        self._target = target
        self._id_value = id_value
        self._ilog = get_interaction_logger()

        self.setWindowTitle(f"Edit Metadata: {id_value}")
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.resize(620, 700)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        # Header with ID display
        header = QHBoxLayout()
        header.addWidget(QLabel(f"<b>{target}:</b> {id_value}"))
        header.addStretch(1)
        lay.addLayout(header)

        # Metadata form (reusing MetadataFormV2)
        self.form = MetadataFormV2()
        self.form.set_target(target)
        self.form.set_id_value(id_value)
        lay.addWidget(self.form, 1)

        # Button row
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_save = QPushButton("Save")
        self.btn_save.setDefault(True)
        self.btn_save.clicked.connect(self._on_save)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.close)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_save)
        lay.addLayout(btn_row)

        # Load existing metadata
        self._load_metadata()

    def _load_metadata(self):
        """Load existing metadata from CSV."""
        id_col = ap.id_column_name(self._target)
        csv_paths = self._csv_paths_for_read()
        rows = read_rows_multi(csv_paths)
        latest_map = last_row_per_id(rows, id_col)
        # Normalize ID for lookup
        from src.data.id_registry import normalize_id_value
        data = latest_map.get(normalize_id_value(self._id_value), {})
        data[id_col] = self._id_value
        self.form.populate(data)

    def _csv_paths_for_read(self):
        """Get CSV paths for reading metadata (both old and new locations)."""
        return ap.metadata_csv_paths_for_read(self._target)

    def _on_save(self):
        """Save metadata to CSV."""
        self._ilog.log("button_click", "btn_save_metadata_edit", value=self._id_value,
                      context={"target": self._target})

        # Capture old state before save (for Gallery metadata history)
        old_values = {}
        if self._target == "Gallery":
            old_values = get_current_metadata_for_gallery(self._id_value)

        csv_path, header = ap.metadata_csv_for(self._target)
        row = self.form.collect_row()
        row[ap.id_column_name(self._target)] = self._id_value
        append_row(csv_path, header, row)

        # Record metadata history for Gallery
        if self._target == "Gallery":
            record_bulk_update(
                gallery_id=self._id_value,
                old_values=old_values,
                new_values=row,
                source=SOURCE_UI,
                source_ref="first_order_metadata_edit",
            )

        self.form.mark_clean()
        self.saved.emit()
        self.close()

    def closeEvent(self, event):
        """Warn if there are unsaved changes."""
        if self.form.is_dirty():
            from PySide6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Discard them?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
        event.accept()


# ---------------- promote query to gallery dialog ----------------
class _PromoteQueryDialog(QDialog):
    """
    Dialog to promote a Query to a new Gallery identity.
    Allows user to specify a new gallery ID (defaults to query_id).
    """
    def __init__(self, query_id: str, parent=None):
        super().__init__(parent)
        self._query_id = query_id
        self._ilog = get_interaction_logger()
        
        self.setWindowTitle("Promote Query to Gallery")
        self.setModal(True)
        self.resize(420, 200)
        
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)
        
        # Info label
        info = QLabel(
            f"Create a new Gallery identity from Query <b>{query_id}</b>.\n"
            "This will copy all encounter folders and metadata to the Gallery,\n"
            "and mark the Query as silent (hidden from matching)."
        )
        info.setWordWrap(True)
        lay.addWidget(info)
        
        # Gallery ID input
        id_row = QHBoxLayout()
        id_row.addWidget(QLabel("New Gallery ID:"))
        self.txt_gallery_id = QLineEdit()
        self.txt_gallery_id.setText(query_id)
        self.txt_gallery_id.setPlaceholderText("Enter gallery ID...")
        self.txt_gallery_id.textChanged.connect(self._validate_id)
        id_row.addWidget(self.txt_gallery_id, 1)
        lay.addLayout(id_row)
        
        # Validation message
        self.lbl_validation = QLabel("")
        self.lbl_validation.setStyleSheet("QLabel { color: #c0392b; }")
        lay.addWidget(self.lbl_validation)
        
        # Copy metadata checkbox
        self.chk_copy_metadata = QCheckBox("Copy metadata from query")
        self.chk_copy_metadata.setChecked(True)
        lay.addWidget(self.chk_copy_metadata)
        
        lay.addStretch(1)
        
        # Button box
        self.btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.btn_box.accepted.connect(self.accept)
        self.btn_box.rejected.connect(self.reject)
        self.btn_ok = self.btn_box.button(QDialogButtonBox.Ok)
        self.btn_ok.setText("Promote")
        lay.addWidget(self.btn_box)
        
        # Initial validation
        self._validate_id()
    
    def _validate_id(self):
        """Validate the gallery ID and update UI accordingly."""
        from src.data.validators import validate_id
        from src.data.id_registry import id_exists
        
        gallery_id = self.txt_gallery_id.text().strip()
        
        if not gallery_id:
            self.lbl_validation.setText("Gallery ID cannot be empty.")
            self.btn_ok.setEnabled(False)
            return
        
        v = validate_id(gallery_id)
        if not v.ok:
            self.lbl_validation.setText(v.message)
            self.btn_ok.setEnabled(False)
            return
        
        if id_exists("Gallery", gallery_id):
            self.lbl_validation.setText(f"Gallery ID '{gallery_id}' already exists.")
            self.btn_ok.setEnabled(False)
            return
        
        self.lbl_validation.setText("")
        self.btn_ok.setEnabled(True)
    
    def get_gallery_id(self) -> str:
        """Return the gallery ID entered by the user."""
        return self.txt_gallery_id.text().strip()
    
    def should_copy_metadata(self) -> bool:
        """Return whether to copy metadata from query."""
        return self.chk_copy_metadata.isChecked()


# ---------------- exclusion dialog ----------------
class _ExcludeDialog(QDialog):
    """
    Lets the user temporarily exclude specific gallery IDs and/or
    exclude all items from selected 'Last location' values.
    Also supports filtering to INCLUDE only individuals seen at specific locations (location history).
    Exclusions are not persisted; caller should store results in memory.
    """
    def __init__(self, all_gallery_ids: List[str], last_locations: List[str],
                 excluded_ids: Set[str], excluded_locations: Set[str], 
                 include_history_locations: Set[str] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filter ranking")
        self.resize(560, 750)
        
        self._ilog = get_interaction_logger()

        self._ids_initial = set(excluded_ids)
        self._locs_initial = set(excluded_locations)
        self._hist_locs_initial = set(include_history_locations or set())

        self._ids_out: Set[str] = set(excluded_ids)
        self._locs_out: Set[str] = set(excluded_locations)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # --- by gallery ID
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("<b>Exclude gallery members (IDs)</b>"))
        row1.addStretch(1)
        self.txt_filter = QLineEdit()
        self.txt_filter.setPlaceholderText("filter IDs…")
        self.txt_filter.textChanged.connect(self._apply_filter)
        row1.addWidget(self.txt_filter)
        root.addLayout(row1)

        self.list_ids = QListWidget()
        self.list_ids.setSelectionMode(QListWidget.NoSelection)
        for gid in all_gallery_ids:
            it = QListWidgetItem(gid)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if gid in excluded_ids else Qt.Unchecked)
            self.list_ids.addItem(it)
        root.addWidget(self.list_ids, 1)

        # --- by location (current metadata)
        root.addWidget(QLabel("<b>Exclude by current location</b>"))
        row2 = QHBoxLayout()
        self.cmb_location = QComboBox()
        self.cmb_location.addItem("— choose location —")
        for loc in sorted({l for l in last_locations if l.strip()}):
            self.cmb_location.addItem(loc)
        row2.addWidget(self.cmb_location)
        btn_add_loc = QPushButton("Add")
        btn_add_loc.clicked.connect(self._on_add_location)
        row2.addWidget(btn_add_loc)
        btn_clear_locs = QPushButton("Clear")
        btn_clear_locs.clicked.connect(self._on_clear_locations)
        row2.addWidget(btn_clear_locs)
        row2.addStretch(1)
        root.addLayout(row2)

        self.list_locations = QListWidget()
        self.list_locations.setSelectionMode(QListWidget.NoSelection)
        for loc in sorted(excluded_locations):
            self.list_locations.addItem(QListWidgetItem(loc))
        root.addWidget(self.list_locations, 0)

        # --- by location history (INCLUDE filter)
        root.addWidget(QLabel("<b>Include only individuals seen at (location history)</b>"))
        row3 = QHBoxLayout()
        self.cmb_history_location = QComboBox()
        self.cmb_history_location.addItem("— choose location —")
        # Get all historical locations
        try:
            from src.data.location_history import get_all_historical_locations
            hist_locs = get_all_historical_locations(all_gallery_ids)
            for loc in sorted(hist_locs):
                self.cmb_history_location.addItem(loc)
        except Exception:
            pass
        row3.addWidget(self.cmb_history_location)
        btn_add_hist = QPushButton("Add")
        btn_add_hist.clicked.connect(self._on_add_history_location)
        row3.addWidget(btn_add_hist)
        btn_clear_hist = QPushButton("Clear")
        btn_clear_hist.clicked.connect(self._on_clear_history_locations)
        row3.addWidget(btn_clear_hist)
        row3.addStretch(1)
        root.addLayout(row3)

        self.list_history_locations = QListWidget()
        self.list_history_locations.setSelectionMode(QListWidget.NoSelection)
        for loc in sorted(include_history_locations or set()):
            self.list_history_locations.addItem(QListWidgetItem(loc))
        root.addWidget(self.list_history_locations, 0)
        
        # Help text
        help_lbl = QLabel("<i>Location history filter shows only individuals ever seen at selected locations.</i>")
        help_lbl.setWordWrap(True)
        root.addWidget(help_lbl)

        # --- buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _apply_filter(self, text: str):
        text = (text or "").strip().lower()
        for i in range(self.list_ids.count()):
            it = self.list_ids.item(i)
            it.setHidden(text not in it.text().lower())

    def _on_add_location(self):
        loc = self.cmb_location.currentText().strip()
        if not loc or loc.startswith("—"):
            return
        existing = [self.list_locations.item(i).text() for i in range(self.list_locations.count())]
        if loc not in existing:
            self._ilog.log("button_click", "btn_add_location", value=loc)
            self.list_locations.addItem(QListWidgetItem(loc))

    def _on_clear_locations(self):
        self.list_locations.clear()

    def _on_add_history_location(self):
        loc = self.cmb_history_location.currentText().strip()
        if not loc or loc.startswith("—"):
            return
        existing = [self.list_history_locations.item(i).text() for i in range(self.list_history_locations.count())]
        if loc not in existing:
            self._ilog.log("button_click", "btn_add_history_location", value=loc)
            self.list_history_locations.addItem(QListWidgetItem(loc))

    def _on_clear_history_locations(self):
        self.list_history_locations.clear()

    def get_results(self) -> Tuple[Set[str], Set[str], Set[str]]:
        """Returns (excluded_ids, excluded_locations, include_history_locations)."""
        ids: Set[str] = set()
        for i in range(self.list_ids.count()):
            it = self.list_ids.item(i)
            if it.checkState() == Qt.Checked:
                ids.add(it.text())
        locs: Set[str] = {self.list_locations.item(i).text() for i in range(self.list_locations.count())}
        hist_locs: Set[str] = {self.list_history_locations.item(i).text() for i in range(self.list_history_locations.count())}
        return ids, locs, hist_locs


# ---------------- Collapsible wrapper ----------------
class CollapsibleSection(QWidget):
    """
    Simple collapsible header with one content widget (e.g., a QScrollArea).
    Emits `toggled(bool)` when expanded/collapsed.
    """
    toggled = Signal(bool)

    def __init__(self, title: str, start_collapsed: bool = True, parent: QWidget | None = None):
        super().__init__(parent)
        self._expanded = not start_collapsed

        self.toggle = QToolButton(self)
        self.toggle.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(self._expanded)
        self.toggle.setArrowType(Qt.DownArrow if self._expanded else Qt.RightArrow)
        self.toggle.setText(title)
        self.toggle.toggled.connect(self._on_toggled)

        self.line = QFrame(self)
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.content = QWidget(self)
        self.content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(0)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(self.toggle)
        lay.addWidget(self.line)
        lay.addWidget(self.content, 1)
        self._apply_visibility()

    def setContent(self, widget: QWidget):
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.content_layout.addWidget(widget)

    def _on_toggled(self, checked: bool):
        self._expanded = checked
        self.toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self._apply_visibility()
        self.toggled.emit(checked)

    def _apply_visibility(self):
        self.content.setVisible(self._expanded)


class TabFirstOrder(QWidget):
    """
    First-order ranking & line-up.

    New in this update:
    - **Query date filter**: filter the Query combo by last sampling date (From/To; inclusive)
      with optional inclusion of queries that have no detected date.
    - “Exclude…” button to temporarily exclude specific gallery IDs and/or by Last location.
    - Automatic background exclusion: if a gallery member already has a **Yes** match for
      any query observed on the same day as the current query, it is hidden for this ranking.
    - Queries that already have a **Yes** match are pushed to the bottom of the Query combo.
    - Existing improvements retained: resizable panels, scrollable fields, numeric offsets, pins, metadata pop‑out.
    - **CHANGE:** Include fields + Numeric Offsets combined into a single collapsible, scrollable section.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine = FirstOrderSearchEngine()
        self._pinned: List[str] = []
        self._current_query: str = ""
        self._cards: List[LineupCard] = []
        self._meta_popup: _MetadataPopup | None = None
        self._meta_edit_popup: _MetadataEditPopup | None = None

        # session-scoped exclusions (NOT persisted)
        self._excluded_ids: Set[str] = set()
        self._excluded_locations: Set[str] = set()
        self._include_history_locations: Set[str] = set()  # location history filter (include mode)

        # date filter state
        self._dates_initialized: bool = False
        
        # Flag to suppress file watcher during internal saves (Option A fix)
        self._suppress_csv_watch: bool = False
        
        # Interaction logger for user analytics
        self._ilog = get_interaction_logger()
        
        # Stored evaluation data for query sorting by confidence
        self._stored_evaluation = None  # StoredEvaluation or None
        self._sort_by_confidence = False  # Whether to sort queries by similarity

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        # --- Controls row 1: Query, Preset, Top-K, action buttons ---
        controls_row1 = QHBoxLayout()
        
        # Sort by confidence toggle (before Query combo)
        self.chk_sort_confidence = QCheckBox("🎯")
        self.chk_sort_confidence.setToolTip(
            "Sort queries by match confidence (highest similarity first).\n"
            "Requires evaluation to be run in Deep Learning tab."
        )
        self.chk_sort_confidence.setEnabled(False)  # Enabled when evaluation available
        self.chk_sort_confidence.toggled.connect(self._on_sort_confidence_toggled)
        controls_row1.addWidget(self.chk_sort_confidence)
        
        controls_row1.addWidget(QLabel("Query:"))
        self.cmb_query = QComboBox()
        self.cmb_query.setMinimumWidth(320)
        # Make combo editable for type-to-search functionality
        self.cmb_query.setEditable(True)
        self.cmb_query.setInsertPolicy(QComboBox.NoInsert)
        self.cmb_query.completer().setFilterMode(Qt.MatchContains)
        self.cmb_query.completer().setCompletionMode(QCompleter.PopupCompletion)
        # Apply state-based color coding delegate
        self._query_state_delegate = QueryStateDelegate(self.cmb_query)
        self.cmb_query.setItemDelegate(self._query_state_delegate)
        controls_row1.addWidget(self.cmb_query)

        # Query navigation buttons
        self.btn_prev_query = QPushButton("◀")
        self.btn_prev_query.setFixedWidth(28)
        self.btn_prev_query.setToolTip("Previous query in list")
        self.btn_prev_query.clicked.connect(self._on_prev_query_clicked)
        controls_row1.addWidget(self.btn_prev_query)

        self.btn_next_query = QPushButton("▶")
        self.btn_next_query.setFixedWidth(28)
        self.btn_next_query.setToolTip("Next query in list")
        self.btn_next_query.clicked.connect(self._on_next_query_clicked)
        controls_row1.addWidget(self.btn_next_query)

        controls_row1.addWidget(QLabel("Preset:"))
        self.cmb_preset = QComboBox()
        self.cmb_preset.addItems(list(PRESETS.keys()))
        self.cmb_preset.currentIndexChanged.connect(self._apply_preset)
        controls_row1.addWidget(self.cmb_preset)

        controls_row1.addWidget(QLabel("Top-K:"))
        self.spin_topk = QSpinBox()
        self.spin_topk.setRange(1, 500)
        self.spin_topk.setValue(50)
        self.spin_topk.valueChanged.connect(
            lambda v: self._ilog.log("spin_change", "spin_topk", value=str(v)))
        controls_row1.addWidget(self.spin_topk)

        self.btn_rebuild = QPushButton("Rebuild index")
        self.btn_rebuild.clicked.connect(self._on_rebuild)
        controls_row1.addWidget(self.btn_rebuild)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self._refresh_results)
        self.btn_refresh.clicked.connect(
            lambda: self._ilog.log("button_click", "btn_refresh", value="clicked"))
        controls_row1.addWidget(self.btn_refresh)

        self.btn_exclude = QPushButton("Exclude…")
        self.btn_exclude.setToolTip("Temporarily exclude gallery members or entire locations")
        self.btn_exclude.clicked.connect(self._open_exclude_dialog)
        controls_row1.addWidget(self.btn_exclude)

        self.btn_config = QPushButton("Set up config…")
        self.btn_config.setToolTip("Configure field weights, enabled fields, and scorer parameters")
        self.btn_config.clicked.connect(self._open_config_dialog)
        controls_row1.addWidget(self.btn_config)

        controls_row1.addStretch(1)
        outer.addLayout(controls_row1)

        # --- Controls row 2: Date filter, Visual/DL controls, Fusion, status ---
        controls_row2 = QHBoxLayout()

        # ---- Query date filter (From/To + include-no-date) ----
        self.chk_date = QCheckBox("Date filter")
        self.chk_date.toggled.connect(self._on_date_filter_changed)
        controls_row2.addWidget(self.chk_date)

        controls_row2.addWidget(QLabel("From:"))
        self.date_from = QDateEdit()
        self.date_from.setCalendarPopup(True)
        self.date_from.setDisplayFormat("yyyy-MM-dd")
        self.date_from.dateChanged.connect(self._on_date_filter_changed)
        controls_row2.addWidget(self.date_from)

        controls_row2.addWidget(QLabel("To:"))
        self.date_to = QDateEdit()
        self.date_to.setCalendarPopup(True)
        self.date_to.setDisplayFormat("yyyy-MM-dd")
        self.date_to.dateChanged.connect(self._on_date_filter_changed)
        controls_row2.addWidget(self.date_to)

        self.chk_include_nodate = QCheckBox("Include no‑date")
        self.chk_include_nodate.setToolTip("Include queries with no detectable sampling date")
        self.chk_include_nodate.toggled.connect(self._on_date_filter_changed)
        controls_row2.addWidget(self.chk_include_nodate)

        # initialize enabled/disabled state
        self._update_date_widgets_enabled()

        # ---- Visual Ranking (DL) controls ----
        controls_row2.addSpacing(20)
        self.chk_visual = QCheckBox("Visual")
        self.chk_visual.setToolTip("Enable deep learning visual similarity ranking")
        self.chk_visual.toggled.connect(self._on_visual_toggled)
        controls_row2.addWidget(self.chk_visual)

        self.cmb_model = QComboBox()
        self.cmb_model.setMinimumWidth(120)
        self.cmb_model.setToolTip("Select the visual model to use")
        self.cmb_model.currentIndexChanged.connect(self._on_model_changed)
        controls_row2.addWidget(self.cmb_model)

        # Visual mode selector (Centroid vs Image-Based)
        self.cmb_visual_mode = QComboBox()
        self.cmb_visual_mode.addItem("Image", "image")  # Default
        self.cmb_visual_mode.addItem("Centroid", "centroid")
        self.cmb_visual_mode.setToolTip("Image: rank by current image; Centroid: rank by identity average")
        self.cmb_visual_mode.currentIndexChanged.connect(self._on_visual_mode_changed)
        controls_row2.addWidget(self.cmb_visual_mode)

        # Refresh button for image-based mode
        self.btn_refresh_visual = QPushButton("↻")
        self.btn_refresh_visual.setFixedWidth(28)
        self.btn_refresh_visual.setToolTip("Refresh ranking for current image")
        self.btn_refresh_visual.clicked.connect(self._on_refresh_visual)
        controls_row2.addWidget(self.btn_refresh_visual)

        # Roll to closest toggle (for image-based mode)
        self.chk_roll_to_closest = QCheckBox("Roll to closest")
        self.chk_roll_to_closest.setToolTip(
            "When enabled, top-ranked gallery identities show the image "
            "most similar to the current query image"
        )
        self.chk_roll_to_closest.setChecked(False)
        self.chk_roll_to_closest.toggled.connect(self._on_roll_to_closest_toggled)
        controls_row2.addWidget(self.chk_roll_to_closest)

        # Limit for roll-to-closest (only apply to top N results)
        self.spin_roll_limit = QSpinBox()
        self.spin_roll_limit.setRange(1, 100)
        self.spin_roll_limit.setValue(5)
        self.spin_roll_limit.setToolTip(
            "Only roll to closest for the top N gallery identities.\n"
            "DL similarity is most meaningful for top matches; lower ranks show curated best photo."
        )
        self.spin_roll_limit.setFixedWidth(50)
        self.spin_roll_limit.valueChanged.connect(self._on_roll_to_closest_toggled)
        controls_row2.addWidget(self.spin_roll_limit)

        controls_row2.addWidget(QLabel("Fusion:"))
        self.slider_fusion = QSlider(Qt.Horizontal)
        self.slider_fusion.setRange(0, 100)
        self.slider_fusion.setValue(50)
        self.slider_fusion.setFixedWidth(80)
        self.slider_fusion.setToolTip("0% = metadata only, 100% = visual only")
        self.slider_fusion.valueChanged.connect(self._on_fusion_changed)
        controls_row2.addWidget(self.slider_fusion)
        self.lbl_fusion = QLabel("50%")
        self.lbl_fusion.setFixedWidth(35)
        controls_row2.addWidget(self.lbl_fusion)

        # ---- Verification controls (P(same) from verification model) ----
        controls_row2.addSpacing(16)
        self.chk_verification = QCheckBox("Verification")
        self.chk_verification.setToolTip(
            "Use verification model P(same) scores for ranking.\n"
            "Requires precomputed verification data from Deep Learning tab."
        )
        self.chk_verification.toggled.connect(self._on_verification_toggled)
        controls_row2.addWidget(self.chk_verification)

        self.cmb_verif_model = QComboBox()
        self.cmb_verif_model.setMinimumWidth(100)
        self.cmb_verif_model.setToolTip("Select verification model")
        self.cmb_verif_model.currentIndexChanged.connect(self._on_verif_model_changed)
        controls_row2.addWidget(self.cmb_verif_model)

        controls_row2.addWidget(QLabel("V-Fusion:"))
        self.slider_verif_fusion = QSlider(Qt.Horizontal)
        self.slider_verif_fusion.setRange(0, 100)
        self.slider_verif_fusion.setValue(50)
        self.slider_verif_fusion.setFixedWidth(80)
        self.slider_verif_fusion.setToolTip(
            "Blend verification with metadata/visual:\n"
            "0% = no verification, 100% = verification only"
        )
        self.slider_verif_fusion.valueChanged.connect(self._on_verif_fusion_changed)
        controls_row2.addWidget(self.slider_verif_fusion)
        self.lbl_verif_fusion = QLabel("50%")
        self.lbl_verif_fusion.setFixedWidth(35)
        controls_row2.addWidget(self.lbl_verif_fusion)

        # Initialize visual ranking state
        self._visual_available = False
        self._visual_lookup = None
        self._image_lookup = None  # For image-based mode
        # {gallery_id: (local_idx, path)} - both for robustness (try index, fallback to path)
        self._closest_gallery_image: Dict[str, Tuple[int, str]] = {}
        self._init_visual_controls()

        # Initialize verification ranking state
        self._verification_available = False
        self._verification_lookup = None
        self._init_verification_controls()
        
        # Load stored evaluation for query sorting by confidence
        self._load_stored_evaluation()

        controls_row2.addStretch(1)
        self.lbl_excluded = QLabel("Excluded: 0")
        controls_row2.addWidget(self.lbl_excluded)
        self.lbl_pinned = QLabel("Pinned: 0")
        controls_row2.addWidget(self.lbl_pinned)
        outer.addLayout(controls_row2)

        # ------------- Include fields panel (now placed raw; outer scroll wraps both) -------------
        gb_fields = QGroupBox("Include fields (averaged by default)")
        gb_fields.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        gb_fields_lay = QVBoxLayout(gb_fields)
        gb_fields_lay.setContentsMargins(8, 4, 8, 4)
        gb_fields_lay.setSpacing(4)

        content_fields = QWidget()
        grid = QGridLayout(content_fields)
        grid.setContentsMargins(4, 2, 4, 2)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(4)

        self.chk_by_name: Dict[str, QCheckBox] = {}
        row = 0
        col = 0
        for group_title, fields in FIELD_GROUPS:
            title = QLabel(f"<b>{group_title}</b>")
            grid.addWidget(title, row, col, 1, 1)
            row += 1
            for f in fields:
                chk = QCheckBox(f)
                chk.setChecked(True)
                # Log field toggle
                chk.toggled.connect(
                    lambda checked, field=f: self._ilog.log(
                        "checkbox_toggle", f"chk_field_{field}", value=str(checked)))
                self.chk_by_name[f] = chk
                grid.addWidget(chk, row, col, 1, 1)
                row += 1
            # next column
            row = 0
            col += 1

        gb_fields_lay.addWidget(content_fields)

        # ------------- Numeric Offsets (placed raw; outer scroll wraps both) -------------
        gb_offsets = QGroupBox("Numeric Offsets (applied to query values)")
        gb_offsets.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        lay_off_outer = QVBoxLayout(gb_offsets)
        lay_off_outer.setContentsMargins(8, 4, 8, 4)
        lay_off_outer.setSpacing(4)

        content_off = QWidget()
        grid_off = QGridLayout(content_off)
        grid_off.setContentsMargins(6, 2, 6, 2)
        grid_off.setHorizontalSpacing(8)
        grid_off.setVerticalSpacing(2)

        self._offset_widgets: Dict[str, QSpinBox | QDoubleSpinBox] = {}

        def _dspin(minv: float, maxv: float, step: float, dec: int, suffix: str = "") -> QDoubleSpinBox:
            w = QDoubleSpinBox()
            w.setRange(float(minv), float(maxv))
            w.setDecimals(int(dec))
            w.setSingleStep(float(step))
            w.setMaximumWidth(120)
            if suffix:
                w.setSuffix(f" {suffix}")
            w.valueChanged.connect(self._on_offsets_changed)
            return w

        def _ispin(minv: int, maxv: int, step: int = 1, suffix: str = "") -> QSpinBox:
            w = QSpinBox()
            w.setRange(int(minv), int(maxv))
            w.setSingleStep(int(step))
            w.setMaximumWidth(100)
            if suffix:
                w.setSuffix(f" {suffix}")
            w.valueChanged.connect(self._on_offsets_changed)
            return w

        r = 0
        # Numeric field offsets
        grid_off.addWidget(QLabel("tip_to_tip_size_cm Δ"), r, 0, Qt.AlignRight)
        self._offset_widgets["tip_to_tip_size_cm"] = _dspin(-100.0, 100.0, 0.5, 2, "cm"); grid_off.addWidget(self._offset_widgets["tip_to_tip_size_cm"], r, 1); r += 1
        grid_off.addWidget(QLabel("num_apparent_arms Δ"), r, 0, Qt.AlignRight)
        self._offset_widgets["num_apparent_arms"] = _ispin(-10, 10, 1); grid_off.addWidget(self._offset_widgets["num_apparent_arms"], r, 1); r += 1
        grid_off.addWidget(QLabel("num_total_arms Δ"), r, 0, Qt.AlignRight)
        self._offset_widgets["num_total_arms"] = _ispin(-10, 10, 1); grid_off.addWidget(self._offset_widgets["num_total_arms"], r, 1); r += 1

        btn_reset_offsets = QPushButton("Reset offsets")
        btn_reset_offsets.clicked.connect(self._reset_offsets)
        grid_off.addWidget(btn_reset_offsets, r, 0, 1, 2, Qt.AlignLeft)

        lay_off_outer.addWidget(content_off)

        # ------------- Split area: left Query | right Gallery -------------
        self.hsplit = QSplitter(Qt.Horizontal)

        # Query panel (with a vertical splitter so the image area can be resized)
        self.query_panel = QGroupBox("Query")
        qwrap = _QW()
        qwrap_l = _QVL(qwrap)
        qwrap_l.setContentsMargins(8, 8, 8, 8)
        
        # Query ID row with suggested match
        id_row = QHBoxLayout()
        id_row.addWidget(QLabel("ID:"))
        self.lbl_query_id = QLabel("—")
        id_row.addWidget(self.lbl_query_id)
        id_row.addSpacing(12)
        self.lbl_suggested_match = QLabel("")
        self.lbl_suggested_match.setStyleSheet("color: #1b9e77; font-weight: bold;")
        self.lbl_suggested_match.setToolTip("Top suggested match from evaluation")
        id_row.addWidget(self.lbl_suggested_match)
        id_row.addStretch(1)
        qwrap_l.addLayout(id_row)

        self.query_strip = ImageStrip(files=[], long_edge=768)

        # query footer (Open Folder + View Metadata + Image Quality Panel)
        q_footer = QWidget()
        qf_l = QVBoxLayout(q_footer)
        qf_l.setContentsMargins(0, 0, 0, 0)
        qf_l.setSpacing(4)
        
        # Image quality panel (compact horizontal layout)
        self.query_quality_panel = ImageQualityPanel(
            parent=q_footer,
            show_save_button=True,
            compact=True,
            title="",
        )
        self.query_quality_panel.set_target("Queries")
        self.query_quality_panel.saved.connect(self._on_query_quality_saved)
        qf_l.addWidget(self.query_quality_panel)
        
        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        self.btn_open_query = QPushButton("Open Folder")
        self.btn_best_query = QPushButton("Set Best (Query)")
        self.btn_open_query.clicked.connect(self._open_query_folder)
        self.btn_best_query.clicked.connect(self._on_set_best_query)
        self.btn_meta_query = QPushButton("View Metadata")
        self.btn_meta_query.clicked.connect(self._show_query_metadata)
        self.btn_edit_meta_query = QPushButton("Edit Metadata")
        self.btn_edit_meta_query.clicked.connect(self._edit_query_metadata)
        self.btn_promote_query = QPushButton("Promote to Gallery")
        self.btn_promote_query.setToolTip(
            "Create a new Gallery identity from this Query.\n"
            "Use when this is a unique individual not yet in the Gallery."
        )
        self.btn_promote_query.clicked.connect(self._on_promote_query)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_open_query)
        btn_row.addWidget(self.btn_best_query)
        btn_row.addWidget(self.btn_meta_query)
        btn_row.addWidget(self.btn_edit_meta_query)
        btn_row.addWidget(self.btn_promote_query)
        qf_l.addLayout(btn_row)

        self.query_vsplit = QSplitter(Qt.Vertical)
        qtop = _QW()
        qtop_l = _QVL(qtop)
        qtop_l.setContentsMargins(0, 0, 0, 0)
        qtop_l.addWidget(self.query_strip)
        qbot = _QW()
        qbot_l = _QVL(qbot)
        qbot_l.setContentsMargins(0, 0, 0, 0)
        qbot_l.addWidget(q_footer)

        self.query_vsplit.addWidget(qtop)
        self.query_vsplit.addWidget(qbot)
        self.query_vsplit.setCollapsible(0, False)
        self.query_vsplit.setCollapsible(1, True)
        self.query_vsplit.setSizes([420, 80])
        self.query_vsplit.splitterMoved.connect(self._on_query_split_resized)

        # Track horizontal splitter (Query panel width changes)
        self.hsplit.splitterMoved.connect(self._on_query_panel_resized)

        # Keep gallery cards' minimum image height in sync with the Query view
        self.query_strip.pixmapResized.connect(self._on_query_pixmap_resized)
        QTimer.singleShot(0, self._sync_card_min_height_from_query)  # first sync after layout

        qwrap_l.addWidget(self.query_vsplit, 1)
        qp_lay = QVBoxLayout(self.query_panel)
        qp_lay.setContentsMargins(0, 0, 0, 0)
        qp_lay.addWidget(qwrap)

        self.hsplit.addWidget(self.query_panel)

        # Gallery (right): horizontal scroll of cards
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.viewport().installEventFilter(self)

        self.cards_container = QWidget()
        self.cards_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.cards_layout = QHBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(8)
        self.scroll.setWidget(self.cards_container)

        self.hsplit.addWidget(self.scroll)

        # ------------- Master vertical splitter: Filters (collapsible) | Query/Gallery -------------
        # Combine both groups under a single scrollable, collapsible section.
        filters_content = QWidget()
        filters_lay = QVBoxLayout(filters_content)
        filters_lay.setContentsMargins(0, 0, 0, 0)
        filters_lay.setSpacing(8)
        filters_lay.addWidget(gb_fields)
        filters_lay.addWidget(gb_offsets)
        filters_lay.addStretch(1)

        filters_scroll = QScrollArea()
        filters_scroll.setWidgetResizable(True)
        filters_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        filters_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        filters_scroll.setWidget(filters_content)

        self.filters_section = CollapsibleSection("Fields & Offsets", start_collapsed=True, parent=self)
        self.filters_section.setContent(filters_scroll)
        self.filters_section.toggled.connect(self._on_filters_toggled)

        self.vsplit_all = QSplitter(Qt.Vertical)
        self.vsplit_all.setHandleWidth(6)
        self.vsplit_all.setChildrenCollapsible(False)
        self.vsplit_all.addWidget(self.filters_section)
        self.vsplit_all.addWidget(self.hsplit)

        self.vsplit_all.setStretchFactor(0, 0)
        self.vsplit_all.setStretchFactor(1, 1)
        self.vsplit_all.setSizes([160, 800])

        outer.addWidget(self.vsplit_all, 1)

        # Sync field checkboxes with config BEFORE populating queries
        # (otherwise the first ranking uses all fields instead of config)
        self._sync_checkboxes_from_config()

        # Populate query IDs and hook signals
        self._refresh_query_ids()
        self.cmb_query.currentIndexChanged.connect(self._on_query_changed)

        # --- Auto-reload First-order on metadata saves (watch CSVs) ---
        self._meta_watcher = QFileSystemWatcher(self)
        try:
            # Ensure the write CSVs exist so we can watch them
            from src.data.csv_io import ensure_header
            g_csv, g_header = ap.metadata_csv_for("Gallery")
            q_csv, q_header = ap.metadata_csv_for("Queries")
            ensure_header(g_csv, g_header)
            ensure_header(q_csv, q_header)
            self._meta_watcher.addPath(str(g_csv))
            self._meta_watcher.addPath(str(q_csv))
            self._meta_watcher.fileChanged.connect(self._on_metadata_csv_changed)
        except Exception:
            pass

    # ---------------- Qt events ----------------
    def eventFilter(self, obj, ev):
        if obj is self.scroll.viewport() and ev.type() == QEvent.Resize:
            QTimer.singleShot(0, self._fit_cards_to_viewport)
        return super().eventFilter(obj, ev)

    # ---------------- pins persistence ----------------
    def _pins_path(self, qid: str):
        return ap.queries_root(prefer_new=True) / qid / "_pins_first_order.json"

    def _load_pins(self, qid: str) -> List[str]:
        try:
            p = self._pins_path(qid)
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                return list(dict.fromkeys([str(x) for x in data.get("pinned", [])]))
        except Exception:
            pass
        return []

    def _save_pins(self):
        qid = self._current_query
        if not qid:
            return
        try:
            p = self._pins_path(qid)
            p.parent.mkdir(parents=True, exist_ok=True)
            obj = {"query_id": qid, "pinned": list(self._pinned)}
            p.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        except Exception:
            pass

    # ---------------- helpers ----------------
    def _on_query_split_resized(self, *_):
        # Any time the Query pane's vertical split changes, propagate the new min height.
        self._sync_card_min_height_from_query()

    def _refresh_query_ids(self):
        prev_sel = self.cmb_query.currentText()
        # Use central registry with unified silence logic
        ids = list_ids("Queries", exclude_silent=True)

        # date bounds init and filter
        last_obs = last_observation_for_all("Queries")
        self._ensure_date_controls_bounds(last_obs)
        ids = self._apply_query_date_filter(ids, last_obs)

        # Sort key: (no_date_last, oldest_to_newest, alpha_id)
        def _date_alpha_key(qid: str):
            d = last_obs.get(qid)
            return (d is None, d or _date.max, qid.lower())

        # Compute query states and move matched queries to the bottom
        # States: NOT_ATTEMPTED (no labels), ATTEMPTED (has labels, no yes), MATCHED (has yes)
        query_states: Dict[str, QueryState] = {}
        unmatched: List[str] = []
        matched: List[str] = []
        for qid in ids:
            state = get_query_state(qid)
            query_states[qid] = state
            if state == QueryState.MATCHED:
                matched.append(qid)
            else:
                unmatched.append(qid)
        
        # Apply confidence sorting if enabled, otherwise use date/alpha sorting
        if self._sort_by_confidence and self._stored_evaluation:
            # Sort unmatched queries by confidence (highest first)
            ordered = self._get_confidence_sorted_queries(unmatched) + sorted(matched, key=_date_alpha_key)
        else:
            ordered = sorted(unmatched, key=_date_alpha_key) + sorted(matched, key=_date_alpha_key)

        self.cmb_query.blockSignals(True)
        self.cmb_query.clear()
        self.cmb_query.addItems(ordered)
        
        # Apply state-based color coding to combo items
        apply_query_states_to_combobox(self.cmb_query, ordered, query_states)
        # Apply quality indicator symbols
        apply_quality_to_combobox(self.cmb_query, ordered, "Queries")
        
        # try to restore previous selection if still present
        if prev_sel and (idx := self.cmb_query.findText(prev_sel)) >= 0:
            self.cmb_query.setCurrentIndex(idx)
        elif ordered:
            self.cmb_query.setCurrentIndex(0)
        self.cmb_query.blockSignals(False)

        # Option D: Only call _on_query_changed if query actually changed
        # This preserves view state when the same query is still selected
        if ordered:
            new_sel = self.cmb_query.currentText()
            if new_sel != prev_sel:
                # Different query selected - full refresh needed
                self._on_query_changed()
            elif new_sel != self._current_query:
                # Selection restored but internal state out of sync - refresh
                self._on_query_changed()
            # else: same query, same selection - preserve view state, skip refresh
        else:
            # no queries after filter
            self._current_query = ""
            self.lbl_query_id.setText("—")
            self.query_strip.set_files([])
            self._set_cards([])

    def export_current_query_selection(self) -> Dict[str, object]:
        """
        Return the currently selected *Query* context so other tabs can
        re-create the same image and view.

        Dict keys:
          - query_id: str
          - image_path: str (absolute path of the selected query image)
          - view_state: dict from ImageStrip.get_view_state()
        """
        qid = (self._current_query or "").strip()
        files = list(getattr(self, "query_strip", None).files or []) if hasattr(self, "query_strip") else []
        idx = int(getattr(getattr(self, "query_strip", None), "idx", 0)) if hasattr(self, "query_strip") else 0
        img = files[idx] if (files and 0 <= idx < len(files)) else None
        view_state = getattr(self.query_strip, "get_view_state", lambda: {})() if files else {}

        return {
            "query_id": qid,
            "image_path": str(img) if img else "",
            "view_state": view_state,
        }

    def _on_query_panel_resized(self, *_):
        """When the main left/right splitter moves, update gallery min width."""
        self._sync_card_min_width_from_query()

    def _query_viewport_width(self) -> int:
        """Current visible width of the Query viewer viewport."""
        try:
            return int(self.query_strip.view.viewport().width())
        except Exception:
            return 300  # safe default

    def _sync_card_min_width_from_query(self) -> None:
        """Push current Query viewer width as minimum width for all gallery cards."""
        w = max(500, self._query_viewport_width())
        for c in getattr(self, "_cards", []):
            if hasattr(c, "set_min_image_width"):
                try:
                    c.set_min_image_width(w)
                except Exception:
                    pass

    def _apply_preset(self):
        name = self.cmb_preset.currentText()
        
        # Log preset selection
        self._ilog.log("combo_change", "cmb_preset", value=name)
        
        # "Config" preset reads from fields_config.yaml
        if name == PRESET_CONFIG:
            self._sync_checkboxes_from_config()
            return
        
        fields = PRESETS.get(name, set())
        for f, chk in self.chk_by_name.items():
            chk.setChecked(f in fields)

    def _selected_fields(self) -> Set[str]:
        return {f for f, chk in self.chk_by_name.items() if chk.isChecked()}

    def _on_rebuild(self):
        self._ilog.log("button_click", "btn_rebuild", value="clicked")
        # Reset built flag to force a full rebuild when user explicitly requests it
        self.engine.reset_built()
        self.engine.rebuild()
        # Rebuild can change date coverage; refresh combo accordingly
        self._dates_initialized = False
        self._refresh_query_ids()
        self._refresh_results()

    def _open_query_folder(self):
        qid = self._current_query
        if not qid:
            return
        self._ilog.log("file_open", "btn_open_query", value=qid)
        folder = ap.queries_root(prefer_new=True) / qid
        try:
            if platform.system() == "Windows":
                os.startfile(str(folder))
            elif platform.system() == "Darwin":
                subprocess.call(["open", str(folder)])
            else:
                subprocess.call(["xdg-open", str(folder)])
        except Exception:
            pass

    def _show_query_metadata(self):
        if not self._current_query:
            return
        self._ilog.log("button_click", "btn_meta_query", value=self._current_query)
        q_row = self.engine._queries_rows_by_id.get(self._current_query, {})
        if not q_row:
            return
        if self._meta_popup is None or not self._meta_popup.isVisible():
            self._meta_popup = _MetadataPopup(f"Metadata: {self._current_query}", self)
            # Clear reference when C++ object is deleted (due to WA_DeleteOnClose)
            self._meta_popup.destroyed.connect(lambda: setattr(self, '_meta_popup', None))
        self._meta_popup.populate(q_row)
        self._meta_popup.show()
        self._meta_popup.raise_()

    def _show_gallery_metadata(self, gallery_id: str):
        """Show metadata popup for a gallery ID."""
        g_row = self.engine._gallery_rows_by_id.get(gallery_id, {})
        if not g_row:
            return
        popup = _MetadataPopup(f"Metadata: {gallery_id}", self, gallery_id=gallery_id)
        popup.destroyed.connect(lambda: None)  # prevent crash; non-modal so no caching
        popup.populate(g_row)
        popup.show()
        popup.raise_()

    def _edit_query_metadata(self):
        """Open edit metadata popup for the current query."""
        if not self._current_query:
            return
        self._ilog.log("button_click", "btn_edit_meta_query", value=self._current_query)
        if self._meta_edit_popup is None or not self._meta_edit_popup.isVisible():
            self._meta_edit_popup = _MetadataEditPopup("Queries", self._current_query, self)
            self._meta_edit_popup.saved.connect(self._on_metadata_edited)
            self._meta_edit_popup.destroyed.connect(lambda: setattr(self, '_meta_edit_popup', None))
        self._meta_edit_popup.show()
        self._meta_edit_popup.raise_()

    def _on_promote_query(self):
        """Promote the current query to a new gallery identity."""
        from PySide6.QtWidgets import QMessageBox
        from src.data.promote_query import promote_query_to_gallery
        
        qid = self._current_query
        if not qid:
            return
        
        self._ilog.log("button_click", "btn_promote_query", value=qid)
        
        # Open promote dialog
        dlg = _PromoteQueryDialog(qid, self)
        if dlg.exec() != QDialog.Accepted:
            return
        
        gallery_id = dlg.get_gallery_id()
        copy_metadata = dlg.should_copy_metadata()
        
        self._ilog.log("promote_query", "confirmed", value=qid,
                      context={"gallery_id": gallery_id, "copy_metadata": copy_metadata})
        
        # Perform promotion
        report = promote_query_to_gallery(
            query_id=qid,
            new_gallery_id=gallery_id,
            copy_metadata=copy_metadata
        )
        
        # Show result
        if report.success:
            QMessageBox.information(
                self,
                "Promote to Gallery",
                f"Successfully promoted Query to Gallery.\n\n"
                f"Query: {report.query_id}\n"
                f"New Gallery ID: {report.gallery_id}\n"
                f"Encounter folders copied: {report.num_encounter_dirs}\n"
                f"Metadata copied: {'Yes' if report.metadata_copied else 'No'}\n\n"
                "The Query is now hidden from matching."
            )
        else:
            error_text = "\n".join(f"- {e}" for e in report.errors) if report.errors else "Unknown error"
            QMessageBox.warning(
                self,
                "Promote to Gallery",
                f"Promotion completed with issues.\n\n"
                f"Encounter folders copied: {report.num_encounter_dirs}\n"
                f"Errors:\n{error_text}"
            )
        
        # Refresh query list (promoted query is now silent)
        self._refresh_query_ids()
        
        # Rebuild engine to include the new gallery identity
        self.engine.rebuild()
        
        # Refresh results if we still have a query selected
        if self._current_query:
            self._refresh_results()

    def _edit_gallery_metadata(self, gallery_id: str):
        """Open edit metadata popup for a gallery ID."""
        self._ilog.log("button_click", "btn_edit_meta_gallery", value=gallery_id)
        popup = _MetadataEditPopup("Gallery", gallery_id, self)
        popup.saved.connect(self._on_metadata_edited)
        popup.destroyed.connect(lambda: None)
        popup.show()
        popup.raise_()

    def _on_metadata_edited(self):
        """Refresh rankings after metadata is edited."""
        self._refresh_results()

    def _open_exclude_dialog(self):
        self._ilog.log("dialog_open", "exclude_dialog")
        # Gather gallery IDs and available location values
        gallery_ids = sorted(list(self.engine._gallery_rows_by_id.keys()))
        locs: List[str] = []
        for r in self.engine._gallery_rows_by_id.values():
            locs.append((r.get("location", "") or "").strip())
        dlg = _ExcludeDialog(
            gallery_ids, locs, 
            set(self._excluded_ids), set(self._excluded_locations),
            set(self._include_history_locations), self
        )
        if dlg.exec():
            ids, locs_set, hist_locs = dlg.get_results()
            self._excluded_ids = set(ids)
            self._excluded_locations = set(locs_set)
            self._include_history_locations = set(hist_locs)
            # Update label to show all filter counts
            filter_parts = []
            if self._excluded_ids:
                filter_parts.append(f"{len(self._excluded_ids)} IDs")
            if self._excluded_locations:
                filter_parts.append(f"{len(self._excluded_locations)} loc")
            if self._include_history_locations:
                filter_parts.append(f"{len(self._include_history_locations)} hist")
            self.lbl_excluded.setText(f"Filters: {', '.join(filter_parts) if filter_parts else 'none'}")
            self._ilog.log("dialog_close", "exclude_dialog", value="accepted",
                          context={"excluded_ids": len(ids), "excluded_locs": len(locs_set), 
                                   "include_hist_locs": len(hist_locs)})
            self._refresh_results()
        else:
            self._ilog.log("dialog_close", "exclude_dialog", value="cancelled")

    def _open_config_dialog(self):
        """Open the fields configuration dialog."""
        self._ilog.log("dialog_open", "config_dialog")
        dlg = FieldsConfigDialog(self)
        dlg.configSaved.connect(self._on_config_saved)
        dlg.exec()

    def _on_config_saved(self):
        """Handle config saved - rebuild engine and refresh UI."""
        # Sync field checkboxes with the new config
        self._sync_checkboxes_from_config()
        # Reset preset combo to "Config" to indicate current state
        idx = self.cmb_preset.findText(PRESET_CONFIG)
        if idx >= 0:
            self.cmb_preset.blockSignals(True)
            self.cmb_preset.setCurrentIndex(idx)
            self.cmb_preset.blockSignals(False)
        # Rebuild engine with new configuration
        self.engine.rebuild()
        self._dates_initialized = False
        self._refresh_query_ids()
        self._refresh_results()

    def _sync_checkboxes_from_config(self):
        """Update field checkboxes to match the enabled state in fields_config.yaml."""
        from src.data.fields_config import get_fields_config
        config = get_fields_config(reload=True)
        enabled_fields = config.enabled_field_names()
        
        for field_name, chk in self.chk_by_name.items():
            chk.setChecked(field_name in enabled_fields)

    def _on_query_changed(self):
        qid = self.cmb_query.currentText()
        self._current_query = qid or ""
        self.lbl_query_id.setText(qid or "—")
        
        # Update suggested match display
        self._update_suggested_match_display(qid)
        
        # Log query selection
        self._ilog.log("combo_change", "cmb_query", value=qid)
        files = list_image_files("Queries", qid) if qid else []
        files = reorder_files_with_best("Queries", qid, files) if qid else files
        self.query_strip.set_files(files)
        # Query changed: make sure gallery cards adhere to this query view's height
        QTimer.singleShot(0, self._sync_card_min_height_from_query)

        # reset offsets on query change (quietly)
        for w in getattr(self, "_offset_widgets", {}).values():
            try:
                w.blockSignals(True); w.setValue(0); w.blockSignals(False)
            except Exception:
                pass

        # load persisted pins for this query
        self._pinned = self._load_pins(self._current_query) if self._current_query else []
        self.lbl_pinned.setText(f"Pinned: {len(self._pinned)}")
        
        # Update image quality panel for this query
        if hasattr(self, 'query_quality_panel'):
            self.query_quality_panel.load_for_id("Queries", qid)
        
        self._refresh_results()

    def _on_prev_query_clicked(self) -> None:
        """Navigate to the previous query in the combo box list."""
        self._ilog.log("button_click", "btn_prev_query", value="clicked")
        current_idx = self.cmb_query.currentIndex()
        if current_idx > 0:
            self.cmb_query.setCurrentIndex(current_idx - 1)

    def _on_next_query_clicked(self) -> None:
        """Navigate to the next query in the combo box list."""
        self._ilog.log("button_click", "btn_next_query", value="clicked")
        current_idx = self.cmb_query.currentIndex()
        max_idx = self.cmb_query.count() - 1
        if current_idx < max_idx:
            self.cmb_query.setCurrentIndex(current_idx + 1)

    def _on_set_best_query(self):
        qid = self._current_query
        if not qid or not self.query_strip.files:
            return
        idx = max(0, min(self.query_strip.idx, len(self.query_strip.files) - 1))
        self._ilog.log("button_click", "btn_best_query", value=qid,
                      context={"image_idx": idx})
        save_best_for_id("Queries", qid, self.query_strip.files[idx])
        files = reorder_files_with_best("Queries", qid, list(self.query_strip.files))
        self.query_strip.set_files(files)

    def _on_query_quality_saved(self, target: str, id_value: str) -> None:
        """Handle image quality saved for the query.
        
        Option A: Suppress the file watcher rebuild to preserve view state.
        The file watcher would trigger a full rebuild which resets both query
        and gallery views - we don't need that for internal saves.
        """
        # Set suppress flag - the file watcher event may already be queued
        self._suppress_csv_watch = True
        # Clear the flag after the debounce period (200ms) plus margin
        QTimer.singleShot(300, self._clear_csv_watch_suppress)
        
        # Update the quality indicator symbols for this specific query in the combo
        idx = self.cmb_query.findText(id_value)
        if idx >= 0:
            quality_data = get_quality_for_ids("Queries", [id_value])
            madreporite, anus, posture = quality_data.get(id_value, (-1.0, -1.0, -1.0))
            model = self.cmb_query.model()
            model_idx = model.index(idx, 0)
            model.setData(model_idx, madreporite, QUALITY_MADREPORITE_ROLE)
            model.setData(model_idx, anus, QUALITY_ANUS_ROLE)
            model.setData(model_idx, posture, QUALITY_POSTURE_ROLE)

    def _clear_csv_watch_suppress(self) -> None:
        """Clear the CSV watch suppress flag after internal save completes."""
        self._suppress_csv_watch = False

    def _auto_excluded_for_same_day(self, qid: str) -> Set[str]:
        """
        Return gallery IDs that should be excluded because they already have a YES verdict
        with ANY query observed on the same day as 'qid'.
        """
        try:
            data = load_match_matrix()
            q_date = data.last_obs_by_query.get(qid)
            if not q_date:
                return set()
            out: Set[str] = set()
            for (q2, gid), verdict in data.verdict_by_pair.items():
                if verdict == "yes" and data.last_obs_by_query.get(q2) == q_date:
                    out.add(gid)
            return out
        except Exception:
            return set()

    def _on_filters_toggled(self, expanded: bool):
        """When the filters section collapses, reclaim the vertical space in the splitter."""
        self._ilog.log("checkbox_toggle", "filters_section", value=str(expanded))
        sizes = self.vsplit_all.sizes()
        total = sum(sizes) if sizes else 0
        if expanded:
            top = min(220, max(140, total // 4)) if total else 180
            self.vsplit_all.setSizes([top, max(1, total - top) if total else 800])
        else:
            if total:
                self.vsplit_all.setSizes([0, total])

    def _refresh_results(self):
        qid = self._current_query
        if not qid:
            self._set_cards([])
            return

        fields = self._selected_fields()
        use_visual = self.chk_visual.isChecked() and self._visual_available
        use_verification = self.chk_verification.isChecked() and self._verification_available
        
        if not fields and not use_visual and not use_verification:
            # Do NOT clear the gallery when no fields are selected.
            # Keep what's currently visible and just re-apply sizing.
            self._sync_card_min_height_from_query()
            self._fit_cards_to_viewport()
            return

        # Load weights configuration (reload=True to pick up any changes)
        config = get_fields_config(reload=True)
        equalize = config.equalize_weights
        
        # Build weights dict from config (only needed if not equalizing)
        weights = None
        if not equalize:
            weights = {f: config.get_weight(f) for f in fields}

        # Get metadata-based ranking results
        if fields:
            results = self.engine.rank(
                qid,
                include_fields=fields,
                equalize_weights=equalize,
                weights=weights,
                top_k=int(self.spin_topk.value()) * 2 if use_visual else int(self.spin_topk.value()),
                numeric_offsets=self._collect_numeric_offsets(),
            )
        else:
            results = []

        # Get visual scores if enabled
        visual_scores: Dict[str, float] = {}
        if use_visual:
            visual_scores = self._get_visual_scores(qid)

        # Get verification scores if enabled
        verification_scores: Dict[str, float] = {}
        if use_verification:
            verification_scores = self._get_verification_scores(qid)

        # Fuse scores if both are available
        fusion_alpha = self.slider_fusion.value() / 100.0 if use_visual else 0.0
        verif_alpha = self.slider_verif_fusion.value() / 100.0 if use_verification else 0.0
        
        # Build a combined result set with visual and/or verification fusion
        from dataclasses import dataclass
        
        @dataclass
        class FusedResult:
            gallery_id: str
            score: float
            k_contrib: float
            field_breakdown: Dict[str, float]
            verification_score: Optional[float] = None

        need_fusion = (use_visual and visual_scores) or (use_verification and verification_scores)
        
        if need_fusion:
            # Collect metadata scores
            metadata_scores: Dict[str, float] = {}
            result_by_id: Dict[str, object] = {}
            for it in results:
                metadata_scores[it.gallery_id] = it.score
                result_by_id[it.gallery_id] = it
            
            # All gallery IDs (union of all sources)
            all_gallery_ids = set(metadata_scores.keys())
            if visual_scores:
                all_gallery_ids |= set(visual_scores.keys())
            if verification_scores:
                all_gallery_ids |= set(verification_scores.keys())
            
            # Compute fused scores in two stages:
            # Stage 1: metadata + visual fusion
            # Stage 2: (metadata+visual) + verification fusion
            fused_scores: Dict[str, float] = {}
            for gid in all_gallery_ids:
                m_score = metadata_scores.get(gid, 0.0)
                
                # Stage 1: metadata + visual
                if use_visual and visual_scores:
                    v_score = visual_scores.get(gid, 0.0)
                    base_score = fusion_alpha * v_score + (1 - fusion_alpha) * m_score
                else:
                    base_score = m_score
                
                # Stage 2: base + verification
                if use_verification and verification_scores:
                    verif_score = verification_scores.get(gid, 0.0)
                    final_score = verif_alpha * verif_score + (1 - verif_alpha) * base_score
                else:
                    final_score = base_score
                
                fused_scores[gid] = final_score
            
            # Sort by fused score and limit to top_k
            sorted_gids = sorted(fused_scores.keys(), key=lambda g: -fused_scores[g])
            sorted_gids = sorted_gids[:int(self.spin_topk.value())]
            
            # Rebuild result items with updated scores
            fused_results = []
            for gid in sorted_gids:
                # Build field breakdown
                if gid in result_by_id:
                    orig = result_by_id[gid]
                    breakdown = dict(orig.field_breakdown)
                    k_contrib = orig.k_contrib
                else:
                    breakdown = {}
                    k_contrib = 0
                
                # Add visual score to breakdown if present
                if gid in visual_scores:
                    breakdown["visual"] = visual_scores[gid]
                
                # Add verification score to breakdown if present
                if gid in verification_scores:
                    breakdown["P(same)"] = verification_scores[gid]
                
                fused_results.append(FusedResult(
                    gallery_id=gid,
                    score=fused_scores[gid],
                    k_contrib=k_contrib,
                    field_breakdown=breakdown,
                    verification_score=verification_scores.get(gid) if verification_scores else None
                ))
            
            results = fused_results
        else:
            # No fusion needed - wrap results in FusedResult for consistent interface
            wrapped_results = []
            for it in results:
                wrapped_results.append(FusedResult(
                    gallery_id=it.gallery_id,
                    score=it.score,
                    k_contrib=it.k_contrib,
                    field_breakdown=dict(it.field_breakdown),
                    verification_score=None
                ))
            results = wrapped_results

        # ---- Manual exclusions still EXCLUDE; auto same-day YES now DEMOTES ----
        # EXCEPT: if the current query has its own YES match, that gallery is PROMOTED to top
        demote_ids = set(self._auto_excluded_for_same_day(qid))  # keep logic; change handling
        excluded_ids = set(self._excluded_ids)  # manual-only
        excluded_locs = set(self._excluded_locations)
        include_hist_locs = set(self._include_history_locations)  # location history filter
        
        # Pre-compute which gallery IDs pass the location history filter
        hist_allowed_ids: Set[str] | None = None
        if include_hist_locs:
            try:
                from src.data.location_history import find_galleries_with_location_history
                all_gids = [it.gallery_id for it in results]
                hist_allowed_ids = find_galleries_with_location_history(all_gids, include_hist_locs)
            except Exception:
                hist_allowed_ids = None  # On error, don't filter

        # Find the gallery that has a YES match specifically for THIS query (promote it)
        promote_gid: str | None = None
        try:
            latest_for_qid = load_latest_map_for_query(qid)
            for gid, row in latest_for_qid.items():
                if (row.get("verdict", "") or "").strip().lower() == "yes":
                    promote_gid = gid
                    break  # take the first YES match found
        except Exception:
            pass

        # Don't demote the gallery that is the query's own match - it will be promoted instead
        if promote_gid:
            demote_ids.discard(promote_gid)

        kept, demoted, promoted = [], [], []
        for it in results:
            # Manual ID exclusion => drop
            if it.gallery_id in excluded_ids:
                continue
            # Manual location exclusion => drop
            g_row = self.engine._gallery_rows_by_id.get(it.gallery_id, {})  # safe default
            loc = (g_row.get("location", "") or "").strip()
            if loc and loc in excluded_locs:
                continue
            # Location history filter (include mode) => drop if not in allowed set
            if hist_allowed_ids is not None and it.gallery_id not in hist_allowed_ids:
                continue
            # Query's own YES match => promote to top
            if promote_gid and it.gallery_id == promote_gid:
                promoted.append(it)
            # Auto same-day YES (from other queries) => demote to the end
            elif it.gallery_id in demote_ids:
                demoted.append(it)
            else:
                kept.append(it)

        ordered = promoted + kept + demoted  # promoted first, then normal, then demoted

        # ---- Build cards (with robust constructor call & rich tooltips) ----
        cards: List[LineupCard] = []
        q_row = self.engine._queries_rows_by_id.get(qid, {})
        
        # Determine if we should roll gallery images to closest match
        use_roll_to_closest = (
            use_visual
            and self.cmb_visual_mode.currentData() == "image"
            and self.chk_roll_to_closest.isChecked()
        )
        # Only apply roll-to-closest for top N results (DL is noise after that)
        roll_limit = self.spin_roll_limit.value() if use_roll_to_closest else 0
        
        for rank, it in enumerate(ordered):
            g_row = self.engine._gallery_rows_by_id.get(it.gallery_id, {})
            tooltips = {
                f: self._tooltip_for_field(
                    f,
                    q_row.get(f, ""),
                    g_row.get(f, ""),
                    it.field_breakdown.get(f, 0.0),
                )
                for f in it.field_breakdown.keys()
            }
            
            # Get closest image info for this gallery identity (if applicable)
            # Only apply to top N results - DL similarity is noise after that
            closest_image_info = None
            if use_roll_to_closest and rank < roll_limit:
                closest_image_info = self._closest_gallery_image.get(it.gallery_id)

            # Get verification score if available (FusedResult has it, others don't)
            verif_score = getattr(it, 'verification_score', None)
            
            # Try the new signature first (with query_id for decision UI)
            card = None
            try:
                card = LineupCard(
                    it.gallery_id,
                    it.score,
                    it.k_contrib,
                    it.field_breakdown,
                    field_tooltips=tooltips,
                    query_id=self._current_query,  # enables decision combo + save
                    closest_image_info=closest_image_info,  # (idx, path) tuple for O(1) with fallback
                    verification_score=verif_score,  # P(same) from verification model
                )
            except TypeError:
                try:
                    badges = sorted(list(it.field_breakdown.keys()))
                    card = LineupCard(it.gallery_id, it.score, it.k_contrib, badges)
                except TypeError:
                    card = LineupCard(it.gallery_id, it.score, it.k_contrib, it.field_breakdown)

            if hasattr(card, "btn_pin"):
                card.btn_pin.clicked.connect(lambda _=None, gid=it.gallery_id: self._toggle_pin(gid))
            if hasattr(card, "metadataRequested"):
                card.metadataRequested.connect(self._show_gallery_metadata)
            if hasattr(card, "editMetadataRequested"):
                card.editMetadataRequested.connect(self._edit_gallery_metadata)
            if hasattr(card, "decisionSaved"):
                card.decisionSaved.connect(self._on_card_decision_saved)
            cards.append(card)

        self._set_cards(cards)
        self._fit_cards_to_viewport()

    def _set_cards(self, cards: List[LineupCard]):
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()  # Properly destroy old cards instead of orphaning them
        self._cards = cards
        for c in cards:
            c.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
            self.cards_layout.addWidget(c)
        self.cards_layout.addStretch(1)

        # Already present:
        self._sync_card_min_height_from_query()  # keep

        # NEW: ensure min width is also synced immediately
        self._sync_card_min_width_from_query()

        # Update gallery search combo with current cards
        self._update_gallery_search_combo()

    def _fit_cards_to_viewport(self):
        vh = self.scroll.viewport().height()
        if vh <= 0:
            return
        for c in self._cards:
            if hasattr(c, "fit_to_viewport_height"):
                try:
                    c.fit_to_viewport_height(vh)
                except Exception:
                    pass
            elif hasattr(c, "set_strip_height"):
                try:
                    c.set_strip_height(vh - 120)
                except Exception:
                    pass

    def _update_gallery_search_combo(self):
        """Update the gallery search combo box with current card IDs."""
        if not hasattr(self, 'cmb_gallery_search') or self.cmb_gallery_search is None:
            return
        self.cmb_gallery_search.blockSignals(True)
        self.cmb_gallery_search.clear()
        for card in self._cards:
            self.cmb_gallery_search.addItem(card.gallery_id)
        self.cmb_gallery_search.setCurrentIndex(-1)  # No selection initially
        self.cmb_gallery_search.blockSignals(False)
        self.cmb_gallery_search.setEnabled(len(self._cards) > 0)

    def _on_gallery_search_changed(self, index: int):
        """Handle gallery search combo selection - scroll to the selected card."""
        if not hasattr(self, '_cards') or index < 0 or index >= len(self._cards):
            return
        gid = self._cards[index].gallery_id if index < len(self._cards) else ""
        self._ilog.log("combo_change", "cmb_gallery_search", value=gid)
        self._scroll_to_gallery_card(index)

    def _toggle_pin(self, gid: str):
        was_pinned = gid in self._pinned
        if was_pinned:
            self._pinned.remove(gid)
        else:
            self._pinned.append(gid)
        self.lbl_pinned.setText(f"Pinned: {len(self._pinned)}")
        self._save_pins()
        # Log pin action
        self._ilog.log("button_click", "btn_pin", value="unpinned" if was_pinned else "pinned",
                      context={"gallery_id": gid, "query_id": self._current_query})
        # Update query state color (pins affect the PINNED state)
        self._update_single_query_state(self._current_query)

    def _on_card_decision_saved(self, query_id: str, gallery_id: str, verdict: str):
        """Handle decision saved from a LineupCard - update query state color."""
        self._ilog.log("decision_save", "lineup_card", value=verdict,
                      context={"query_id": query_id, "gallery_id": gallery_id})
        self._update_single_query_state(query_id)
    
    def _update_single_query_state(self, qid: str) -> None:
        """Update the state color for a single query in the combo box."""
        idx = self.cmb_query.findText(qid)
        if idx >= 0:
            state = get_query_state(qid)
            model = self.cmb_query.model()
            model.setData(model.index(idx, 0), state, QUERY_STATE_ROLE)

    def _on_metadata_csv_changed(self, _path: str):
        # Option A: Skip rebuild if this was triggered by an internal save
        if self._suppress_csv_watch:
            return
        
        # Debounce: in case of multiple appends in quick succession
        if getattr(self, "_meta_debounce", None) is None:
            self._meta_debounce = QTimer(self)
            self._meta_debounce.setSingleShot(True)
            self._meta_debounce.timeout.connect(self._rebuild_after_metadata_change)
        self._meta_debounce.start(200)

    def _rebuild_after_metadata_change(self):
        try:
            # Same semantics as pressing “Rebuild index” followed by refreshing the UI
            self.engine.rebuild()
            self._dates_initialized = False
            # Re-populate IDs (preserve selection if possible) and refresh results
            self._refresh_query_ids()
            self._refresh_results()
        except Exception:
            pass

    # ---- offsets helpers ----
    def _collect_numeric_offsets(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, w in getattr(self, "_offset_widgets", {}).items():
            try:
                v = float(w.value())
                if abs(v) > 0.0:
                    out[k] = v
            except Exception:
                pass
        return out

    def _reset_offsets(self):
        self._ilog.log("button_click", "btn_reset_offsets", value="clicked")
        for w in getattr(self, "_offset_widgets", {}).values():
            try:
                w.blockSignals(True)
                w.setValue(0)
            finally:
                try:
                    w.blockSignals(False)
                except Exception:
                    pass
        self._refresh_results()

    def _on_offsets_changed(self, *_):
        # Log current offset values (throttled via interaction logger)
        offsets = self._collect_numeric_offsets()
        if offsets:
            self._ilog.log("offset_change", "offset_widgets", value=str(offsets))
        self._refresh_results()

    # ---------- Min-height sync between Query viewer and Gallery cards ----------
    def _query_top_height(self) -> int:
        """
        Current height of the top half (image area) of the Query vertical splitter.
        Falls back to the Query view's viewport height if needed.
        """
        try:
            sizes = self.query_vsplit.sizes()
            if sizes and sizes[0] > 0:
                return int(sizes[0])
        except Exception:
            pass
        try:
            return int(self.query_strip.view.viewport().height())
        except Exception:
            return 240  # safe floor

    def _sync_card_min_height_from_query(self) -> None:
        """
        Push the current Query image area height as the *minimum* image height
        for all gallery LineupCards (they may grow taller if badges/controls need it).
        """
        h = max(80, int(self._query_top_height()))
        for c in getattr(self, "_cards", []):
            if hasattr(c, "set_strip_height"):
                try:
                    c.set_strip_height(h)
                except Exception:
                    pass

    def _on_query_pixmap_resized(self, *_):
        # When the underlying Query pixmap changes size, re-sync + refit cards.
        self._sync_card_min_height_from_query()
        self._fit_cards_to_viewport()

    # ---------------- Gallery navigation (Next/Prev result buttons) ----------------
    def add_gallery_nav_toolbar(self) -> None:
        """
        Install a tiny navigation toolbar (Prev / Next result) under the gallery
        scroll area on the right side of the First‑order tab.

        This is intentionally 'surgical':
        - We DO NOT change how cards are created or sized.
        - We simply wrap the existing QScrollArea in a lightweight QWidget
          with a vertical layout and add two buttons below it.
        - The original scroll bar remains fully usable.
        """
        # Avoid double‑installation if called more than once.
        if getattr(self, "_gallery_nav_installed", False):
            return

        scroll = getattr(self, "scroll", None)
        hsplit = getattr(self, "hsplit", None)
        if scroll is None or hsplit is None:
            # Something is unexpected; fail silently rather than breaking the tab.
            return

        # Wrapper that will host the existing scroll area + the new buttons.
        wrapper = _QW()
        vlay = _QVL(wrapper)
        vlay.setContentsMargins(0, 0, 0, 0)
        vlay.setSpacing(4)

        # Insert the wrapper into the splitter at the same index where the
        # gallery scroll area currently lives.
        idx = hsplit.indexOf(scroll)
        if idx < 0:
            # Fallback: if the scroll area is not in the splitter for some reason,
            # just add the wrapper at the end.
            hsplit.addWidget(wrapper)
        else:
            # Insert wrapper before the scroll, then reparent the scroll into it.
            hsplit.insertWidget(idx, wrapper)

        # Reparent the existing scroll area into the wrapper.
        vlay.addWidget(scroll, 1)

        # --- Buttons row ---
        row = QHBoxLayout()
        row.setContentsMargins(4, 0, 4, 0)
        row.setSpacing(6)

        # Gallery search combo box
        row.addWidget(QLabel("Jump to:"))
        self.cmb_gallery_search = QComboBox()
        self.cmb_gallery_search.setMinimumWidth(140)
        self.cmb_gallery_search.setEditable(True)
        self.cmb_gallery_search.setInsertPolicy(QComboBox.NoInsert)
        self.cmb_gallery_search.setPlaceholderText("Gallery ID…")
        self.cmb_gallery_search.completer().setFilterMode(Qt.MatchContains)
        self.cmb_gallery_search.completer().setCompletionMode(QCompleter.PopupCompletion)
        self.cmb_gallery_search.currentIndexChanged.connect(self._on_gallery_search_changed)
        row.addWidget(self.cmb_gallery_search)

        row.addStretch(1)

        self.btn_gallery_prev = QPushButton("◀ Prev result")
        self.btn_gallery_next = QPushButton("Next result ▶")

        self.btn_gallery_prev.setToolTip(
            "Scroll to the previous gallery identity and center it in view."
        )
        self.btn_gallery_next.setToolTip(
            "Scroll to the next gallery identity and center it in view."
        )

        row.addWidget(self.btn_gallery_prev)
        row.addWidget(self.btn_gallery_next)
        vlay.addLayout(row)

        # Wire up behavior.
        self.btn_gallery_prev.clicked.connect(self._on_gallery_prev_clicked)
        self.btn_gallery_next.clicked.connect(self._on_gallery_next_clicked)

        # Populate the gallery search combo with current cards
        self._update_gallery_search_combo()

        self._gallery_nav_installed = True

    # ---- coordinate helpers ----
    def _current_gallery_card_index(self) -> int:
        """
        Return the index of the card whose horizontal center is closest to the
        current viewport center. If there are no cards, return -1.

        All coordinates are in the cards_container / scroll‑contents space, so
        they are directly comparable to the scroll bar value.
        """
        if not self._cards or not hasattr(self, "scroll"):
            return -1

        viewport = self.scroll.viewport()
        hbar = self.scroll.horizontalScrollBar()
        if viewport is None or hbar is None:
            return -1

        view_center_x = float(hbar.value()) + float(viewport.width()) / 2.0

        best_idx = -1
        best_dist = float("inf")

        for i, card in enumerate(self._cards):
            rect = card.geometry()
            if rect.isNull():
                continue
            center_x = float(rect.x()) + float(rect.width()) / 2.0
            d = abs(center_x - view_center_x)
            if d < best_dist:
                best_dist = d
                best_idx = i

        return best_idx

    def _scroll_to_gallery_card(self, idx: int) -> None:
        """
        Scroll horizontally so that card `idx` is centered in the gallery
        viewport as much as the scroll range allows.
        """
        if not (0 <= idx < len(self._cards)):
            return
        if not hasattr(self, "scroll"):
            return

        card = self._cards[idx]
        rect = card.geometry()
        if rect.isNull():
            return

        viewport = self.scroll.viewport()
        hbar = self.scroll.horizontalScrollBar()
        if viewport is None or hbar is None:
            return

        target_center_x = float(rect.x()) + float(rect.width()) / 2.0
        new_value = target_center_x - float(viewport.width()) / 2.0

        # Clamp to valid scroll range.
        new_int = int(round(new_value))
        new_int = max(hbar.minimum(), min(hbar.maximum(), new_int))
        hbar.setValue(new_int)

    # ---- button handlers ----
    def _on_gallery_prev_clicked(self) -> None:
        """
        Jump to the previous gallery card (if any) and center it.
        """
        self._ilog.log("button_click", "btn_gallery_prev", value="clicked")
        if not self._cards:
            return
        cur = self._current_gallery_card_index()
        if cur < 0:
            # Nothing visible yet; treat as first card.
            target = 0
        else:
            target = max(0, cur - 1)
        self._scroll_to_gallery_card(target)

    def _on_gallery_next_clicked(self) -> None:
        """
        Jump to the next gallery card (if any) and center it.
        """
        self._ilog.log("button_click", "btn_gallery_next", value="clicked")
        if not self._cards:
            return
        cur = self._current_gallery_card_index()
        if cur < 0:
            # Nothing visible yet; treat as first card.
            target = 0
        else:
            target = min(len(self._cards) - 1, cur + 1)
        self._scroll_to_gallery_card(target)



    # ---- field tooltips with raw values ----
    def _tooltip_for_field(self, field: str, q_val: str, g_val: str, s: float) -> str:
        def _fmt(x: str) -> str:
            return (x or "").strip()

        # Integer numeric fields (arm counts)
        if field in ("num_total_arms", "num_apparent_arms"):
            try:
                q = int(float(q_val)) if q_val else None
                g = int(float(g_val)) if g_val else None
            except Exception:
                q = g = None
            if q is not None and g is not None:
                d = g - q
                return f"{field}: Q={q}, G={g}, Δ={d}  |  s={s:.3f}"
            return f"{field}: Q={_fmt(q_val)}, G={_fmt(g_val)}  |  s={s:.3f}"

        # Float numeric field (size)
        if field == "tip_to_tip_size_cm":
            try:
                q = float(q_val) if q_val else None
                g = float(g_val) if g_val else None
            except Exception:
                q = g = None
            if q is not None and g is not None:
                d = g - q
                return f"{field}: Q={q:.1f}cm, G={g:.1f}cm, Δ={d:+.1f}cm  |  s={s:.3f}"
            return f"{field}: Q={_fmt(q_val)}cm, G={_fmt(g_val)}cm  |  s={s:.3f}"

        # Ordinal fields (display as numeric with delta)
        if field in ORDINAL_FIELDS:
            try:
                q = float(q_val) if q_val else None
                g = float(g_val) if g_val else None
            except Exception:
                q = g = None
            if q is not None and g is not None:
                d = g - q
                return f"{field}: Q={q:g}, G={g:g}, Δ={d:+g}  |  s={s:.3f}"
            return f"{field}: Q={_fmt(q_val)}, G={_fmt(g_val)}  |  s={s:.3f}"

        # Color categorical fields
        if field in COLOR_FIELDS:
            return f"{field}: Q={_fmt(q_val)}, G={_fmt(g_val)}  |  s={s:.3f}"

        # Short arm code (set comparison) - human-readable format
        if field == "short_arm_code":
            import re
            
            def parse_arm_codes(raw: str) -> Dict[str, List[int]]:
                """Parse short arm codes into {severity: [positions]}."""
                result = {"tiny": [], "small": [], "short": []}
                for part in raw.replace(",", " ").split():
                    part = part.strip()
                    if not part:
                        continue
                    # Match patterns like "tiny(2)", "small(10)", "short(3)"
                    m = re.match(r"(tiny|small|short)\((\d+)\)", part, re.IGNORECASE)
                    if m:
                        sev = m.group(1).lower()
                        pos = int(m.group(2))
                        if sev in result:
                            result[sev].append(pos)
                return {k: sorted(v) for k, v in result.items()}
            
            def format_codes(codes: Dict[str, List[int]]) -> str:
                """Format parsed codes for display."""
                parts = []
                for sev in ["tiny", "small", "short"]:
                    if codes[sev]:
                        positions = ", ".join(str(p) for p in codes[sev])
                        parts.append(f"{sev}: arms {positions}")
                return "; ".join(parts) if parts else "none"
            
            q_codes = parse_arm_codes(_fmt(q_val))
            g_codes = parse_arm_codes(_fmt(g_val))
            
            # Find shared and unique positions per severity
            shared_parts = []
            q_only_parts = []
            g_only_parts = []
            
            for sev in ["tiny", "small", "short"]:
                q_set = set(q_codes[sev])
                g_set = set(g_codes[sev])
                shared = sorted(q_set & g_set)
                q_only = sorted(q_set - g_set)
                g_only = sorted(g_set - q_set)
                
                if shared:
                    shared_parts.append(f"{sev} @ {', '.join(str(p) for p in shared)}")
                if q_only:
                    q_only_parts.append(f"{sev} @ {', '.join(str(p) for p in q_only)}")
                if g_only:
                    g_only_parts.append(f"{sev} @ {', '.join(str(p) for p in g_only)}")
            
            lines = [
                f"Short Arm Codes  (similarity: {s:.2f})",
                f"",
                f"Query:   {format_codes(q_codes)}",
                f"Gallery: {format_codes(g_codes)}",
                f"",
                f"Shared:      {'; '.join(shared_parts) if shared_parts else 'none'}",
                f"Query only:  {'; '.join(q_only_parts) if q_only_parts else 'none'}",
                f"Gallery only: {'; '.join(g_only_parts) if g_only_parts else 'none'}",
            ]
            return "\n".join(lines)

        # Text fields: show short snippets
        def _snip(x: str, n: int = 160) -> str:
            x = (x or "").strip().replace("\n", " ")
            return (x[:n] + "…") if len(x) > n else x

        return f"{field}: Q=\"{_snip(q_val)}\", G=\"{_snip(g_val)}\"  |  s={s:.3f}"

    # ---- date filter helpers ----
    def _ensure_date_controls_bounds(self, last_obs: Dict[str, Optional[_date]]) -> None:
        """Initialize date pickers to min/max coverage once per session (or after rebuild)."""
        if self._dates_initialized:
            return
        dates = [d for d in last_obs.values() if d is not None]
        if not dates:
            # no dates detected; still set to today so widgets are valid
            today = QDate.currentDate()
            self.date_from.setDate(today)
            self.date_to.setDate(today)
        else:
            d_min = min(dates)
            d_max = max(dates)
            self.date_from.setDate(QDate(d_min.year, d_min.month, d_min.day))
            self.date_to.setDate(QDate(d_max.year, d_max.month, d_max.day))
        self._dates_initialized = True
        self._update_date_widgets_enabled()

    def _on_date_filter_changed(self, *_):
        self._ilog.log("date_change", "date_filter", value=str(self.chk_date.isChecked()),
                      context={"from": self.date_from.date().toString("yyyy-MM-dd"),
                              "to": self.date_to.date().toString("yyyy-MM-dd"),
                              "include_nodate": self.chk_include_nodate.isChecked()})
        self._update_date_widgets_enabled()
        self._refresh_query_ids()

    def _update_date_widgets_enabled(self):
        on = self.chk_date.isChecked()
        for w in (self.date_from, self.date_to, self.chk_include_nodate):
            w.setEnabled(on)

    def _apply_query_date_filter(self, ids: List[str], last_obs: Dict[str, Optional[_date]]) -> List[str]:
        if not self.chk_date.isChecked():
            return ids
        include_nodate = self.chk_include_nodate.isChecked()

        def _qdate_to_py(qd: QDate) -> Optional[_date]:
            if not qd or not qd.isValid():
                return None
            return _date(qd.year(), qd.month(), qd.day())

        d_from = _qdate_to_py(self.date_from.date())
        d_to = _qdate_to_py(self.date_to.date())
        if not (d_from and d_to):
            return ids

        out: List[str] = []
        for qid in ids:
            d = last_obs.get(qid)
            if d is None:
                if include_nodate:
                    out.append(qid)
                continue
            if d_from <= d <= d_to:
                out.append(qid)
        return out

    # ---- visual ranking helpers ----
    def _init_visual_controls(self):
        """Initialize visual ranking controls based on DL availability."""
        try:
            from src.dl import DL_AVAILABLE
            from src.dl.registry import DLRegistry
            
            if not DL_AVAILABLE:
                self._visual_available = False
                self.chk_visual.setEnabled(False)
                self.chk_visual.setToolTip("PyTorch not installed. Install with: pip install -r requirements-dl.txt")
                self.cmb_model.setEnabled(False)
                self.cmb_visual_mode.setEnabled(False)
                self.btn_refresh_visual.setEnabled(False)
                self.chk_roll_to_closest.setEnabled(False)
                self.spin_roll_limit.setEnabled(False)
                self.slider_fusion.setEnabled(False)
                self.lbl_fusion.setEnabled(False)
                return
            
            registry = DLRegistry.load()
            precomputed = registry.get_precomputed_models()
            
            if not precomputed:
                self._visual_available = False
                self.chk_visual.setEnabled(False)
                self.chk_visual.setToolTip("Run precomputation in Deep Learning tab to enable")
                self.cmb_model.setEnabled(False)
                self.cmb_visual_mode.setEnabled(False)
                self.btn_refresh_visual.setEnabled(False)
                self.chk_roll_to_closest.setEnabled(False)
                self.spin_roll_limit.setEnabled(False)
                self.slider_fusion.setEnabled(False)
                self.lbl_fusion.setEnabled(False)
                return
            
            # Populate model combo
            self.cmb_model.blockSignals(True)
            self.cmb_model.clear()
            for key, entry in precomputed.items():
                self.cmb_model.addItem(entry.display_name, key)
            
            # Select active model
            if registry.active_model and registry.active_model in precomputed:
                idx = self.cmb_model.findData(registry.active_model)
                if idx >= 0:
                    self.cmb_model.setCurrentIndex(idx)
            
            self.cmb_model.blockSignals(False)
            
            self._visual_available = True
            self.chk_visual.setEnabled(True)
            self.chk_visual.setToolTip("Enable deep learning visual similarity ranking")
            self._update_visual_widgets_enabled()
            
        except ImportError:
            self._visual_available = False
            self.chk_visual.setEnabled(False)
            self.chk_visual.setToolTip("DL module not available")
            self.cmb_model.setEnabled(False)
            self.cmb_visual_mode.setEnabled(False)
            self.btn_refresh_visual.setEnabled(False)
            self.chk_roll_to_closest.setEnabled(False)
            self.spin_roll_limit.setEnabled(False)
            self.slider_fusion.setEnabled(False)
            self.lbl_fusion.setEnabled(False)
    
    def _update_visual_widgets_enabled(self):
        """Update visual widget enabled states based on checkbox."""
        on = self.chk_visual.isChecked() and self._visual_available
        is_image_mode = self.cmb_visual_mode.currentData() == "image"
        roll_enabled = on and is_image_mode and self.chk_roll_to_closest.isChecked()
        self.cmb_model.setEnabled(on and self.cmb_model.count() > 1)
        self.cmb_visual_mode.setEnabled(on)
        self.btn_refresh_visual.setEnabled(on and is_image_mode)
        self.chk_roll_to_closest.setEnabled(on and is_image_mode)
        self.spin_roll_limit.setEnabled(roll_enabled)
        self.slider_fusion.setEnabled(on)
        self.lbl_fusion.setEnabled(on)
    
    def _on_visual_toggled(self, checked: bool):
        """Handle visual checkbox toggle."""
        self._ilog.log("checkbox_toggle", "chk_visual", value=str(checked))
        self._update_visual_widgets_enabled()
        if checked:
            self._load_visual_lookup()
        self._refresh_results()
    
    def _on_model_changed(self, index: int):
        """Handle model selection change."""
        model_key = self.cmb_model.currentData()
        self._ilog.log("combo_change", "cmb_model", value=str(model_key))
        if self.chk_visual.isChecked():
            self._load_visual_lookup()
            self._refresh_results()
    
    def _on_fusion_changed(self, value: int):
        """Handle fusion slider change."""
        self._ilog.log("slider_change", "slider_fusion", value=str(value))
        self.lbl_fusion.setText(f"{value}%")
        if self.chk_visual.isChecked():
            self._refresh_results()
    
    def _load_visual_lookup(self):
        """Load the similarity lookup for the selected model."""
        if not self._visual_available:
            self._visual_lookup = None
            self._image_lookup = None
            return
        
        try:
            from src.dl.similarity_lookup import get_lookup, get_image_lookup
            
            model_key = self.cmb_model.currentData()
            if model_key:
                self._visual_lookup = get_lookup(model_key)
                self._image_lookup = get_image_lookup(model_key)
            else:
                self._visual_lookup = None
                self._image_lookup = None
        except Exception as e:
            import logging
            logging.getLogger("starBoard.ui.tab_first_order").warning("Failed to load visual lookup: %s", e)
            self._visual_lookup = None
            self._image_lookup = None
    
    def _get_visual_scores(self, query_id: str) -> Dict[str, float]:
        """Get visual similarity scores for a query."""
        visual_mode = self.cmb_visual_mode.currentData()
        
        if visual_mode == "image":
            return self._get_image_based_scores()
        else:
            return self._get_centroid_scores(query_id)
    
    def _get_centroid_scores(self, query_id: str) -> Dict[str, float]:
        """Get centroid-based similarity scores (identity average)."""
        if not self._visual_lookup or not self._visual_lookup.is_loaded():
            return {}
        
        return self._visual_lookup.get_scores_for_query(query_id)
    
    def _get_image_based_scores(self) -> Dict[str, float]:
        """
        Get image-based similarity scores for the current query image.
        
        Ranks gallery identities by the best-matching image from each.
        Also populates self._closest_gallery_image with (local_idx, path) tuples
        for each gallery identity - uses index for O(1) lookup with path fallback.
        """
        # Clear previous closest image mapping
        self._closest_gallery_image = {}
        
        if not self._image_lookup or not self._image_lookup.is_loaded():
            # Fall back to centroid if image lookup not available
            return self._get_centroid_scores(self._current_query) if self._current_query else {}
        
        # Get current query image path
        current_image_path = self._get_current_query_image_path()
        if not current_image_path:
            return {}
        
        # Get ranked gallery by best-match to current image
        # Returns: List[Tuple[gallery_id, best_score, best_gallery_image_path, local_idx]]
        ranked = self._image_lookup.rank_gallery_by_query_image(current_image_path)
        
        # Build score dict and closest image mapping (both index and path for robustness)
        scores: Dict[str, float] = {}
        for gid, score, path, local_idx in ranked:
            scores[gid] = score
            self._closest_gallery_image[gid] = (local_idx, path)
        
        return scores
    
    def _get_current_query_image_path(self) -> Optional[str]:
        """Get the path of the currently displayed query image."""
        if not hasattr(self, 'query_strip') or not self.query_strip.files:
            return None
        
        try:
            idx = self.query_strip.idx
            if 0 <= idx < len(self.query_strip.files):
                return str(self.query_strip.files[idx])
        except Exception:
            pass
        
        return None
    
    def _on_visual_mode_changed(self, index: int):
        """Handle visual mode selection change."""
        mode = self.cmb_visual_mode.currentData()
        self._ilog.log("combo_change", "cmb_visual_mode", value=str(mode))
        self._update_visual_widgets_enabled()
        if self.chk_visual.isChecked():
            self._refresh_results()
    
    def _on_roll_to_closest_toggled(self, checked: bool):
        """Handle roll-to-closest toggle change."""
        self._ilog.log("checkbox_toggle", "chk_roll_to_closest", value=str(self.chk_roll_to_closest.isChecked()),
                      context={"roll_limit": self.spin_roll_limit.value()})
        # Re-render cards so they roll to closest (or back to best)
        if self.chk_visual.isChecked() and self.cmb_visual_mode.currentData() == "image":
            self._refresh_results()
    
    def _on_refresh_visual(self):
        """Refresh visual ranking for current image."""
        self._ilog.log("button_click", "btn_refresh_visual", value="clicked")
        if self.chk_visual.isChecked():
            self._refresh_results()
    
    def refresh_visual_state(self):
        """Refresh visual ranking state (call after precomputation completes)."""
        self._init_visual_controls()
        if self.chk_visual.isChecked():
            self._load_visual_lookup()
            self._refresh_results()

    # ==================== Verification Model Controls ====================

    def _init_verification_controls(self):
        """Initialize verification ranking controls based on precomputed data availability."""
        try:
            from src.dl import DL_AVAILABLE
            
            if not DL_AVAILABLE:
                self._verification_available = False
                self.chk_verification.setEnabled(False)
                self.chk_verification.setToolTip("PyTorch not installed. Install with: pip install -r requirements-dl.txt")
                self.cmb_verif_model.setEnabled(False)
                self.slider_verif_fusion.setEnabled(False)
                self.lbl_verif_fusion.setEnabled(False)
                return
            
            registry = DLRegistry.load()
            precomputed_verif = {
                k: v for k, v in registry.verification_models.items()
                if v.precomputed
            }
            
            if not precomputed_verif:
                self._verification_available = False
                self.chk_verification.setEnabled(False)
                self.chk_verification.setToolTip(
                    "Run verification precomputation in Deep Learning tab to enable.\n"
                    "This computes P(same individual) scores for all query-gallery pairs."
                )
                self.cmb_verif_model.setEnabled(False)
                self.slider_verif_fusion.setEnabled(False)
                self.lbl_verif_fusion.setEnabled(False)
                return
            
            # Populate verification model combo
            self.cmb_verif_model.blockSignals(True)
            self.cmb_verif_model.clear()
            for key, entry in precomputed_verif.items():
                self.cmb_verif_model.addItem(entry.display_name, key)
            
            # Select active verification model
            if registry.active_verification_model and registry.active_verification_model in precomputed_verif:
                idx = self.cmb_verif_model.findData(registry.active_verification_model)
                if idx >= 0:
                    self.cmb_verif_model.setCurrentIndex(idx)
            
            self.cmb_verif_model.blockSignals(False)
            
            self._verification_available = True
            self.chk_verification.setEnabled(True)
            self.chk_verification.setToolTip(
                "Enable verification model P(same) scores for ranking.\n"
                "Uses cross-attention model to predict probability of same individual."
            )
            self._update_verification_widgets_enabled()
            
        except ImportError:
            self._verification_available = False
            self.chk_verification.setEnabled(False)
            self.chk_verification.setToolTip("DL module not available")
            self.cmb_verif_model.setEnabled(False)
            self.slider_verif_fusion.setEnabled(False)
            self.lbl_verif_fusion.setEnabled(False)

    def _update_verification_widgets_enabled(self):
        """Update verification widget enabled states based on checkbox."""
        on = self.chk_verification.isChecked() and self._verification_available
        self.cmb_verif_model.setEnabled(on and self.cmb_verif_model.count() > 1)
        self.slider_verif_fusion.setEnabled(on)
        self.lbl_verif_fusion.setEnabled(on)

    def _on_verification_toggled(self, checked: bool):
        """Handle verification checkbox toggle."""
        self._ilog.log("checkbox_toggle", "chk_verification", value=str(checked))
        self._update_verification_widgets_enabled()
        if checked:
            self._load_verification_lookup()
        self._refresh_results()

    def _on_verif_model_changed(self, index: int):
        """Handle verification model selection change."""
        model_key = self.cmb_verif_model.currentData()
        self._ilog.log("combo_change", "cmb_verif_model", value=str(model_key))
        if self.chk_verification.isChecked():
            self._load_verification_lookup()
            self._refresh_results()

    def _on_verif_fusion_changed(self, value: int):
        """Handle verification fusion slider change."""
        self._ilog.log("slider_change", "slider_verif_fusion", value=str(value))
        self.lbl_verif_fusion.setText(f"{value}%")
        if self.chk_verification.isChecked():
            self._refresh_results()

    def _load_verification_lookup(self):
        """Load the verification lookup for the selected model."""
        if not self._verification_available:
            self._verification_lookup = None
            return
        
        try:
            model_key = self.cmb_verif_model.currentData()
            if model_key:
                self._verification_lookup = get_verification_lookup(model_key)
            else:
                self._verification_lookup = None
        except Exception as e:
            import logging
            logging.getLogger("starBoard.ui.tab_first_order").warning(
                "Failed to load verification lookup: %s", e
            )
            self._verification_lookup = None

    def _get_verification_scores(self, query_id: str) -> Dict[str, float]:
        """Get verification P(same) scores for a query."""
        if not self._verification_lookup or not self._verification_lookup.is_loaded():
            return {}
        
        return self._verification_lookup.get_scores_for_query(query_id)

    def refresh_verification_state(self):
        """Refresh verification state (call after verification precomputation completes)."""
        self._init_verification_controls()
        if self.chk_verification.isChecked():
            self._load_verification_lookup()
            self._refresh_results()
    
    # ==================== Evaluation-based Query Sorting ====================
    
    def refresh_evaluation_state(self):
        """Refresh evaluation state (call after evaluation completes in DL tab)."""
        self._load_stored_evaluation()
    
    def on_archive_data_changed(self):
        """Handle new data added to archive (e.g., from Morphometric tab)."""
        # Reset engine so it picks up new gallery/query data on next rank()
        self.engine.reset_built()
        # Refresh query list immediately
        self._refresh_query_ids()
    
    def _load_stored_evaluation(self):
        """Load stored evaluation results for the active model."""
        try:
            from src.dl.evaluation import load_evaluation, has_evaluation
            from src.dl.registry import DLRegistry
            
            registry = DLRegistry.load()
            model_key = registry.active_model
            
            if model_key and has_evaluation(model_key):
                self._stored_evaluation = load_evaluation(model_key)
                self.chk_sort_confidence.setEnabled(True)
                self.chk_sort_confidence.setToolTip(
                    f"Sort queries by match confidence (highest similarity first).\n"
                    f"Using evaluation from: {self._stored_evaluation.timestamp[:10] if self._stored_evaluation else 'N/A'}"
                )
            else:
                self._stored_evaluation = None
                self.chk_sort_confidence.setEnabled(False)
                self.chk_sort_confidence.setChecked(False)
                self.chk_sort_confidence.setToolTip(
                    "Sort queries by match confidence (highest similarity first).\n"
                    "Requires evaluation to be run in Deep Learning tab."
                )
        except Exception as e:
            import logging
            logging.getLogger("starBoard.ui.tab_first_order").warning(
                "Failed to load stored evaluation: %s", e
            )
            self._stored_evaluation = None
            self.chk_sort_confidence.setEnabled(False)
    
    def _on_sort_confidence_toggled(self, checked: bool):
        """Handle toggle of sort-by-confidence checkbox."""
        self._ilog.log("checkbox_toggle", "chk_sort_confidence", value=str(checked))
        self._sort_by_confidence = checked
        self._refresh_query_ids()
    
    def _get_confidence_sorted_queries(self, query_ids: List[str]) -> List[str]:
        """
        Sort query IDs by match confidence if evaluation data is available.
        
        Args:
            query_ids: List of query IDs to sort
            
        Returns:
            Sorted list with highest confidence first, then remaining alphabetically
        """
        if not self._stored_evaluation:
            return query_ids
        
        # Get sorted queries from evaluation (highest score first)
        eval_sorted = self._stored_evaluation.get_sorted_queries()
        eval_order = {qid: i for i, (qid, _, _) in enumerate(eval_sorted)}
        
        # Separate queries with evaluation data from those without
        with_eval = []
        without_eval = []
        
        for qid in query_ids:
            if qid in eval_order:
                with_eval.append((eval_order[qid], qid))
            else:
                without_eval.append(qid)
        
        # Sort: evaluated queries by rank, then unevaluated alphabetically
        with_eval.sort(key=lambda x: x[0])
        sorted_queries = [qid for _, qid in with_eval] + sorted(without_eval)
        
        return sorted_queries
    
    def get_top_suggested_match(self, query_id: str) -> Optional[Tuple[str, float]]:
        """
        Get the top suggested gallery match for a query.
        
        Args:
            query_id: The query to look up
            
        Returns:
            (gallery_id, similarity_score) or None if no suggestion
        """
        if self._stored_evaluation:
            return self._stored_evaluation.get_top_match(query_id)
        return None
    
    def _update_suggested_match_display(self, query_id: str):
        """Update the suggested match label for the current query."""
        if not query_id or not self._stored_evaluation:
            self.lbl_suggested_match.setText("")
            return
        
        match = self.get_top_suggested_match(query_id)
        if match:
            gallery_id, score = match
            self.lbl_suggested_match.setText(f"→ {gallery_id} ({score:.2f})")
            self.lbl_suggested_match.setToolTip(
                f"Top suggested match: {gallery_id}\n"
                f"Similarity score: {score:.3f}"
            )
        else:
            self.lbl_suggested_match.setText("")
    
    def on_match_decision_made(self, query_id: str):
        """
        Called when a match decision is made for a query.
        
        Updates the stored evaluation and refreshes the query list if needed.
        """
        if self._stored_evaluation and query_id in self._stored_evaluation.query_suggestions:
            # Remove from stored evaluation
            self._stored_evaluation.remove_query(query_id)
            
            # Also update the persisted file
            try:
                from src.dl.evaluation import update_stored_evaluation
                from src.dl.registry import DLRegistry
                registry = DLRegistry.load()
                if registry.active_model:
                    update_stored_evaluation(registry.active_model, query_id)
            except Exception:
                pass
            
            # Refresh query list if sorting by confidence
            if self._sort_by_confidence:
                self._refresh_query_ids()
