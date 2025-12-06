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
    QCompleter,
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
from src.data.best_photo import reorder_files_with_best, save_best_for_id
from src.data.archive_paths import last_observation_for_all

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
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)                     # non-blocking
        self.setAttribute(Qt.WA_DeleteOnClose)   # close frees memory
        self.resize(540, 520)

        lay = QVBoxLayout(self)
        self.table = QTableWidget(0, 2, self)
        self.table.setHorizontalHeaderLabels(["Field", "Value"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setWordWrap(True)
        lay.addWidget(self.table)

    def populate(self, row: Dict[str, str]):
        fields = [k for k in row.keys() if k]
        self.table.setRowCount(len(fields))
        for i, k in enumerate(fields):
            self.table.setItem(i, 0, QTableWidgetItem(k))
            self.table.setItem(i, 1, QTableWidgetItem(row.get(k, "")))


# ---------------- exclusion dialog ----------------
class _ExcludeDialog(QDialog):
    """
    Lets the user temporarily exclude specific gallery IDs and/or
    exclude all items from selected 'Last location' values.
    Exclusions are not persisted; caller should store results in memory.
    """
    def __init__(self, all_gallery_ids: List[str], last_locations: List[str],
                 excluded_ids: Set[str], excluded_locations: Set[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Exclude from ranking")
        self.resize(560, 600)

        self._ids_initial = set(excluded_ids)
        self._locs_initial = set(excluded_locations)

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

        # --- by location
        root.addWidget(QLabel("<b>Exclude by location</b>"))
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
            self.list_locations.addItem(QListWidgetItem(loc))

    def _on_clear_locations(self):
        self.list_locations.clear()

    def get_results(self) -> Tuple[Set[str], Set[str]]:
        ids: Set[str] = set()
        for i in range(self.list_ids.count()):
            it = self.list_ids.item(i)
            if it.checkState() == Qt.Checked:
                ids.add(it.text())
        locs: Set[str] = {self.list_locations.item(i).text() for i in range(self.list_locations.count())}
        return ids, locs


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
                w.setParent(None)
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

        # session-scoped exclusions (NOT persisted)
        self._excluded_ids: Set[str] = set()
        self._excluded_locations: Set[str] = set()

        # date filter state
        self._dates_initialized: bool = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        # --- Controls row ---
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Query:"))
        self.cmb_query = QComboBox()
        self.cmb_query.setMinimumWidth(260)
        # Make combo editable for type-to-search functionality
        self.cmb_query.setEditable(True)
        self.cmb_query.setInsertPolicy(QComboBox.NoInsert)
        self.cmb_query.completer().setFilterMode(Qt.MatchContains)
        self.cmb_query.completer().setCompletionMode(QCompleter.PopupCompletion)
        controls.addWidget(self.cmb_query)

        controls.addWidget(QLabel("Preset:"))
        self.cmb_preset = QComboBox()
        self.cmb_preset.addItems(list(PRESETS.keys()))
        self.cmb_preset.currentIndexChanged.connect(self._apply_preset)
        controls.addWidget(self.cmb_preset)

        controls.addWidget(QLabel("Top-K:"))
        self.spin_topk = QSpinBox()
        self.spin_topk.setRange(1, 500)
        self.spin_topk.setValue(50)
        controls.addWidget(self.spin_topk)

        self.btn_rebuild = QPushButton("Rebuild index")
        self.btn_rebuild.clicked.connect(self._on_rebuild)
        controls.addWidget(self.btn_rebuild)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self._refresh_results)
        controls.addWidget(self.btn_refresh)

        self.btn_exclude = QPushButton("Exclude…")
        self.btn_exclude.setToolTip("Temporarily exclude gallery members or entire locations")
        self.btn_exclude.clicked.connect(self._open_exclude_dialog)
        controls.addWidget(self.btn_exclude)

        self.btn_config = QPushButton("Set up config…")
        self.btn_config.setToolTip("Configure field weights, enabled fields, and scorer parameters")
        self.btn_config.clicked.connect(self._open_config_dialog)
        controls.addWidget(self.btn_config)

        # ---- NEW: Query date filter (From/To + include-no-date) ----
        controls.addSpacing(10)
        self.chk_date = QCheckBox("Date filter")
        self.chk_date.toggled.connect(self._on_date_filter_changed)
        controls.addWidget(self.chk_date)

        controls.addWidget(QLabel("From:"))
        self.date_from = QDateEdit()
        self.date_from.setCalendarPopup(True)
        self.date_from.setDisplayFormat("yyyy-MM-dd")
        self.date_from.dateChanged.connect(self._on_date_filter_changed)
        controls.addWidget(self.date_from)

        controls.addWidget(QLabel("To:"))
        self.date_to = QDateEdit()
        self.date_to.setCalendarPopup(True)
        self.date_to.setDisplayFormat("yyyy-MM-dd")
        self.date_to.dateChanged.connect(self._on_date_filter_changed)
        controls.addWidget(self.date_to)

        self.chk_include_nodate = QCheckBox("Include no‑date")
        self.chk_include_nodate.setToolTip("Include queries with no detectable sampling date")
        self.chk_include_nodate.toggled.connect(self._on_date_filter_changed)
        controls.addWidget(self.chk_include_nodate)

        # initialize enabled/disabled state
        self._update_date_widgets_enabled()

        controls.addStretch(1)
        self.lbl_excluded = QLabel("Excluded: 0")
        controls.addWidget(self.lbl_excluded)
        self.lbl_pinned = QLabel("Pinned: 0")
        controls.addWidget(self.lbl_pinned)
        outer.addLayout(controls)

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
        self.lbl_query_id = QLabel("—")
        qwrap_l.addWidget(QLabel("ID:"))
        qwrap_l.addWidget(self.lbl_query_id)

        self.query_strip = ImageStrip(files=[], long_edge=768)

        # query footer (Open Folder + View Metadata)
        q_footer = QWidget()
        qf_l = QHBoxLayout(q_footer)
        qf_l.setContentsMargins(0, 0, 0, 0)
        self.btn_open_query = QPushButton("Open Folder")
        self.btn_best_query = QPushButton("Set Best (Query)")
        self.btn_open_query.clicked.connect(self._open_query_folder)
        self.btn_best_query.clicked.connect(self._on_set_best_query)
        self.btn_meta_query = QPushButton("View Metadata")
        self.btn_meta_query.clicked.connect(self._show_query_metadata)
        qf_l.addStretch(1)
        qf_l.addWidget(self.btn_open_query)
        qf_l.addWidget(self.btn_best_query)
        qf_l.addWidget(self.btn_meta_query)

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

        # Populate query IDs and hook signals
        self._refresh_query_ids()
        self.cmb_query.currentIndexChanged.connect(self._on_query_changed)

        # Sync field checkboxes with config (initial load)
        self._sync_checkboxes_from_config()

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

        # Move queries that already have a positive match ("yes") to the bottom
        unmatched: List[str] = []
        matched: List[str] = []
        for qid in ids:
            try:
                latest = load_latest_map_for_query(qid)  # {gid -> row}
                has_yes = any(((row.get("verdict", "") or "").strip().lower() == "yes") for row in latest.values())
            except Exception:
                has_yes = False
            (matched if has_yes else unmatched).append(qid)
        ordered = sorted(unmatched, key=_date_alpha_key) + sorted(matched, key=_date_alpha_key)

        self.cmb_query.blockSignals(True)
        self.cmb_query.clear()
        self.cmb_query.addItems(ordered)
        # try to restore previous selection if still present
        if prev_sel and (idx := self.cmb_query.findText(prev_sel)) >= 0:
            self.cmb_query.setCurrentIndex(idx)
        elif ordered:
            self.cmb_query.setCurrentIndex(0)
        self.cmb_query.blockSignals(False)

        # ensure downstream state aligns
        if ordered:
            self._on_query_changed()
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
        self.engine.rebuild()
        # Rebuild can change date coverage; refresh combo accordingly
        self._dates_initialized = False
        self._refresh_query_ids()
        self._refresh_results()

    def _open_query_folder(self):
        qid = self._current_query
        if not qid:
            return
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
        q_row = self.engine._queries_rows_by_id.get(self._current_query, {})
        if not q_row:
            return
        if self._meta_popup is None or not self._meta_popup.isVisible():
            self._meta_popup = _MetadataPopup(f"Metadata: {self._current_query}", self)
        self._meta_popup.populate(q_row)
        self._meta_popup.show()
        self._meta_popup.raise_()

    def _open_exclude_dialog(self):
        # Gather gallery IDs and available location values
        gallery_ids = sorted(list(self.engine._gallery_rows_by_id.keys()))
        locs: List[str] = []
        for r in self.engine._gallery_rows_by_id.values():
            locs.append((r.get("location", "") or "").strip())
        dlg = _ExcludeDialog(gallery_ids, locs, set(self._excluded_ids), set(self._excluded_locations), self)
        if dlg.exec():
            ids, locs_set = dlg.get_results()
            self._excluded_ids = set(ids)
            self._excluded_locations = set(locs_set)
            self.lbl_excluded.setText(f"Excluded: {len(self._excluded_ids) or 0} + {len(self._excluded_locations) or 0} loc")
            self._refresh_results()

    def _open_config_dialog(self):
        """Open the fields configuration dialog."""
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
        self._refresh_results()

    def _on_set_best_query(self):
        qid = self._current_query
        if not qid or not self.query_strip.files:
            return
        idx = max(0, min(self.query_strip.idx, len(self.query_strip.files) - 1))
        save_best_for_id("Queries", qid, self.query_strip.files[idx])
        files = reorder_files_with_best("Queries", qid, list(self.query_strip.files))
        self.query_strip.set_files(files)

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
        if not fields:
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

        results = self.engine.rank(
            qid,
            include_fields=fields,
            equalize_weights=equalize,
            weights=weights,
            top_k=int(self.spin_topk.value()),
            numeric_offsets=self._collect_numeric_offsets(),
        )

        # ---- Manual exclusions still EXCLUDE; auto same-day YES now DEMOTES ----
        demote_ids = set(self._auto_excluded_for_same_day(qid))  # keep logic; change handling
        excluded_ids = set(self._excluded_ids)  # manual-only
        excluded_locs = set(self._excluded_locations)

        kept, demoted = [], []
        for it in results:
            # Manual ID exclusion => drop
            if it.gallery_id in excluded_ids:
                continue
            # Manual location exclusion => drop
            g_row = self.engine._gallery_rows_by_id.get(it.gallery_id, {})  # safe default
            loc = (g_row.get("location", "") or "").strip()
            if loc and loc in excluded_locs:
                continue
            # Auto same-day YES => demote to the end; otherwise keep in place
            (demoted if it.gallery_id in demote_ids else kept).append(it)

        ordered = kept + demoted  # preserve relative order within each bucket

        # ---- Build cards (with robust constructor call & rich tooltips) ----
        cards: List[LineupCard] = []
        q_row = self.engine._queries_rows_by_id.get(qid, {})
        for it in ordered:
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
                )
            except TypeError:
                try:
                    badges = sorted(list(it.field_breakdown.keys()))
                    card = LineupCard(it.gallery_id, it.score, it.k_contrib, badges)
                except TypeError:
                    card = LineupCard(it.gallery_id, it.score, it.k_contrib, it.field_breakdown)

            if hasattr(card, "btn_pin"):
                card.btn_pin.clicked.connect(lambda _=None, gid=it.gallery_id: self._toggle_pin(gid))
            cards.append(card)

        self._set_cards(cards)
        self._fit_cards_to_viewport()

    def _set_cards(self, cards: List[LineupCard]):
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
        self._cards = cards
        for c in cards:
            c.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
            self.cards_layout.addWidget(c)
        self.cards_layout.addStretch(1)

        # Already present:
        self._sync_card_min_height_from_query()  # keep

        # NEW: ensure min width is also synced immediately
        self._sync_card_min_width_from_query()

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

    def _toggle_pin(self, gid: str):
        if gid in self._pinned:
            self._pinned.remove(gid)
        else:
            self._pinned.append(gid)
        self.lbl_pinned.setText(f"Pinned: {len(self._pinned)}")
        self._save_pins()

    def _on_metadata_csv_changed(self, _path: str):
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

        self.btn_gallery_prev = QPushButton("◀ Prev result")
        self.btn_gallery_next = QPushButton("Next result ▶")

        self.btn_gallery_prev.setToolTip(
            "Scroll to the previous gallery identity and center it in view."
        )
        self.btn_gallery_next.setToolTip(
            "Scroll to the next gallery identity and center it in view."
        )

        row.addStretch(1)
        row.addWidget(self.btn_gallery_prev)
        row.addWidget(self.btn_gallery_next)
        vlay.addLayout(row)

        # Wire up behavior.
        self.btn_gallery_prev.clicked.connect(self._on_gallery_prev_clicked)
        self.btn_gallery_next.clicked.connect(self._on_gallery_next_clicked)

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
