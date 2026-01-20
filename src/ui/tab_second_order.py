# src/ui/tab_second_order.py
from __future__ import annotations

import json, os, platform, subprocess
from pathlib import Path
from typing import Dict

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QSplitter, QLineEdit, QScrollArea, QSizePolicy, QFrame, QCompleter,
)
from datetime import date as _date
from src.data.id_registry import list_ids
from src.data.image_index import list_image_files
from src.data import archive_paths as ap
from src.data.best_photo import reorder_files_with_best, save_best_for_id
from src.ui.annotator_view_second import AnnotatorViewSecond
from src.data.compare_labels import get_label_for_pair, save_label_for_pair, load_latest_map_for_query
from src.data.merge_yes import is_query_silent
from src.ui.query_state_delegate import (
    QueryStateDelegate, QueryState, QUERY_STATE_ROLE,
    get_query_state, apply_query_states_to_combobox, apply_quality_to_combobox
)

# Reuse First‑order’s field scorers + palette to build the bottom diff bar
from src.search.engine import FirstOrderSearchEngine, ALL_FIELDS
from src.ui.lineup_card import (
    DEFAULT_CMAP, FIELD_COLORMAPS, _interp_color, _css_color, _text_color_for_bg
)
from src.data.archive_paths import last_observation_for_all
from src.data.encounter_info import (
    get_encounter_date_from_path, format_encounter_date,
    get_queries_for_encounter, format_queries_for_display,
    invalidate_gallery_queries_cache
)
from src.ui.image_quality_panel import ImageQualityPanel
from src.utils.interaction_logger import get_interaction_logger
from src.dl.verification_lookup import get_active_verification_lookup
from src.dl.verification_evaluation import (
    get_optimal_threshold_for_active_model, colorize_verification_score
)

class TabSecondOrder(QWidget):
    """
    Second-order compare tab:
      - Choose one Query ID and one Gallery ID (twin viewers)
      - "Pinned / Maybe" combo: shows gallery IDs that are pinned OR have a "maybe" verdict
      - Decision combo (Yes/Maybe/No) + Notes + Save per (query_id, gallery_id)
      - Metadata diff bar at the bottom (fields color‑coded by similarity)

    Change requested:
      After saving a decision (Yes/No/Maybe) OR after a picture changes
      (by changing Query/Gallery ID or navigating), reset tools to Select.
    """
    
    # Signal emitted when a match decision is saved (query_id, verdict)
    matchDecisionMade = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)

        # First‑order engine (lazy build) for field-level comparisons
        self._engine = FirstOrderSearchEngine()
        self._ilog = get_interaction_logger()
        
        # Verification lookup (lazy-loaded)
        self._verification_lookup = None
        self._verification_loaded = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        # ---- Row 1: IDs + Recommendations ----
        row1 = QHBoxLayout(); row1.setContentsMargins(0, 0, 0, 0)

        row1.addWidget(QLabel("Query:"))
        self.cmb_query = QComboBox(); self.cmb_query.setMinimumWidth(280)
        # Make combo editable for type-to-search functionality
        self.cmb_query.setEditable(True)
        self.cmb_query.setInsertPolicy(QComboBox.NoInsert)
        self.cmb_query.completer().setFilterMode(Qt.MatchContains)
        self.cmb_query.completer().setCompletionMode(QCompleter.PopupCompletion)
        # Apply state-based color coding delegate
        self._query_state_delegate = QueryStateDelegate(self.cmb_query)
        self.cmb_query.setItemDelegate(self._query_state_delegate)
        row1.addWidget(self.cmb_query)

        # Query navigation buttons
        self.btn_prev_query = QPushButton("◀")
        self.btn_prev_query.setFixedWidth(28)
        self.btn_prev_query.setToolTip("Previous query in list")
        self.btn_prev_query.clicked.connect(self._on_prev_query_clicked)
        row1.addWidget(self.btn_prev_query)

        self.btn_next_query = QPushButton("▶")
        self.btn_next_query.setFixedWidth(28)
        self.btn_next_query.setToolTip("Next query in list")
        self.btn_next_query.clicked.connect(self._on_next_query_clicked)
        row1.addWidget(self.btn_next_query)

        row1.addSpacing(12)
        row1.addWidget(QLabel("Pinned / Maybe:"))
        self.cmb_recommended = QComboBox(); self.cmb_recommended.setMinimumWidth(220)
        row1.addWidget(self.cmb_recommended)

        row1.addSpacing(12)
        row1.addWidget(QLabel("Gallery:"))
        self.cmb_gallery = QComboBox(); self.cmb_gallery.setMinimumWidth(280)
        # Make combo editable for type-to-search functionality
        self.cmb_gallery.setEditable(True)
        self.cmb_gallery.setInsertPolicy(QComboBox.NoInsert)
        self.cmb_gallery.completer().setFilterMode(Qt.MatchContains)
        self.cmb_gallery.completer().setCompletionMode(QCompleter.PopupCompletion)
        # Apply quality indicator delegate (no workflow state for gallery)
        self._gallery_delegate = QueryStateDelegate(self.cmb_gallery, show_quality_symbols=True)
        self.cmb_gallery.setItemDelegate(self._gallery_delegate)
        row1.addWidget(self.cmb_gallery)

        row1.addStretch(1)
        outer.addLayout(row1)

        # ---- Row 2: Decision + Notes + Open folders ----
        row2 = QHBoxLayout(); row2.setContentsMargins(0, 0, 0, 0)

        row2.addWidget(QLabel("Decision:"))
        self.cmb_verdict = QComboBox()
        self.cmb_verdict.addItems(["—", "Yes (positive match)", "Maybe", "No"])
        row2.addWidget(self.cmb_verdict)

        self.btn_save_decision = QPushButton("Save decision")
        row2.addWidget(self.btn_save_decision)

        # Verification score display
        row2.addSpacing(16)
        self.lbl_verification = QLabel("")
        self.lbl_verification.setMinimumWidth(120)
        self.lbl_verification.setToolTip(
            "Verification model confidence that these are the same individual.\n"
            "High (>0.7): likely match | Medium (0.4-0.7): review | Low (<0.4): unlikely"
        )
        row2.addWidget(self.lbl_verification)

        row2.addSpacing(16)
        row2.addWidget(QLabel("Notes:"))
        self.edit_notes = QLineEdit()
        self.edit_notes.setPlaceholderText("Reasoning / landmarks / caveats…")
        self.edit_notes.setMinimumWidth(300)
        self.edit_notes.setClearButtonEnabled(True)
        row2.addWidget(self.edit_notes, 1)

        self.btn_open_q = QPushButton("Open Query Folder")
        self.btn_open_g = QPushButton("Open Gallery Folder")
        row2.addSpacing(8); row2.addWidget(self.btn_open_q)
        row2.addSpacing(4); row2.addWidget(self.btn_open_g)

        outer.addLayout(row2)

        # ---- Twin viewers ----
        self.split = QSplitter(Qt.Horizontal)
        
        # Query side with image quality panel
        query_container = QWidget()
        query_layout = QVBoxLayout(query_container)
        query_layout.setContentsMargins(0, 0, 0, 0)
        query_layout.setSpacing(4)
        self.view_q = AnnotatorViewSecond(target="Queries", title="Query")
        query_layout.addWidget(self.view_q, 1)
        self.query_quality_panel = ImageQualityPanel(
            parent=query_container,
            show_save_button=True,
            compact=True,
            title="",
        )
        self.query_quality_panel.set_target("Queries")
        self.query_quality_panel.saved.connect(self._on_query_quality_saved)
        query_layout.addWidget(self.query_quality_panel)
        
        # Gallery side with image quality panel and encounter info
        gallery_container = QWidget()
        gallery_layout = QVBoxLayout(gallery_container)
        gallery_layout.setContentsMargins(0, 0, 0, 0)
        gallery_layout.setSpacing(4)
        
        # Encounter info label (date and matched queries)
        self._gallery_encounter_info = QLabel("")
        self._gallery_encounter_info.setStyleSheet(
            "QLabel { color: #e67e22; font-size: 12px; font-weight: bold; padding: 2px 4px; }"
        )
        self._gallery_encounter_info.setToolTip(
            "Shows encounter date and queries that matched this gallery member"
        )
        gallery_layout.addWidget(self._gallery_encounter_info)
        
        self.view_g = AnnotatorViewSecond(target="Gallery", title="Gallery")
        gallery_layout.addWidget(self.view_g, 1)
        self.gallery_quality_panel = ImageQualityPanel(
            parent=gallery_container,
            show_save_button=True,
            compact=True,
            title="",
        )
        self.gallery_quality_panel.set_target("Gallery")
        self.gallery_quality_panel.saved.connect(self._on_gallery_quality_saved)
        gallery_layout.addWidget(self.gallery_quality_panel)
        
        self.split.addWidget(query_container)
        self.split.addWidget(gallery_container)
        self.split.setStretchFactor(0, 1)
        self.split.setStretchFactor(1, 1)
        self.split.setHandleWidth(6)
        outer.addWidget(self.split, 1)

        # ---- Set Best buttons row (below image panes) ----
        best_row = QHBoxLayout()
        best_row.setContentsMargins(0, 0, 0, 0)
        self.btn_best_q = QPushButton("Set Best (Query)")
        self.btn_best_g = QPushButton("Set Best (Gallery)")
        best_row.addStretch(1)
        best_row.addWidget(self.btn_best_q)
        best_row.addSpacing(8)
        best_row.addWidget(self.btn_best_g)
        best_row.addStretch(1)
        outer.addLayout(best_row)

        # Reset tools to Select whenever either viewer changes picture
        # (covers: changing ID, switching encounter, prev/next buttons)
        #   AnnotatorViewSecond emits this in set_files() and after navigation. :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}
        self.view_q.currentImageChanged.connect(self._reset_tools_to_select)
        self.view_g.currentImageChanged.connect(self._reset_tools_to_select)
        
        # Update gallery encounter info when image changes
        self.view_g.currentImageChanged.connect(self._update_gallery_encounter_info)

        # ---- Bottom metadata diff bar ----
        self._meta_frame = QFrame(); self._meta_frame.setObjectName("MetaBar")
        self._meta_frame.setFrameShape(QFrame.StyledPanel)
        self._meta_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._meta_frame.setMaximumHeight(44)
        meta_lay = QHBoxLayout(self._meta_frame)
        meta_lay.setContentsMargins(6, 4, 6, 4); meta_lay.setSpacing(6)

        self._meta_scroll = QScrollArea(); self._meta_scroll.setWidgetResizable(True)
        self._meta_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._meta_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._meta_inner = QWidget()
        self._meta_hbox = QHBoxLayout(self._meta_inner)
        self._meta_hbox.setContentsMargins(0, 0, 0, 0); self._meta_hbox.setSpacing(4)
        self._meta_hbox.addStretch(1)
        self._meta_scroll.setWidget(self._meta_inner)
        meta_lay.addWidget(self._meta_scroll, 1)
        outer.addWidget(self._meta_frame)

        # ---- Signals ----
        self.cmb_query.currentIndexChanged.connect(self._on_query_changed)
        self.cmb_gallery.currentIndexChanged.connect(self._on_gallery_changed)
        self.cmb_recommended.currentIndexChanged.connect(self._on_recommended_changed)
        self.btn_save_decision.clicked.connect(self._on_save_decision)
        self.btn_open_q.clicked.connect(lambda: self._open_id_folder("Queries", self.cmb_query.currentText()))
        self.btn_open_g.clicked.connect(lambda: self._open_id_folder("Gallery", self.cmb_gallery.currentText()))
        self.btn_best_q.clicked.connect(self._on_set_best_query)
        self.btn_best_g.clicked.connect(self._on_set_best_gallery)

        # ---- Populate IDs ----
        self._refresh_ids()

    # ----- helpers -----
    def _refresh_ids(self) -> None:
        # Remember prior selections so we can restore them if still present
        prev_q = self.cmb_query.currentText()
        prev_g = self.cmb_gallery.currentText()

        # Include *all* queries; we'll demote silent/past-match ones instead of hiding them
        qs_all = list_ids("Queries")  # includes silent
        silent = {qid for qid in qs_all if is_query_silent(qid)}

        last_obs = last_observation_for_all("Queries")

        def _date_alpha_key(qid: str):
            d = last_obs.get(qid)
            # Keep items with a date first (ascending by date), then no-date,
            # then case-insensitive alphabetical.
            return (d is None, d or _date.max, qid.lower())

        # Active queries first, then silent/merged queries, each internally date+alpha sorted
        active = sorted([q for q in qs_all if q not in silent], key=_date_alpha_key)
        demoted = sorted([q for q in qs_all if q in silent], key=_date_alpha_key)
        qs = active + demoted

        gs = list_ids("Gallery")
        
        # Compute query states for color coding
        query_states = {qid: get_query_state(qid) for qid in qs}

        # Repopulate with signals blocked to avoid spurious slot executions
        self.cmb_query.blockSignals(True)
        self.cmb_gallery.blockSignals(True)
        try:
            self.cmb_query.clear()
            self.cmb_gallery.clear()

            if qs:
                self.cmb_query.addItems(qs)
                # Apply state-based color coding
                apply_query_states_to_combobox(self.cmb_query, qs, query_states)
                # Apply quality indicator symbols
                apply_quality_to_combobox(self.cmb_query, qs, "Queries")
            if gs:
                self.cmb_gallery.addItems(gs)
                # Apply quality indicator symbols to gallery
                apply_quality_to_combobox(self.cmb_gallery, gs, "Gallery")

            # Try to restore selections; otherwise default to first item
            if qs:
                i = self.cmb_query.findText(prev_q) if prev_q else -1
                self.cmb_query.setCurrentIndex(i if i >= 0 else 0)

            if gs:
                j = self.cmb_gallery.findText(prev_g) if prev_g else -1
                self.cmb_gallery.setCurrentIndex(j if j >= 0 else 0)

        finally:
            self.cmb_query.blockSignals(False)
            self.cmb_gallery.blockSignals(False)

        # Drive dependent UI exactly once per control
        self._on_query_changed()
        self._on_gallery_changed()

    def add_first_order_sync(self, first_order: "TabFirstOrder") -> None:
        """
        Inject a small one-click bridge that pulls the currently selected
        First‑order Query *ID*, *image*, and *view* (pan/zoom/rotation)
        into this Second‑order tab's Query viewer.

        This is 'surgical': it does not alter existing row wiring — it simply
        inserts a tiny toolbar row with a button and stores a weak reference.
        """
        self._first_order_ref = first_order

        # Add a thin row under the existing Row 1 (IDs + Recommendations).
        # We do this at runtime so we don't need to modify __init__.
        lay = self.layout()
        if lay is None:
            return

        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(6)

        self.btn_use_first = QPushButton("Use First‑order Selection")
        self.btn_use_first.setToolTip(
            "Copy the current First‑order Query, selected image, and view here."
        )
        self.btn_use_first.clicked.connect(self._use_first_order_selection)

        bar.addStretch(1)
        bar.addWidget(self.btn_use_first)

        # Insert just after the top controls row (index 1 is right after the first addLayout)
        idx = min(1, lay.count())
        lay.insertLayout(idx, bar)

    def _use_first_order_selection(self) -> None:
        fo = getattr(self, "_first_order_ref", None)
        if fo is None:
            return

        try:
            payload = fo.export_current_query_selection() or {}
        except Exception:
            payload = {}

        qid = (payload.get("query_id") or "").strip()
        img_str = payload.get("image_path") or ""
        view_state = payload.get("view_state") or {}

        if not qid:
            return  # no selection in First‑order

        # 1) Switch our Query combo to that ID without firing redundant signals
        i = self.cmb_query.findText(qid)
        if i >= 0:
            self.cmb_query.blockSignals(True)
            self.cmb_query.setCurrentIndex(i)
            self.cmb_query.blockSignals(False)
        # Ensure the viewer is refreshed (uses the tab's existing pipeline/order)
        self._on_query_changed()

        # 2) Align the current image + view
        try:
            target_path = Path(img_str) if img_str else None
            files = list(self.view_q.strip.files or [])
            idx = 0
            if target_path:
                # Normalize for cross‑platform case differences
                for j, p in enumerate(files):
                    if Path(p) == target_path:
                        idx = j
                        break
            # Override idx in the view state and apply
            vs = dict(view_state)
            vs["idx"] = idx
            self.view_q.strip.set_view_state(vs)
        except Exception:
            pass

    def _pins_path(self, qid: str) -> Path:
        return ap.queries_root(prefer_new=True) / qid / "_pins_first_order.json"

    def _load_pinned_and_maybes(self, qid: str) -> None:
        """Load gallery IDs that are either pinned or have a 'maybe' verdict."""
        self.cmb_recommended.blockSignals(True)
        self.cmb_recommended.clear()
        if not qid:
            self.cmb_recommended.blockSignals(False)
            return
        try:
            # 1. Load explicitly pinned gallery IDs
            pinned_ids: list[str] = []
            p = self._pins_path(qid)
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                pinned_ids = list(dict.fromkeys([str(x) for x in (data.get("pinned") or [])]))
            
            # 2. Load gallery IDs with "maybe" verdict from second-order labels
            maybe_ids: list[str] = []
            try:
                latest = load_latest_map_for_query(qid)
                for gallery_id, row in latest.items():
                    verdict = (row.get("verdict", "") or "").strip().lower()
                    if verdict == "maybe":
                        maybe_ids.append(gallery_id)
            except Exception:
                pass
            
            # 3. Build combined list: pinned first, then maybes (deduplicated)
            # Track which IDs came from which source for display
            combined: list[tuple[str, str]] = []  # (gallery_id, source)
            seen: set[str] = set()
            
            for gid in pinned_ids:
                if gid not in seen:
                    # Check if also has maybe verdict
                    source = "pinned+maybe" if gid in maybe_ids else "pinned"
                    combined.append((gid, source))
                    seen.add(gid)
            
            for gid in maybe_ids:
                if gid not in seen:
                    combined.append((gid, "maybe"))
                    seen.add(gid)
            
            # 4. Populate combo box with visual indicators
            self.cmb_recommended.addItem("—", userData=None)
            for gid, source in combined:
                if source == "pinned+maybe":
                    label = f"📌 {gid} (maybe)"
                elif source == "pinned":
                    label = f"📌 {gid}"
                else:  # maybe only
                    label = f"❓ {gid}"
                self.cmb_recommended.addItem(label, userData=gid)
                
        except Exception:
            self.cmb_recommended.addItem("—", userData=None)
        finally:
            self.cmb_recommended.blockSignals(False)



    def _on_query_changed(self) -> None:
        qid = self.cmb_query.currentText()
        self._ilog.log("combo_change", "cmb_query_second", value=qid)
        files = list_image_files("Queries", qid) if qid else []
        files = reorder_files_with_best("Queries", qid, files) if qid else files
        self.view_q.set_files(files)
        self._load_pinned_and_maybes(qid)
        self._load_decision_ui()
        self._refresh_meta_bar()
        self._refresh_verification_display()
        
        # Update query image quality panel
        if hasattr(self, 'query_quality_panel'):
            self.query_quality_panel.load_for_id("Queries", qid)

    def _on_prev_query_clicked(self) -> None:
        """Navigate to the previous query in the combo box list."""
        self._ilog.log("button_click", "btn_prev_query_second", value="clicked")
        current_idx = self.cmb_query.currentIndex()
        if current_idx > 0:
            self.cmb_query.setCurrentIndex(current_idx - 1)

    def _on_next_query_clicked(self) -> None:
        """Navigate to the next query in the combo box list."""
        self._ilog.log("button_click", "btn_next_query_second", value="clicked")
        current_idx = self.cmb_query.currentIndex()
        max_idx = self.cmb_query.count() - 1
        if current_idx < max_idx:
            self.cmb_query.setCurrentIndex(current_idx + 1)

    def _on_gallery_changed(self) -> None:
        gid = self.cmb_gallery.currentText()
        self._ilog.log("combo_change", "cmb_gallery_second", value=gid)
        files = list_image_files("Gallery", gid) if gid else []
        files = reorder_files_with_best("Gallery", gid, files) if gid else files
        self.view_g.set_files(files)
        self._load_decision_ui()
        self._refresh_meta_bar()
        self._refresh_verification_display()
        
        # Update gallery encounter info display
        self._update_gallery_encounter_info()
        
        # Update gallery image quality panel
        if hasattr(self, 'gallery_quality_panel'):
            self.gallery_quality_panel.load_for_id("Gallery", gid)

    def _on_set_best_query(self) -> None:
        qid = self.cmb_query.currentText()
        if not qid or not self.view_q.strip.files:
            return
        idx = max(0, min(self.view_q.strip.idx, len(self.view_q.strip.files) - 1))
        self._ilog.log("button_click", "btn_best_query_second", value=qid,
                      context={"image_idx": idx})
        save_best_for_id("Queries", qid, self.view_q.strip.files[idx])
        # Reload with best rolled to front
        files = reorder_files_with_best("Queries", qid, list(self.view_q.strip.files))
        self.view_q.set_files(files)

    def _on_set_best_gallery(self) -> None:
        gid = self.cmb_gallery.currentText()
        if not gid or not self.view_g.strip.files:
            return
        idx = max(0, min(self.view_g.strip.idx, len(self.view_g.strip.files) - 1))
        self._ilog.log("button_click", "btn_best_gallery_second", value=gid,
                      context={"image_idx": idx})
        save_best_for_id("Gallery", gid, self.view_g.strip.files[idx])
        files = reorder_files_with_best("Gallery", gid, list(self.view_g.strip.files))
        self.view_g.set_files(files)

    def _on_recommended_changed(self) -> None:
        # Get the actual gallery ID from userData (not the display label)
        gid = self.cmb_recommended.currentData()
        if gid is None:
            gid = ""
        self._ilog.log("combo_change", "cmb_recommended", value=gid)
        if gid:
            idx = self.cmb_gallery.findText(gid)
            if idx >= 0:
                self.cmb_gallery.setCurrentIndex(idx)
            else:
                self.cmb_gallery.addItem(gid)
                self.cmb_gallery.setCurrentIndex(self.cmb_gallery.count() - 1)
        # Always refresh the bar regardless of whether selection changed
        self._refresh_meta_bar()

    def _load_decision_ui(self) -> None:
        qid = self.cmb_query.currentText() or ""
        gid = self.cmb_gallery.currentText() or ""
        self.cmb_verdict.blockSignals(True)
        self.edit_notes.blockSignals(True)
        try:
            self.cmb_verdict.setCurrentIndex(0)
            self.edit_notes.setText("")
            if not qid or not gid:
                return
            row = get_label_for_pair(qid, gid)
            if not row:
                return
            verdict = (row.get("verdict", "") or "").lower()
            notes = row.get("notes", "") or ""
            idx = 0
            if verdict == "yes": idx = 1
            elif verdict == "maybe": idx = 2
            elif verdict == "no": idx = 3
            self.cmb_verdict.setCurrentIndex(idx)
            self.edit_notes.setText(notes)
        finally:
            self.cmb_verdict.blockSignals(False)
            self.edit_notes.blockSignals(False)

    def _on_save_decision(self) -> None:
        qid = self.cmb_query.currentText() or ""
        gid = self.cmb_gallery.currentText() or ""
        if not qid or not gid:
            return
        idx = self.cmb_verdict.currentIndex()
        verdict = ""
        if idx == 1: verdict = "yes"
        elif idx == 2: verdict = "maybe"
        elif idx == 3: verdict = "no"
        notes = self.edit_notes.text()
        self._ilog.log("decision_save", "second_order", value=verdict,
                      context={"query_id": qid, "gallery_id": gid})
        save_label_for_pair(qid, gid, verdict, notes)
        
        # Invalidate global gallery-to-queries cache (verdict changes affect matched queries)
        invalidate_gallery_queries_cache()
        # Update encounter info display to reflect new match status
        self._update_gallery_encounter_info()
        
        # Update the query state color for this query
        self._update_single_query_state(qid)
        
        # Refresh pinned/maybe dropdown if verdict changed (add/remove maybe)
        self._load_pinned_and_maybes(qid)
        
        # After any concrete decision, reset both tools back to Select
        if idx in (1, 2, 3):
            self._reset_tools_to_select()
        
        # Emit signal so other tabs can update (e.g., first-order evaluation sorting)
        if verdict:
            self.matchDecisionMade.emit(qid, verdict)
    
    def _update_single_query_state(self, qid: str) -> None:
        """Update the state color for a single query in the combo box."""
        idx = self.cmb_query.findText(qid)
        if idx >= 0:
            state = get_query_state(qid)
            model = self.cmb_query.model()
            model.setData(model.index(idx, 0), state, QUERY_STATE_ROLE)

    def _on_query_quality_saved(self, target: str, id_value: str) -> None:
        """Handle image quality saved for query."""
        # Optionally refresh metadata bar or trigger other updates
        self._refresh_meta_bar()

    def _on_gallery_quality_saved(self, target: str, id_value: str) -> None:
        """Handle image quality saved for gallery."""
        # Optionally refresh metadata bar or trigger other updates
        self._refresh_meta_bar()

    def _reset_tools_to_select(self, *_) -> None:
        """Ensure both sides are back on **Select** tool after picture/ID changes."""
        try: self.view_q.reset_tool_to_select()
        except Exception: pass
        try: self.view_g.reset_tool_to_select()
        except Exception: pass

    def _update_gallery_encounter_info(self, *_) -> None:
        """Update the gallery encounter info label with date and the query for this encounter."""
        try:
            gid = self.cmb_gallery.currentText() or ""
            current_path = self.view_g.current_path()
            
            if not gid or not current_path:
                self._gallery_encounter_info.setText("")
                return
            
            # Get encounter date from the current image path
            enc_date = get_encounter_date_from_path(current_path)
            date_str = format_encounter_date(enc_date) if enc_date else ""
            
            # Get queries whose observation date matches this encounter date
            matched_queries = get_queries_for_encounter(gid, enc_date)
            queries_str = format_queries_for_display(matched_queries, max_display=2)
            
            # Build display string
            if date_str and queries_str:
                self._gallery_encounter_info.setText(f"{date_str}  ←  {queries_str}")
            elif date_str:
                self._gallery_encounter_info.setText(date_str)
            elif queries_str:
                self._gallery_encounter_info.setText(f"←  {queries_str}")
            else:
                self._gallery_encounter_info.setText("")
        except Exception:
            self._gallery_encounter_info.setText("")

    def _open_id_folder(self, target: str, id_str: str) -> None:
        if not id_str:
            return
        folder = ap.root_for(target) / id_str  # archive path helper exists. :contentReference[oaicite:6]{index=6}
        try:
            if platform.system() == "Windows":
                os.startfile(str(folder))  # type: ignore[attr-defined]
            elif platform.system() == "Darwin":
                subprocess.call(["open", str(folder)])
            else:
                subprocess.call(["xdg-open", str(folder)])
        except Exception:
            pass

    # -------------------- metadata bar logic --------------------
    def _score_breakdown_for_pair(self, qid: str, gid: str) -> Dict[str, float]:
        """
        Compute per-field similarities in [0,1] for the selected (qid, gid),
        using the same scorers as First‑order. Fields absent on either side are omitted.
        """
        br: Dict[str, float] = {}
        if not qid or not gid:
            return br
        self._engine.rebuild_if_needed()  # no‑op after first build
        q_row = self._engine._queries_rows_by_id.get(qid, {})
        if not q_row:
            return br
        # Prepare query states per field
        q_states: Dict[str, object] = {}
        active = []
        for f, sc in self._engine.scorers.items():
            qs = sc.prepare_query(q_row)
            if sc.has_query_signal(qs):
                q_states[f] = qs
                active.append(f)
        # Score a single gallery id across active fields
        for f in active:
            sc = self._engine.scorers[f]
            s, present = sc.score_pair(q_states[f], gid)
            if present:
                br[f] = float(s)
        return br

    def _tooltip_for_field(self, field: str, q_val: str, g_val: str, s: float) -> str:
        # Same compact format as First‑order cards
        def _fmt(x: str) -> str:
            return (x or "").strip()
        return f"{field}\nQ: {_fmt(q_val)}\nG: {_fmt(g_val)}\nΔ score = {s:.3f}"

    def _refresh_meta_bar(self) -> None:
        # Clear previous (keep trailing stretch)
        while self._meta_hbox.count() > 1:
            item = self._meta_hbox.takeAt(0)
            w = item.widget()
            if w: w.deleteLater()

        qid = self.cmb_query.currentText() or ""
        gid = self.cmb_gallery.currentText() or ""
        if not qid or not gid:
            return

        try:
            self._engine.rebuild_if_needed()
            q_row = self._engine._queries_rows_by_id.get(qid, {}) or {}
            g_row = self._engine._gallery_rows_by_id.get(gid, {}) or {}
            br = self._score_breakdown_for_pair(qid, gid)

            for field in ALL_FIELDS:
                if field not in br:
                    continue
                s = br[field]
                cmap = FIELD_COLORMAPS.get(field, DEFAULT_CMAP)
                col = _interp_color(cmap, s)
                css_bg = _css_color(col)
                css_fg = _text_color_for_bg(col)
                label = QLabel(field)
                label.setStyleSheet(
                    f"background-color:{css_bg}; color:{css_fg};"
                    " border-radius:4px; padding:2px 6px; font-size:12px;"
                )
                label.setToolTip(self._tooltip_for_field(field, str(q_row.get(field, "")), str(g_row.get(field, "")), s))
                self._meta_hbox.insertWidget(self._meta_hbox.count() - 1, label)
        except Exception:
            # Never break the tab if engine data is missing
            pass

    # -------------------- verification display --------------------
    def _load_verification_lookup(self) -> None:
        """Load the verification lookup (lazy initialization)."""
        if self._verification_loaded:
            return
        
        try:
            self._verification_lookup = get_active_verification_lookup()
            self._verification_loaded = True
        except Exception:
            self._verification_lookup = None
            self._verification_loaded = True

    def _refresh_verification_display(self) -> None:
        """Update the verification score label for the current query-gallery pair."""
        # Ensure verification lookup is loaded
        if not self._verification_loaded:
            self._load_verification_lookup()
        
        qid = self.cmb_query.currentText() or ""
        gid = self.cmb_gallery.currentText() or ""
        
        if not qid or not gid or not self._verification_lookup:
            self.lbl_verification.setText("")
            return
        
        try:
            score = self._verification_lookup.get_score(qid, gid)
            
            if score is None:
                self.lbl_verification.setText("<i style='color:#888'>P(same): N/A</i>")
                self.lbl_verification.setToolTip(
                    "Verification score not available for this pair.\n"
                    "Run verification precomputation in Deep Learning tab."
                )
            else:
                optimal_thresh = get_optimal_threshold_for_active_model()
                color, tooltip = colorize_verification_score(score, optimal_thresh)
                
                self.lbl_verification.setText(
                    f"<b style='color:{color}'>P(same): {score:.2f}</b>"
                )
                self.lbl_verification.setToolTip(tooltip)
        except Exception:
            self.lbl_verification.setText("")
