# src/ui/tab_second_order.py
from __future__ import annotations

import json, os, platform, subprocess
from pathlib import Path
from typing import Dict

from PySide6.QtCore import Qt
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
from src.data.compare_labels import get_label_for_pair, save_label_for_pair
from src.data.merge_yes import is_query_silent

# Reuse First‑order’s field scorers + palette to build the bottom diff bar
from src.search.engine import FirstOrderSearchEngine, ALL_FIELDS
from src.ui.lineup_card import (
    DEFAULT_CMAP, FIELD_COLORMAPS, _interp_color, _css_color, _text_color_for_bg
)
from src.data.archive_paths import last_observation_for_all

class TabSecondOrder(QWidget):
    """
    Second-order compare tab:
      - Choose one Query ID and one Gallery ID (twin viewers)
      - "Recommended (from pins)" combo from First‑order pins
      - Decision combo (Yes/Maybe/No) + Notes + Save per (query_id, gallery_id)
      - Metadata diff bar at the bottom (fields color‑coded by similarity)

    Change requested:
      After saving a decision (Yes/No/Maybe) OR after a picture changes
      (by changing Query/Gallery ID or navigating), reset tools to Select.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # First‑order engine (lazy build) for field-level comparisons
        self._engine = FirstOrderSearchEngine()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        # ---- Row 1: IDs + Recommendations ----
        row1 = QHBoxLayout(); row1.setContentsMargins(0, 0, 0, 0)

        row1.addWidget(QLabel("Query:"))
        self.cmb_query = QComboBox(); self.cmb_query.setMinimumWidth(220)
        # Make combo editable for type-to-search functionality
        self.cmb_query.setEditable(True)
        self.cmb_query.setInsertPolicy(QComboBox.NoInsert)
        self.cmb_query.completer().setFilterMode(Qt.MatchContains)
        self.cmb_query.completer().setCompletionMode(QCompleter.PopupCompletion)
        row1.addWidget(self.cmb_query)

        row1.addSpacing(12)
        row1.addWidget(QLabel("Recommended (from pins):"))
        self.cmb_recommended = QComboBox(); self.cmb_recommended.setMinimumWidth(220)
        row1.addWidget(self.cmb_recommended)

        row1.addSpacing(12)
        row1.addWidget(QLabel("Gallery:"))
        self.cmb_gallery = QComboBox(); self.cmb_gallery.setMinimumWidth(220)
        # Make combo editable for type-to-search functionality
        self.cmb_gallery.setEditable(True)
        self.cmb_gallery.setInsertPolicy(QComboBox.NoInsert)
        self.cmb_gallery.completer().setFilterMode(Qt.MatchContains)
        self.cmb_gallery.completer().setCompletionMode(QCompleter.PopupCompletion)
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

        self.btn_best_q = QPushButton("Set Best (Query)")
        self.btn_best_g = QPushButton("Set Best (Gallery)")
        row2.addSpacing(12); row2.addWidget(self.btn_best_q)
        row2.addSpacing(4); row2.addWidget(self.btn_best_g)

        outer.addLayout(row2)

        # ---- Twin viewers ----
        self.split = QSplitter(Qt.Horizontal)
        self.view_q = AnnotatorViewSecond(target="Queries", title="Query")
        self.view_g = AnnotatorViewSecond(target="Gallery", title="Gallery")
        self.split.addWidget(self.view_q)
        self.split.addWidget(self.view_g)
        self.split.setStretchFactor(0, 1)
        self.split.setStretchFactor(1, 1)
        self.split.setHandleWidth(6)
        outer.addWidget(self.split, 1)

        # Reset tools to Select whenever either viewer changes picture
        # (covers: changing ID, switching encounter, prev/next buttons)
        #   AnnotatorViewSecond emits this in set_files() and after navigation. :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}
        self.view_q.currentImageChanged.connect(self._reset_tools_to_select)
        self.view_g.currentImageChanged.connect(self._reset_tools_to_select)

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

        # Repopulate with signals blocked to avoid spurious slot executions
        self.cmb_query.blockSignals(True)
        self.cmb_gallery.blockSignals(True)
        try:
            self.cmb_query.clear()
            self.cmb_gallery.clear()

            if qs:
                self.cmb_query.addItems(qs)
            if gs:
                self.cmb_gallery.addItems(gs)

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

    def _load_recommended_from_pins(self, qid: str) -> None:
        self.cmb_recommended.blockSignals(True)
        self.cmb_recommended.clear()
        if not qid:
            self.cmb_recommended.blockSignals(False); return
        try:
            p = self._pins_path(qid)
            pins = []
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                pins = list(dict.fromkeys([str(x) for x in (data.get("pinned") or [])]))
            self.cmb_recommended.addItem("—")
            if pins:
                self.cmb_recommended.addItems(pins)
        except Exception:
            self.cmb_recommended.addItem("—")
        finally:
            self.cmb_recommended.blockSignals(False)



    def _on_query_changed(self) -> None:
        qid = self.cmb_query.currentText()
        files = list_image_files("Queries", qid) if qid else []
        files = reorder_files_with_best("Queries", qid, files) if qid else files
        self.view_q.set_files(files)
        self._load_recommended_from_pins(qid)
        self._load_decision_ui()
        self._refresh_meta_bar()

    def _on_gallery_changed(self) -> None:
        gid = self.cmb_gallery.currentText()
        files = list_image_files("Gallery", gid) if gid else []
        files = reorder_files_with_best("Gallery", gid, files) if gid else files
        self.view_g.set_files(files)
        self._load_decision_ui()
        self._refresh_meta_bar()

    def _on_set_best_query(self) -> None:
        qid = self.cmb_query.currentText()
        if not qid or not self.view_q.strip.files:
            return
        idx = max(0, min(self.view_q.strip.idx, len(self.view_q.strip.files) - 1))
        save_best_for_id("Queries", qid, self.view_q.strip.files[idx])
        # Reload with best rolled to front
        files = reorder_files_with_best("Queries", qid, list(self.view_q.strip.files))
        self.view_q.set_files(files)

    def _on_set_best_gallery(self) -> None:
        gid = self.cmb_gallery.currentText()
        if not gid or not self.view_g.strip.files:
            return
        idx = max(0, min(self.view_g.strip.idx, len(self.view_g.strip.files) - 1))
        save_best_for_id("Gallery", gid, self.view_g.strip.files[idx])
        files = reorder_files_with_best("Gallery", gid, list(self.view_g.strip.files))
        self.view_g.set_files(files)

    def _on_recommended_changed(self) -> None:
        gid = self.cmb_recommended.currentText()
        if gid and gid != "—":
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
        save_label_for_pair(qid, gid, verdict, notes)
        # NEW: After any concrete decision, reset both tools back to Select
        if idx in (1, 2, 3):
            self._reset_tools_to_select()

    def _reset_tools_to_select(self, *_) -> None:
        """Ensure both sides are back on **Select** tool after picture/ID changes."""
        try: self.view_q.reset_tool_to_select()
        except Exception: pass
        try: self.view_g.reset_tool_to_select()
        except Exception: pass

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
