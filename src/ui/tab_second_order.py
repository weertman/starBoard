# src/ui/tab_second_order.py
from __future__ import annotations

import json, os, platform, subprocess
from pathlib import Path
from typing import Dict

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QSplitter, QLineEdit, QScrollArea, QSizePolicy, QFrame
)

from src.data.id_registry import list_ids
from src.data.image_index import list_image_files
from src.data import archive_paths as ap
from src.ui.annotator_view_second import AnnotatorViewSecond
from src.data.compare_labels import get_label_for_pair, save_label_for_pair
from src.data.merge_yes import is_query_silent

# NEW: reuse First‑order’s scoring + field inventory
from src.search.engine import FirstOrderSearchEngine, ALL_FIELDS   # fields + scorers

# NEW: reuse First‑order’s badge colormaps so colors match lineup cards
from src.ui.lineup_card import (                                   # color helpers / colormaps
    DEFAULT_CMAP, FIELD_COLORMAPS, _interp_color, _css_color, _text_color_for_bg
)


class TabSecondOrder(QWidget):
    """
    Second-order compare tab:
      - Choose one Query ID and one Gallery ID (twin viewers)
      - "Recommended (from pins)" combo from First‑order pins
      - Decision combo (Yes/Maybe/No) + Notes + Save per (query_id, gallery_id)
      - NEW: Metadata diff bar at the bottom (all fields, color‑coded by per‑field similarity)
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        # Lightweight First‑order engine for per‑field pair scoring
        self._engine = FirstOrderSearchEngine()  # builds lazily on first use

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        # ---- Row 1: IDs + Recommendations ----
        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)

        row1.addWidget(QLabel("Query:"))
        self.cmb_query = QComboBox(); self.cmb_query.setMinimumWidth(220)
        row1.addWidget(self.cmb_query)

        row1.addSpacing(12)
        row1.addWidget(QLabel("Recommended (from pins):"))
        self.cmb_recommended = QComboBox(); self.cmb_recommended.setMinimumWidth(220)
        row1.addWidget(self.cmb_recommended)

        row1.addSpacing(12)
        row1.addWidget(QLabel("Gallery:"))
        self.cmb_gallery = QComboBox(); self.cmb_gallery.setMinimumWidth(220)
        row1.addWidget(self.cmb_gallery)

        row1.addStretch(1)
        outer.addLayout(row1)

        # ---- Row 2: Decision + Notes + Open folders ----
        row2 = QHBoxLayout()
        row2.setContentsMargins(0, 0, 0, 0)

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

        # ---- NEW: bottom metadata diff bar (thin) ----
        self._meta_frame = QFrame()
        self._meta_frame.setObjectName("MetaBar")
        self._meta_frame.setFrameShape(QFrame.StyledPanel)
        self._meta_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._meta_frame.setMaximumHeight(44)

        meta_lay = QHBoxLayout(self._meta_frame)
        meta_lay.setContentsMargins(6, 4, 6, 4)
        meta_lay.setSpacing(6)

        self._meta_scroll = QScrollArea()
        self._meta_scroll.setWidgetResizable(True)
        self._meta_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._meta_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._meta_inner = QWidget()
        self._meta_hbox = QHBoxLayout(self._meta_inner)
        self._meta_hbox.setContentsMargins(0, 0, 0, 0)
        self._meta_hbox.setSpacing(4)
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

        # ---- Populate IDs ----
        self._refresh_ids()

    # ----- helpers -----
    def _refresh_ids(self) -> None:
        qs = [qid for qid in list_ids("Queries") if not is_query_silent(qid)]
        gs = list_ids("Gallery")
        self.cmb_query.blockSignals(True)
        self.cmb_gallery.blockSignals(True)
        self.cmb_query.clear()
        self.cmb_gallery.clear()
        self.cmb_query.addItems(qs)
        self.cmb_gallery.addItems(gs)
        self.cmb_query.blockSignals(False)
        self.cmb_gallery.blockSignals(False)

        if qs: self.cmb_query.setCurrentIndex(0)
        if gs: self.cmb_gallery.setCurrentIndex(0)

        self._on_query_changed()
        self._on_gallery_changed()

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
        self.view_q.set_files(files)
        self._load_recommended_from_pins(qid)
        self._load_decision_ui()
        self._refresh_meta_bar()

    def _on_gallery_changed(self) -> None:
        gid = self.cmb_gallery.currentText()
        files = list_image_files("Gallery", gid) if gid else []
        self.view_g.set_files(files)
        self._load_decision_ui()
        self._refresh_meta_bar()

    def _on_recommended_changed(self) -> None:
        gid = self.cmb_recommended.currentText()
        if gid and gid != "—":
            idx = self.cmb_gallery.findText(gid)
            if idx >= 0:
                self.cmb_gallery.setCurrentIndex(idx)
            else:
                self.cmb_gallery.addItem(gid)
                self.cmb_gallery.setCurrentIndex(self.cmb_gallery.count() - 1)
        # no matter what, refresh the bar (selection may not have changed if already selected)
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

    def _open_id_folder(self, target: str, id_str: str) -> None:
        if not id_str:
            return
        folder = ap.root_for(target) / id_str
        try:
            if platform.system() == "Windows":
                os.startfile(str(folder))
            elif platform.system() == "Darwin":
                subprocess.call(["open", str(folder)])
            else:
                subprocess.call(["xdg-open", str(folder)])
        except Exception:
            pass

    # -------------------- NEW: metadata bar logic --------------------

    def _score_breakdown_for_pair(self, qid: str, gid: str) -> Dict[str, float]:
        """
        Compute per-field similarities in [0,1] for the selected (qid, gid),
        using the same scorers as First‑order. Fields absent on either side are
        omitted from the dict.
        """
        br: Dict[str, float] = {}
        if not qid or not gid:
            return br
        # Ensure engine built and rows available
        self._engine.rebuild_if_needed()  # no-op after first build
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
        # Same rules/format as First‑order tooltips (Q/G/Δ + s=…).
        def _fmt(x: str) -> str:
            return (x or "").strip()

        if field in ("num_arms", "num_apparent_arms"):
            try:
                q = int(float(q_val)) if q_val else None
                g = int(float(g_val)) if g_val else None
            except Exception:
                q = g = None
            if q is not None and g is not None:
                d = g - q
                return f"{field}: Q={q}, G={g}, Δ={d}  |  s={s:.3f}"
            return f"{field}: Q={_fmt(q_val)}, G={_fmt(g_val)}  |  s={s:.3f}"

        if field in ("diameter_cm", "volume_ml"):
            unit = "cm" if field == "diameter_cm" else "ml"
            try:
                q = float(q_val) if q_val else None
                g = float(g_val) if g_val else None
            except Exception:
                q = g = None
            if q is not None and g is not None:
                d = g - q
                return f"{field}: Q={q:g}{unit}, G={g:g}{unit}, Δ={d:+g}{unit}  |  s={s:.3f}"
            return f"{field}: Q={_fmt(q_val)}{unit}, G={_fmt(g_val)}{unit}  |  s={s:.3f}"

        if field in ("sex", "disk color", "arm color"):
            return f"{field}: Q={_fmt(q_val)}, G={_fmt(g_val)}  |  s={s:.3f}"

        if field == "short_arm_codes":
            q = {t for t in (_fmt(q_val)).upper().replace(",", " ").split() if t}
            g = {t for t in (_fmt(g_val)).upper().replace(",", " ").split() if t}
            inter = sorted(q & g)
            qo = sorted(q - g)
            go = sorted(g - q)
            return (f"{field}: Q={{{', '.join(sorted(q))}}}, G={{{', '.join(sorted(g))}}} "
                    f"| shared={{{', '.join(inter)}}} q_only={{{', '.join(qo)}}} g_only={{{', '.join(go)}}}  |  s={s:.3f}")

        # Text / location: short snippets
        def _snip(x: str, n: int = 160) -> str:
            x = (x or "").strip().replace("\n", " ")
            return (x[:n] + "…") if len(x) > n else x

        return f"{field}: Q=“{_snip(q_val)}”, G=“{_snip(g_val)}”  |  s={s:.3f}"

    def _refresh_meta_bar(self) -> None:
        # Clear existing badges
        while self._meta_hbox.count() > 0:
            item = self._meta_hbox.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)

        qid = self.cmb_query.currentText() or ""
        gid = self.cmb_gallery.currentText() or ""

        # If not both selected, show placeholder
        if not qid or not gid:
            lbl = QLabel("Select a Query and a Gallery to compare metadata…")
            lbl.setStyleSheet("color:#666;")
            self._meta_hbox.addWidget(lbl)
            self._meta_hbox.addStretch(1)
            return

        # Build one-shot breakdown for (qid, gid)
        br = self._score_breakdown_for_pair(qid, gid)

        # Access raw rows for tooltips
        self._engine.rebuild_if_needed()
        q_row = self._engine._queries_rows_by_id.get(qid, {})
        g_row = self._engine._gallery_rows_by_id.get(gid, {})

        # Add a compact badge per field (ALL fields requested)
        for field in ALL_FIELDS:  # canonical order used in First‑order
            s = float(br.get(field, 0.0))
            cm = FIELD_COLORMAPS.get(field, DEFAULT_CMAP)
            # Map to [0..1] like LineupCard; if upstream already in [0..1], this is harmless.
            rgb = _interp_color(cm, max(0.0, min(1.0, (s + 1.0) / 2.0))) if field in br else (220, 220, 220)
            fg = _text_color_for_bg(rgb) if field in br else "#000"

            lbl = QLabel(field)
            lbl.setStyleSheet(
                "QLabel {"
                f"background: {_css_color(rgb)}; color: {fg};"
                "padding: 2px 6px; border-radius: 4px; border: 1px solid rgba(0,0,0,0.18);"
                "font-size: 11px; }"
            )

            tooltip = self._tooltip_for_field(
                field, q_row.get(field, "") or "", g_row.get(field, "") or "", s
            )
            if field not in br:
                tooltip += "\n(missing on one or both sides)"
            lbl.setToolTip(tooltip)

            self._meta_hbox.addWidget(lbl)

        self._meta_hbox.addStretch(1)
