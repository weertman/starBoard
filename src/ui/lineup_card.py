# src/ui/lineup_card.py
from __future__ import annotations
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QFrame,
    QSplitter, QWidget as _QW, QVBoxLayout as _QVL, QSizePolicy, QComboBox
)
from typing import Dict, List, Optional
from pathlib import Path
import platform, subprocess, os

from src.ui.image_strip import ImageStrip
from src.data.image_index import list_image_files
from src.data.compare_labels import get_label_for_pair, save_label_for_pair
from src.data.best_photo import reorder_files_with_best, save_best_for_id, find_best_index
from src.data.encounter_info import (
    get_encounter_date_from_path, format_encounter_date,
    get_queries_for_encounter, format_queries_for_display
)
from src.utils.interaction_logger import get_interaction_logger
from src.dl.verification_evaluation import (
    get_optimal_threshold_for_active_model, colorize_verification_score
)

Color = tuple[int, int, int]
Stop = tuple[float, Color]
DEFAULT_CMAP: List[Stop] = [(0.0,(165,0,38)),(0.5,(255,230,128)),(1.0,(0,104,55))]
FIELD_COLORMAPS: Dict[str, List[Stop]] = {}

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def _interp_color(stops: List[Stop], x: float) -> Color:
    v = max(0.0, min(1.0, float(x)))
    stops = sorted(stops, key=lambda s: s[0])
    for i in range(1, len(stops)):
        if v <= stops[i][0]:
            x0, c0 = stops[i-1]
            x1, c1 = stops[i]
            tt = 0.0 if x1 == x0 else (v - x0) / (x1 - x0)
            r = int(round(_lerp(c0[0], c1[0], tt)))
            g = int(round(_lerp(c0[1], c1[1], tt)))
            b = int(round(_lerp(c0[2], c1[2], tt)))
            return (r, g, b)
    return stops[-1][1]

def _text_color_for_bg(rgb: Color) -> str:
    r, g, b = rgb
    l = 0.299*r + 0.587*g + 0.114*b
    return "#000" if l >= 160 else "#fff"

def _css_color(rgb: Color) -> str:
    return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"


class LineupCard(QFrame):
    """
    Gallery result card with:
      - header (ID + score)
      - badge row (per-field contributions)
      - image strip (top of splitter)
      - footer with Pin + (optional) decision controls + Open Folder + View Metadata

    Notes:
    - Accepts field_breakdown as dict[field->score] (preferred) or list[str] (back-compat).
    - Exposes set_min_image_height()/set_strip_height() so the First-order tab can
      sync a minimum viewer height to the Query pane. Cards can grow taller than
      the viewport if badges/controls require it.
    """
    # Signal emitted when user clicks "View Metadata" - carries gallery_id
    metadataRequested = Signal(str)
    # Signal emitted when user clicks "Edit Metadata" - carries gallery_id
    editMetadataRequested = Signal(str)
    # Signal emitted when a decision is saved - carries (query_id, gallery_id, verdict)
    decisionSaved = Signal(str, str, str)

    def __init__(
        self,
        gallery_id: str,
        score: float,
        k_contrib: int,
        field_breakdown,  # Dict[str, float] preferred; list[str] allowed for back-compat
        field_tooltips: Dict[str, str] | None = None,
        *,
        query_id: str | None = None,
        closest_image_info: tuple[int, str] | None = None,  # (local_idx, cache_path)
        verification_score: Optional[float] = None,  # P(same) from verification model
        parent=None
    ):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setObjectName("LineupCard")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # --- back-compat: coerce list[str] -> dict[str,float]
        if not isinstance(field_breakdown, dict):
            try:
                field_breakdown = {str(f): 1.0 for f in (field_breakdown or [])}
            except Exception:
                field_breakdown = {}
        
        # Store closest image info: (index, path) for O(1) with path fallback
        self._closest_image_info = closest_image_info

        # Minimum desired height for the top image strip (synced from Query pane)
        self._min_image_h: int = 620
        self._min_image_w: int = 620
        self._last_vh: int = 0  # last gallery viewport height we fit to (for immediate re-apply)

        # For decision saving
        self.gallery_id: str = gallery_id
        self._query_id: str = (query_id or "").strip()
        self._ilog = get_interaction_logger()

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # --- Header (wrap for stable sizeHint)
        self._hdr = _QW()
        hbl = QHBoxLayout(self._hdr); hbl.setContentsMargins(0, 0, 0, 0)
        lbl_id = QLabel(f"<b>{gallery_id}</b>")
        lbl_score = QLabel(f"Score: {score:.3f}  <span style='color:#888'>({k_contrib} fields)</span>")
        hbl.addStretch(1); hbl.addWidget(lbl_id); hbl.addStretch(1); hbl.addWidget(lbl_score)
        
        # Add verification score display if available
        if verification_score is not None:
            hbl.addSpacing(8)
            optimal_thresh = get_optimal_threshold_for_active_model()
            verif_color, verif_tooltip = colorize_verification_score(
                verification_score, optimal_thresh
            )
            
            lbl_verif = QLabel(f"<b style='color:{verif_color}'>P(same): {verification_score:.2f}</b>")
            lbl_verif.setToolTip(verif_tooltip)
            hbl.addWidget(lbl_verif)
        
        root.addWidget(self._hdr)

        # --- Badges (wrap for stable sizeHint)
        self._badges = _QW()
        bxl = QHBoxLayout(self._badges); bxl.setContentsMargins(0, 0, 0, 0); bxl.setSpacing(4)
        for field in sorted(field_breakdown.keys()):
            s = float(field_breakdown[field])
            cm = FIELD_COLORMAPS.get(field, DEFAULT_CMAP)
            # Map similarities roughly from [-1..+1] to [0..1] if upstream uses cosine→[0..1], this is harmless
            rgb = _interp_color(cm, max(0.0, min(1.0, (s + 1.0) / 2.0)))
            lbl = QLabel(field)
            lbl.setStyleSheet(
                "QLabel {"
                f"background: {_css_color(rgb)}; color: {_text_color_for_bg(rgb)};"
                "padding: 2px 6px; border-radius: 4px; border: 1px solid rgba(0,0,0,0.18);"
                "font-size: 11px; }"
            )
            lbl.setToolTip((field_tooltips or {}).get(field, f"{field}: similarity {s:.3f}"))
            bxl.addWidget(lbl)
        bxl.addStretch(1)
        root.addWidget(self._badges)

        # --- Encounter info row (date and matched queries)
        self._encounter_info = QLabel("")
        self._encounter_info.setStyleSheet(
            "QLabel { color: #e67e22; font-size: 11px; font-style: italic; }"
        )
        self._encounter_info.setToolTip(
            "Shows encounter date and query that led to this encounter"
        )
        root.addWidget(self._encounter_info)

        # --- Splitter: image (top) / footer (bottom)
        self.split = QSplitter(Qt.Vertical)
        self.split.setOpaqueResize(True)
        self.split.setChildrenCollapsible(False)
        self.split.setHandleWidth(6)

        # top: image strip
        files = list_image_files("Gallery", gallery_id)
        
        # When closest_image_info is provided, skip reordering and use the
        # original file order from list_image_files() for index-based lookup.
        # Pass initial_idx directly to ImageStrip to avoid double image loading.
        if self._closest_image_info is not None:
            local_idx, cache_path = self._closest_image_info
            # Find where the best photo is in the original order
            best_idx = find_best_index("Gallery", gallery_id, files)
            # Pass initial_idx, best_idx, and closest_idx for toggle behavior
            self.strip = ImageStrip(
                files=files, 
                long_edge=512, 
                initial_idx=local_idx, 
                best_idx=best_idx,
                closest_idx=local_idx  # Enable toggle between best and closest
            )
        else:
            # Reorder so best is first, then best_idx = 0, no toggle needed
            files = reorder_files_with_best("Gallery", gallery_id, files)
            self.strip = ImageStrip(files=files, long_edge=512, best_idx=0)
        
        topw = _QW(); tl = _QVL(topw); tl.setContentsMargins(0, 0, 0, 0); tl.setSpacing(4)
        tl.addWidget(self.strip, 1)

        # bottom: footer
        self._foot = _QW()
        fl = _QVL(self._foot); fl.setContentsMargins(0, 0, 0, 0); fl.setSpacing(6)
        footer = QHBoxLayout(); footer.setContentsMargins(0, 0, 0, 0); footer.setSpacing(8)

        self.btn_pin = QPushButton("Pin")

        # Decision UI (enabled only if query_id is provided)
        self.lbl_decision = QLabel("Decision:")
        self.cmb_verdict = QComboBox()
        self.cmb_verdict.addItems(["", "Match", "Maybe", "No"])
        self.btn_save_decision = QPushButton("Save")

        self.btn_open = QPushButton("Open Folder")
        self.btn_best = QPushButton("Best")
        self.btn_meta = QPushButton("View Metadata")
        self.btn_edit_meta = QPushButton("Edit Metadata")
        self.btn_open.clicked.connect(self._open_folder)
        self.btn_best.clicked.connect(self._on_set_best_gallery)
        self.btn_meta.clicked.connect(lambda: self.metadataRequested.emit(self.gallery_id))
        self.btn_edit_meta.clicked.connect(lambda: self.editMetadataRequested.emit(self.gallery_id))

        footer.addWidget(self.btn_pin)
        footer.addWidget(self.btn_best)
        footer.addSpacing(8)
        footer.addWidget(self.lbl_decision)
        footer.addWidget(self.cmb_verdict)
        footer.addWidget(self.btn_save_decision)
        footer.addStretch(1)
        footer.addWidget(self.btn_meta)
        footer.addWidget(self.btn_edit_meta)
        footer.addWidget(self.btn_open)
        fl.addLayout(footer)

        self.split.addWidget(topw)
        self.split.addWidget(self._foot)
        self.split.setStretchFactor(0, 1)
        self.split.setStretchFactor(1, 0)
        root.addWidget(self.split, 1)

        # Decision enablement
        if not self._query_id:
            for w in (self.lbl_decision, self.cmb_verdict, self.btn_save_decision):
                w.setEnabled(False)
                w.setToolTip("Select a Query to enable decisions.")
        else:
            self.btn_save_decision.clicked.connect(self._on_save_decision)
            self._load_decision_state()
        
        # Connect to image strip navigation for dynamic encounter info updates
        self.strip.btn_prev.clicked.connect(self._update_encounter_info)
        self.strip.btn_next.clicked.connect(self._update_encounter_info)
        self.strip.btn_best.clicked.connect(self._update_encounter_info)
        
        # Initial update
        self._update_encounter_info()

    # ---------------- Sizing API ----------------
    def fit_to_viewport_height(self, viewport_h: int):
        """
        Size this card relative to the gallery viewport, but NEVER shrink the image area
        below self._min_image_h (synced to the Query viewer by the tab).
        Allow the card to grow taller than the viewport when badges/controls need space.
        """
        viewport_h = max(120, int(viewport_h))
        self._last_vh = viewport_h

        # Measure "chrome"
        footer_h = self._foot.sizeHint().height()
        header_h = self._hdr.sizeHint().height()
        badges_h = self._badges.sizeHint().height()
        margins = self.layout().contentsMargins()
        spacing = self.layout().spacing() * 2  # hdr↔badges, badges↔split
        chrome = header_h + badges_h + margins.top() + margins.bottom() + spacing + footer_h

        min_top = max(80, int(self._min_image_h))
        required_total = chrome + min_top

        # Do NOT clamp to the viewport: permit vertical growth as needed
        self.setMinimumHeight(required_total)
        self.setMaximumHeight(16777215)

        # Give the image area as much as we can, but keep at least min_top
        top_h = max(min_top, viewport_h - chrome)

        self.strip.set_view_height(int(top_h))
        self.split.setSizes([int(top_h), int(footer_h)])

    def set_min_image_height(self, h: int) -> None:
        """Update the minimum image-area height and re-apply layout if we know a viewport."""
        try:
            self._min_image_h = max(80, int(h))
        except Exception:
            return
        vh = self._last_vh if self._last_vh > 0 else (self.parent().height() if self.parent() else 0)
        if vh:
            try:
                self.fit_to_viewport_height(int(vh))
            except Exception:
                pass

    def set_min_image_width(self, w: int) -> None:
        """Update the minimum image-area width."""
        try:
            self._min_image_w = max(100, int(w))
            if hasattr(self.strip, "set_view_min_width"):
                self.strip.set_view_min_width(self._min_image_w)
        except Exception:
            pass

    def set_strip_width(self, w: int) -> None:
        self.set_min_image_width(w)

    # Back-compat alias used by TabFirstOrder._on_query_split_resized(). :contentReference[oaicite:4]{index=4}
    def set_strip_height(self, h: int) -> None:
        self.set_min_image_height(h)

    # ---------------- Encounter Info ----------------
    def _update_encounter_info(self) -> None:
        """Update the encounter info label with date and the query for this encounter."""
        try:
            files = self.strip.files
            if not files:
                self._encounter_info.setText("")
                return
            
            idx = max(0, min(self.strip.idx, len(files) - 1))
            current_path = files[idx]
            
            # Get encounter date from the current image path
            enc_date = get_encounter_date_from_path(current_path)
            date_str = format_encounter_date(enc_date) if enc_date else ""
            
            # Get queries whose observation date matches this encounter date
            matched_queries = get_queries_for_encounter(self.gallery_id, enc_date)
            queries_str = format_queries_for_display(matched_queries, max_display=2)
            
            # Build display string
            if date_str and queries_str:
                self._encounter_info.setText(f"{date_str}  ←  {queries_str}")
            elif date_str:
                self._encounter_info.setText(date_str)
            elif queries_str:
                self._encounter_info.setText(f"←  {queries_str}")
            else:
                self._encounter_info.setText("")
        except Exception:
            self._encounter_info.setText("")

    # ---------------- Actions ----------------
    def _open_folder(self):
        try:
            if self.strip.files:
                self._ilog.log("file_open", "btn_open_gallery", value=self.gallery_id)
                folder = str(Path(self.strip.files[0]).parent.parent)
                if platform.system() == "Windows":
                    os.startfile(folder)
                elif platform.system() == "Darwin":
                    subprocess.call(["open", folder])
                else:
                    subprocess.call(["xdg-open", folder])
        except Exception:
            pass

    # ---------------- Decision helpers ----------------
    def _load_decision_state(self) -> None:
        """Load latest saved verdict for (query_id, gallery_id)."""
        idx = 0
        try:
            prev = get_label_for_pair(self._query_id, self.gallery_id)
            if prev:
                v = (prev.get("verdict") or "").strip().lower()
                # ["", "Match", "Maybe", "No"] → map
                if v == "yes":   idx = 1
                elif v == "maybe": idx = 2
                elif v == "no":    idx = 3
        except Exception:
            pass
        self.cmb_verdict.setCurrentIndex(idx)

    def _on_save_decision(self):
        idx = self.cmb_verdict.currentIndex()
        if idx == 0 or not self._query_id:
            return
        verdict = "yes" if idx == 1 else ("maybe" if idx == 2 else "no")
        self._ilog.log("decision_save", "lineup_card", value=verdict,
                      context={"query_id": self._query_id, "gallery_id": self.gallery_id})
        notes = ""
        try:
            prev = get_label_for_pair(self._query_id, self.gallery_id)
            if prev:
                notes = prev.get("notes", "") or ""
        except Exception:
            pass
        try:
            save_label_for_pair(self._query_id, self.gallery_id, verdict, notes)
            # Emit signal to notify parent that decision was saved
            self.decisionSaved.emit(self._query_id, self.gallery_id, verdict)
        finally:
            old = self.btn_save_decision.text()
            self.btn_save_decision.setText("Saved ✓")
            self.btn_save_decision.setEnabled(False)
            QTimer.singleShot(
                1000,
                lambda: (self.btn_save_decision.setText(old), self.btn_save_decision.setEnabled(True))
            )

    def _on_set_best_gallery(self) -> None:
        """Persist current image in this card as 'best' for its gallery ID and roll order."""
        try:
            files = self.strip.files or []
            if not files:
                return
            idx = max(0, min(self.strip.idx, len(files) - 1))
            self._ilog.log("button_click", "btn_best_gallery", value=self.gallery_id,
                          context={"image_idx": idx})
            save_best_for_id("Gallery", self.gallery_id, files[idx])
            new_files = reorder_files_with_best("Gallery", self.gallery_id, files)
            self.strip.set_files(new_files)
            # Update encounter info after file change
            self._update_encounter_info()
        except Exception:
            # Never break the UI
            pass
