# src/ui/vis_past_matches.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
from pathlib import Path
import os
import re
from datetime import datetime, date as date_type

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox,
    QTableWidget, QTableWidgetItem, QWidget, QSpinBox, QFileDialog,
    QCheckBox, QScrollArea, QProgressBar, QApplication
)
from PySide6.QtCore import Qt

from src.ui.mpl_embed import MplWidget
from src.data.archive_paths import archive_root
from src.data.past_matches import PastMatchesDataset
from src.utils.interaction_logger import get_interaction_logger
from src.data.id_registry import list_ids
from src.ui.query_state_delegate import get_query_state, QueryState
from src.data.image_index import list_image_files
from src.data.best_photo import reorder_files_with_best

COLORS = {"yes": "#1b9e77", "maybe": "#d95f02", "no": "#d73027"}


class PastMatchesDialogBase(QDialog):
    def __init__(self, title: str, ds: PastMatchesDataset, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.resize(880, 600)
        self._ds = ds
        self._ilog = get_interaction_logger()
        self._reports_dir = archive_root() / "reports" / "figures"
        self._reports_dir.mkdir(parents=True, exist_ok=True)

        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(8, 8, 8, 8)
        self._root.setSpacing(6)

        top = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh")
        self.btn_png = QPushButton("Export PNG")
        self.btn_csv = QPushButton("Export CSV")
        self.lbl_hint = QLabel("")
        self.lbl_hint.setStyleSheet("color:#666")
        top.addWidget(self.btn_refresh)
        top.addSpacing(6)
        top.addWidget(self.btn_png)
        top.addWidget(self.btn_csv)
        top.addStretch(1)
        top.addWidget(self.lbl_hint)
        self._root.addLayout(top)

        self.btn_refresh.clicked.connect(self._on_refresh)
        self.btn_png.clicked.connect(self._on_export_png)
        self.btn_csv.clicked.connect(self._on_export_csv)

        self._build_body()
        self.render()

    def _build_body(self):
        self._chart = MplWidget(self)
        self._root.addWidget(self._chart, 1)

    def render(self):  # override
        self._chart.clear()
        self._chart.ax.text(0.5, 0.5, "No content", ha="center", va="center")
        self._chart.draw()

    def export_rows(self) -> Tuple[List[str], List[List[str]]]:  # override
        return [], []

    def update_dataset(self, ds: PastMatchesDataset):
        self._ds = ds
        self.render()

    def _on_refresh(self):
        self._ilog.log("button_click", f"btn_refresh_{self.windowTitle().lower().replace(' ', '_')}", value="clicked")
        self.render()

    def _on_export_png(self):
        self._ilog.log("button_click", f"btn_export_png_{self.windowTitle().lower().replace(' ', '_')}", value="clicked")
        if not getattr(self._chart, "fig", None):
            return
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        fname = f"{self.windowTitle().lower().replace(' ', '_')}_{ts}.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save figure", str(self._reports_dir / fname), "PNG (*.png)"
        )
        if not path:
            return
        try:
            self._chart.fig.savefig(path, dpi=150)
            self.lbl_hint.setText(f"Saved {os.path.basename(path)}")
        except Exception as e:
            self.lbl_hint.setText(f"Save failed: {e}")

    def _on_export_csv(self):
        self._ilog.log("button_click", f"btn_export_csv_{self.windowTitle().lower().replace(' ', '_')}", value="clicked")
        header, rows = self.export_rows()
        if not header:
            return
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        fname = f"{self.windowTitle().lower().replace(' ', '_')}_{ts}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", str(self._reports_dir / fname), "CSV (*.csv)"
        )
        if not path:
            return
        try:
            import csv
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(rows)
            self.lbl_hint.setText(f"Saved {os.path.basename(path)}")
        except Exception as e:
            self.lbl_hint.setText(f"Save failed: {e}")


class TotalsDialog(PastMatchesDialogBase):
    def __init__(self, ds: PastMatchesDataset, parent=None):
        super().__init__("Totals", ds, parent)

    def render(self):
        c = self._ds.totals_by_verdict
        self._chart.clear()
        ax = self._chart.ax
        x = ["yes", "maybe", "no"]
        y = [int(c.get(k, 0)) for k in x]
        colors = [COLORS[k] for k in x]
        bars = ax.bar(x, y, color=colors, edgecolor="#333")
        for b, v in zip(bars, y):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v}",
                    ha="center", va="bottom", fontsize=10)
        ax.set_ylabel("Count")
        ax.set_title("Past matches: totals")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        self._chart.draw()

    def export_rows(self):
        c = self._ds.totals_by_verdict
        return ["verdict", "count"], [[k, int(c.get(k, 0))] for k in ("yes", "maybe", "no")]


class TimelineDialog(PastMatchesDialogBase):
    def __init__(self, ds: PastMatchesDataset, parent=None):
        self._granularity = "Day"
        super().__init__("Timeline", ds, parent)

    def _build_body(self):
        row = QHBoxLayout()
        row.addWidget(QLabel("Granularity:"))
        self.cmb = QComboBox()
        self.cmb.addItems(["Day", "Week", "Month"])
        row.addWidget(self.cmb)
        row.addStretch(1)
        self._root.addLayout(row)
        self.cmb.currentIndexChanged.connect(self.render)
        self._chart = MplWidget(self)
        self._root.addWidget(self._chart, 1)

    def render(self):
        self._granularity = self.cmb.currentText()
        self._chart.clear()
        ax = self._chart.ax
        series = self._aggregated()
        xs = sorted(series.keys())
        y_yes = [series[d]["yes"] for d in xs]
        y_maybe = [series[d]["maybe"] for d in xs]
        y_no = [series[d]["no"] for d in xs]
        ax.plot(xs, y_yes, marker="o", label="yes", color=COLORS["yes"])
        ax.plot(xs, y_maybe, marker="o", label="maybe", color=COLORS["maybe"])
        ax.plot(xs, y_no, marker="o", label="no", color=COLORS["no"])
        ax.set_ylabel("Count")
        ax.set_title(f"Decisions over time ({self._granularity.lower()})")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)
        ax.margins(x=0.02)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        self._chart.draw()

    def _aggregated(self) -> Dict[str, Dict[str, int]]:
        key = {"Day": "decided_day", "Week": "decided_week", "Month": "decided_month"}[self._granularity]
        from collections import defaultdict, Counter
        agg: Dict[str, Counter] = defaultdict(Counter)
        for r in self._ds.records:
            k = getattr(r, key) if hasattr(r, key) else (r.q_meta.get(key, "") or "")
            if not k:
                continue
            agg[k][r.verdict] += 1
        out: Dict[str, Dict[str, int]] = {}
        for k, c in agg.items():
            out[k] = {"yes": c["yes"], "maybe": c["maybe"], "no": c["no"]}
        return out

    def export_rows(self):
        series = self._aggregated()
        xs = sorted(series.keys())
        rows = [[k, series[k]["yes"], series[k]["maybe"], series[k]["no"],
                 series[k]["yes"] + series[k]["maybe"] + series[k]["no"]] for k in xs]
        return [self._granularity.lower(), "yes", "maybe", "no", "total"], rows


class ByQueryDialog(PastMatchesDialogBase):
    def __init__(self, ds: PastMatchesDataset, parent=None):
        self._topk = 20
        super().__init__("By Query", ds, parent)

    def _build_body(self):
        row = QHBoxLayout()
        row.addWidget(QLabel("Top-K:"))
        from PySide6.QtWidgets import QSpinBox
        self.spin = QSpinBox()
        self.spin.setRange(1, 1000)
        self.spin.setValue(20)
        row.addWidget(self.spin)
        row.addStretch(1)
        self._root.addLayout(row)
        self.spin.valueChanged.connect(self.render)

        self._chart = MplWidget(self)
        self._root.addWidget(self._chart, 2)

        self._table = QTableWidget(0, 5, self)
        self._table.setHorizontalHeaderLabels(["query_id", "yes", "maybe", "no", "total"])
        self._table.setSortingEnabled(True)
        self._root.addWidget(self._table, 1)

    def render(self):
        self._topk = int(self.spin.value())
        counts = self._ds.per_query_counts
        keys = sorted(
            counts.keys(),
            key=lambda q: (-counts[q]["yes"], -counts[q]["total"], -counts[q]["maybe"], q),
        )
        top = keys[: self._topk]

        self._chart.clear()
        ax = self._chart.ax
        ys = [counts[q]["yes"] for q in top]
        ax.barh(range(len(top)), ys, color=COLORS["yes"])
        ax.set_yticks(range(len(top)), labels=top)
        ax.invert_yaxis()
        ax.set_xlabel("Yes count")
        ax.set_title("Top queries by 'yes'")
        for i, v in enumerate(ys):
            ax.text(v + 0.2, i, str(v), va="center")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        self._chart.draw()

        self._table.setRowCount(len(top))
        for i, q in enumerate(top):
            c = counts[q]
            row_vals = [q, c["yes"], c["maybe"], c["no"], c["total"]]
            for j, v in enumerate(row_vals):
                self._table.setItem(i, j, QTableWidgetItem(str(v)))

    def export_rows(self):
        counts = self._ds.per_query_counts
        keys = sorted(
            counts.keys(),
            key=lambda q: (-counts[q]["yes"], -counts[q]["total"], -counts[q]["maybe"], q),
        )
        rows = [[q, counts[q]["yes"], counts[q]["maybe"], counts[q]["no"], counts[q]["total"]] for q in keys]
        return ["query_id", "yes", "maybe", "no", "total"], rows


class ByGalleryDialog(PastMatchesDialogBase):
    def __init__(self, ds: PastMatchesDataset, parent=None):
        self._topk = 20
        super().__init__("By Gallery", ds, parent)

    def _build_body(self):
        row = QHBoxLayout()
        row.addWidget(QLabel("Top-K:"))
        from PySide6.QtWidgets import QSpinBox
        self.spin = QSpinBox()
        self.spin.setRange(1, 1000)
        self.spin.setValue(20)
        row.addWidget(self.spin)
        row.addStretch(1)
        self._root.addLayout(row)
        self.spin.valueChanged.connect(self.render)

        self._chart = MplWidget(self)
        self._root.addWidget(self._chart, 2)

        self._table = QTableWidget(0, 5, self)
        self._table.setHorizontalHeaderLabels(["gallery_id", "yes", "maybe", "no", "total"])
        self._table.setSortingEnabled(True)
        self._root.addWidget(self._table, 1)

    def render(self):
        self._topk = int(self.spin.value())
        counts = self._ds.per_gallery_counts
        keys = sorted(
            counts.keys(),
            key=lambda g: (-counts[g]["yes"], -counts[g]["total"], -counts[g]["maybe"], g),
        )
        top = keys[: self._topk]

        self._chart.clear()
        ax = self._chart.ax
        ys = [counts[g]["yes"] for g in top]
        ax.barh(range(len(top)), ys, color=COLORS["yes"])
        ax.set_yticks(range(len(top)), labels=top)
        ax.invert_yaxis()
        ax.set_xlabel("Yes count")
        ax.set_title("Top gallery IDs by 'yes'")
        for i, v in enumerate(ys):
            ax.text(v + 0.2, i, str(v), va="center")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        self._chart.draw()

        self._table.setRowCount(len(top))
        for i, g in enumerate(top):
            c = counts[g]
            row_vals = [g, c["yes"], c["maybe"], c["no"], c["total"]]
            for j, v in enumerate(row_vals):
                self._table.setItem(i, j, QTableWidgetItem(str(v)))

    def export_rows(self):
        counts = self._ds.per_gallery_counts
        keys = sorted(
            counts.keys(),
            key=lambda g: (-counts[g]["yes"], -counts[g]["total"], -counts[g]["maybe"], g),
        )
        rows = [[g, counts[g]["yes"], counts[g]["maybe"], counts[g]["no"], counts[g]["total"]] for g in keys]
        return ["gallery_id", "yes", "maybe", "no", "total"], rows


INFLOW_COLORS = {
    "matched": "#1b9e77",       # Green (same as "yes")
    "attempted": "#d95f02",      # Orange (same as PINNED/ATTEMPTED background)
    "not_attempted": "#7570b3",  # Muted purple
}


class QueryInflowDialog(PastMatchesDialogBase):
    """
    Stacked bar chart showing query inflow by first observation date,
    segmented into matched, attempted (with pins/maybes/nos), and not attempted.
    """
    def __init__(self, ds: PastMatchesDataset, parent=None):
        super().__init__("Query Inflow", ds, parent)

    def render(self):
        data = self._ds.queries_by_first_obs
        self._chart.clear()
        ax = self._chart.ax

        if not data:
            ax.text(0.5, 0.5, "No query observation data available",
                    ha="center", va="center", fontsize=12)
            self._chart.draw()
            return

        # Sort dates chronologically
        dates = sorted(data.keys())
        matched = [data[d].get("matched", 0) for d in dates]
        attempted = [data[d].get("attempted", 0) for d in dates]
        not_attempted = [data[d].get("not_attempted", 0) for d in dates]

        x = range(len(dates))
        bar_width = 0.8

        # Stacked bar: matched on bottom, attempted in middle, not_attempted on top
        ax.bar(x, matched, bar_width, label="Matched",
               color=INFLOW_COLORS["matched"], edgecolor="#333")

        # Calculate bottom for attempted bars (on top of matched)
        bottom_attempted = matched

        ax.bar(x, attempted, bar_width, bottom=bottom_attempted,
               label="Attempted",
               color=INFLOW_COLORS["attempted"], edgecolor="#333")

        # Calculate bottom for not_attempted bars (on top of matched + attempted)
        bottom_not_attempted = [m + a for m, a in zip(matched, attempted)]

        ax.bar(x, not_attempted, bar_width, bottom=bottom_not_attempted,
               label="Not Attempted",
               color=INFLOW_COLORS["not_attempted"], edgecolor="#333")

        # Add total labels on top of each bar
        for i, (m, a, na) in enumerate(zip(matched, attempted, not_attempted)):
            total = m + a + na
            if total > 0:
                ax.text(i, total + 0.1, str(total), ha="center", va="bottom", fontsize=8)

        ax.set_xticks(list(x))
        ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Query Count")
        ax.set_xlabel("First Observation Date")
        ax.set_title("Query Inflow by First Observation Date")
        ax.legend(loc="upper right")
        ax.margins(x=0.02)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Adjust layout to prevent label clipping
        self._chart.fig.tight_layout()
        self._chart.draw()

    def export_rows(self):
        data = self._ds.queries_by_first_obs
        dates = sorted(data.keys())
        rows = []
        for d in dates:
            matched = data[d].get("matched", 0)
            attempted = data[d].get("attempted", 0)
            not_attempted = data[d].get("not_attempted", 0)
            total = matched + attempted + not_attempted
            rows.append([d, matched, attempted, not_attempted, total])
        return ["date", "matched", "attempted", "not_attempted", "total"], rows


# ---------------------------------------------------------------------------
# Outing Stats Dialog
# ---------------------------------------------------------------------------

# Patterns for extracting outing DESCRIPTION from query folder names
# We no longer rely on these for dates - dates come from encounter folders
_OUTING_PATTERN_A = re.compile(r"^(\d+)__(\d{1,2})_(\d{1,2})_(\d{4})_(.+)$")  # index__mm_dd_yyyy_desc
_OUTING_PATTERN_C = re.compile(r"^(\d{1,2})_(\d{1,2})_(\d{4})_(.+)_(\d+)$")   # mm_dd_yyyy_desc_index
_OUTING_PATTERN_B = re.compile(r"^(\d{1,2})_(\d{1,2})_(\d{4})_(.+?)(\d+)$")   # mm_dd_yyyy_descINDEX
_OUTING_PATTERN_D = re.compile(r"^(\d+)_(\d{1,2})_(\d{4})_(.+)$")             # index_??_yyyy_desc
_OUTING_PATTERN_E = re.compile(r"^(\d+)_(.+)$")                               # index_desc (minimal)

# First-order search tool widgets that indicate the user worked on a query
_FIRST_ORDER_SEARCH_WIDGETS = frozenset({
    "btn_gallery_next", "btn_gallery_prev", "btn_refresh", "btn_refresh_visual",
    "btn_rebuild", "chk_visual", "chk_roll_to_closest", "btn_pin", "lineup_card",
    "btn_best_query", "btn_save_image_quality", "spin_topk", "cmb_preset",
})


def _extract_outing_descriptor(folder_name: str) -> str:
    """
    Extract just the outing descriptor from a query folder name.
    This is used for labeling/grouping, NOT for date extraction.
    
    Examples:
        "0__9_10_2025_release_day" -> "release_day"
        "11_16_2025_twelveweeks_0" -> "twelveweeks"
        "0_13_2026_fivemonths" -> "fivemonths"
    """
    # Format A: {index}__{mm}_{dd}_{yyyy}_{desc}
    m = _OUTING_PATTERN_A.match(folder_name)
    if m:
        return m.group(5)
    
    # Format C: {mm}_{dd}_{yyyy}_{desc}_{index}
    m = _OUTING_PATTERN_C.match(folder_name)
    if m:
        return m.group(4)
    
    # Format B: {mm}_{dd}_{yyyy}_{desc}{index}
    m = _OUTING_PATTERN_B.match(folder_name)
    if m:
        return m.group(4)
    
    # Format D: {index}_{??}_{yyyy}_{desc}
    m = _OUTING_PATTERN_D.match(folder_name)
    if m:
        return m.group(4)
    
    # Format E: {index}_{desc} (minimal fallback)
    m = _OUTING_PATTERN_E.match(folder_name)
    if m:
        return m.group(2)
    
    # Last resort: use whole folder name
    return folder_name


def _parse_query_folder_to_outing(folder_name: str, query_id: str = None) -> Optional[Tuple[str, str]]:
    """
    Get outing info for a query using ENCOUNTER DATES as the source of truth.
    
    Returns (date_str, outing_name) or None if no observation date found.
    
    The date comes from the encounter subfolder (reliable MM_DD_YY format).
    The outing_name combines the date with the descriptor for unique grouping.
    
    This approach is robust because:
    - Dates come from encounter folders which always have MM_DD_YY format
    - Query folder names are only used for the descriptor/label
    """
    qid = query_id or folder_name
    
    # Get the observation date from encounter subfolders (the reliable source)
    try:
        from src.data.observation_dates import first_observation_date
        first_obs = first_observation_date("Queries", qid)
        if first_obs is None:
            return None
        
        date_str = first_obs.isoformat()
        mm, dd, yyyy = first_obs.month, first_obs.day, first_obs.year
        
        # Extract the descriptor from the folder name (just for labeling)
        descriptor = _extract_outing_descriptor(folder_name)
        
        # Create outing name: date + descriptor for unique identification
        outing_name = f"{mm}_{dd}_{yyyy}_{descriptor}"
        
        return (date_str, outing_name)
    except Exception:
        return None


def _load_queries_attempted_via_logs() -> set:
    """
    Scan interaction logs to find queries that were viewed in First-order
    AND had search tool interactions (btn_gallery_next, btn_refresh, etc.).
    
    Returns a set of query_ids that were "attempted" via logs.
    """
    try:
        from src.ui.vis_interaction_logs import load_all_interaction_logs
        events = load_all_interaction_logs()
    except Exception:
        return set()
    
    if not events:
        return set()
    
    # Track which query was active during First-order interactions
    attempted_queries = set()
    current_query_in_first_order: Optional[str] = None
    queries_with_search_interaction: set = set()
    
    for event in events:
        tab = event.get("tab", "") or ""
        widget = event.get("widget", "") or ""
        value = event.get("value", "") or ""
        
        # Track when user selects a query in First-order
        if widget == "cmb_query" and "First" in tab and value:
            current_query_in_first_order = value
        
        # Track tab switches away from First-order
        if widget == "main_tabs" and "First" not in value:
            current_query_in_first_order = None
        
        # If user interacts with search tools while a query is selected
        if current_query_in_first_order and widget in _FIRST_ORDER_SEARCH_WIDGETS:
            queries_with_search_interaction.add(current_query_in_first_order)
    
    return queries_with_search_interaction


def _compute_outing_stats() -> Dict[str, Dict[str, any]]:
    """
    Compute observation counts per outing, classified by identification state.
    
    Uses fallback to interaction logs if a query has no pins/labels but
    was viewed and worked on in First-order.
    
    Returns:
        {outing_name: {"date": str, "matched": int, "attempted": int, "not_attempted": int}}
    """
    outings: Dict[str, Dict[str, any]] = defaultdict(
        lambda: {"date": "", "matched": 0, "attempted": 0, "not_attempted": 0}
    )
    
    query_ids = list_ids("Queries")
    
    # Load interaction log fallback data
    queries_attempted_via_logs = _load_queries_attempted_via_logs()
    
    for qid in query_ids:
        parsed = _parse_query_folder_to_outing(qid, query_id=qid)
        if parsed is None:
            continue
        
        date_str, outing_name = parsed
        state = get_query_state(qid)
        
        if not outings[outing_name]["date"]:
            outings[outing_name]["date"] = date_str
        
        if state == QueryState.MATCHED:
            outings[outing_name]["matched"] += 1
        elif state in (QueryState.PINNED, QueryState.ATTEMPTED):
            outings[outing_name]["attempted"] += 1
        elif qid in queries_attempted_via_logs:
            # Fallback: interaction logs show user worked on this query
            outings[outing_name]["attempted"] += 1
        else:
            outings[outing_name]["not_attempted"] += 1
    
    return dict(outings)


class OutingStatsDialog(QDialog):
    """
    Visualization dialog for observation counts per outing,
    segmented into matched, attempted, and not_attempted.
    
    Features:
    - Outing filter table to toggle outings on/off
    - Multiple chart types: Line / Stacked Bar
    - X-axis modes: Time-proportional / Categorical (equal spacing)
    - Optional regression lines (total or matched only)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Outing Stats")
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.resize(1100, 800)
        self._ilog = get_interaction_logger()
        self._reports_dir = archive_root() / "reports" / "figures"
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        self._outing_data: Dict[str, Dict[str, any]] = {}
        self._excluded_outings: set = set()
        
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(8, 8, 8, 8)
        self._root.setSpacing(6)
        
        # ---- Top bar: Refresh, Export ----
        top = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh")
        self.btn_png = QPushButton("Export PNG")
        self.btn_csv = QPushButton("Export CSV")
        self.lbl_hint = QLabel("")
        self.lbl_hint.setStyleSheet("color:#666")
        top.addWidget(self.btn_refresh)
        top.addSpacing(6)
        top.addWidget(self.btn_png)
        top.addWidget(self.btn_csv)
        top.addStretch(1)
        top.addWidget(self.lbl_hint)
        self._root.addLayout(top)
        
        self.btn_refresh.clicked.connect(self._on_refresh)
        self.btn_png.clicked.connect(self._on_export_png)
        self.btn_csv.clicked.connect(self._on_export_csv)
        
        # ---- Visualization controls ----
        controls = QHBoxLayout()
        
        controls.addWidget(QLabel("Chart:"))
        self.cmb_chart_type = QComboBox()
        self.cmb_chart_type.addItems(["Line", "Stacked Bar"])
        self.cmb_chart_type.setCurrentIndex(0)  # Default: Line
        self.cmb_chart_type.currentIndexChanged.connect(self._on_viz_change)
        controls.addWidget(self.cmb_chart_type)
        
        controls.addSpacing(12)
        controls.addWidget(QLabel("X-Axis:"))
        self.cmb_xaxis = QComboBox()
        self.cmb_xaxis.addItems(["Time-proportional", "Categorical"])
        self.cmb_xaxis.setCurrentIndex(0)  # Default: Time-proportional
        self.cmb_xaxis.currentIndexChanged.connect(self._on_viz_change)
        controls.addWidget(self.cmb_xaxis)
        
        controls.addSpacing(12)
        controls.addWidget(QLabel("Regression:"))
        self.cmb_regression = QComboBox()
        self.cmb_regression.addItems([
            "None", 
            "Linear (Total)", "Linear (Matched)",
            "Quadratic (Total)", "Quadratic (Matched)",
            "Cubic (Total)", "Cubic (Matched)",
            "Exponential (Total)", "Exponential (Matched)",
        ])
        self.cmb_regression.setCurrentIndex(0)
        self.cmb_regression.currentIndexChanged.connect(self._on_viz_change)
        controls.addWidget(self.cmb_regression)
        
        controls.addStretch(1)
        self._root.addLayout(controls)
        
        # ---- Chart ----
        self._chart = MplWidget(self)
        self._root.addWidget(self._chart, 3)
        
        # ---- Outing filter table ----
        filter_header = QHBoxLayout()
        filter_header.addWidget(QLabel("Outing Filter:"))
        self.btn_select_all = QPushButton("Select All")
        self.btn_select_all.clicked.connect(self._on_select_all)
        self.btn_deselect_all = QPushButton("Deselect All")
        self.btn_deselect_all.clicked.connect(self._on_deselect_all)
        filter_header.addWidget(self.btn_select_all)
        filter_header.addWidget(self.btn_deselect_all)
        filter_header.addStretch(1)
        self._root.addLayout(filter_header)
        
        self._filter_table = QTableWidget(0, 6)
        self._filter_table.setHorizontalHeaderLabels(
            ["Include", "Outing", "Date", "Matched", "Attempted", "Not Attempted"]
        )
        self._filter_table.setMaximumHeight(200)
        self._filter_table.horizontalHeader().setStretchLastSection(True)
        self._filter_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._root.addWidget(self._filter_table, 1)
        
        # Load data and render
        self._load_data()
        self._populate_filter_table()
        self.render()
    
    def _load_data(self):
        """Load outing statistics."""
        self._outing_data = _compute_outing_stats()
    
    def _populate_filter_table(self):
        """Populate the filter table with outing data."""
        self._filter_table.setRowCount(0)
        
        sorted_outings = sorted(
            self._outing_data.items(),
            key=lambda x: (x[1]["date"] or "", x[0])
        )
        
        self._filter_table.setRowCount(len(sorted_outings))
        
        for row, (outing_name, stats) in enumerate(sorted_outings):
            # Checkbox
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk.setCheckState(Qt.Unchecked if outing_name in self._excluded_outings else Qt.Checked)
            chk.setData(Qt.UserRole, outing_name)  # Store outing name
            self._filter_table.setItem(row, 0, chk)
            
            # Outing name
            self._filter_table.setItem(row, 1, QTableWidgetItem(outing_name))
            
            # Date
            self._filter_table.setItem(row, 2, QTableWidgetItem(stats["date"]))
            
            # Counts
            self._filter_table.setItem(row, 3, QTableWidgetItem(str(stats["matched"])))
            self._filter_table.setItem(row, 4, QTableWidgetItem(str(stats["attempted"])))
            self._filter_table.setItem(row, 5, QTableWidgetItem(str(stats["not_attempted"])))
        
        self._filter_table.resizeColumnsToContents()
        self._filter_table.itemChanged.connect(self._on_filter_changed)
    
    def _on_filter_changed(self, item: QTableWidgetItem):
        """Handle checkbox toggle in filter table."""
        if item.column() != 0:
            return
        outing_name = item.data(Qt.UserRole)
        if outing_name:
            if item.checkState() == Qt.Checked:
                self._excluded_outings.discard(outing_name)
            else:
                self._excluded_outings.add(outing_name)
            self.render()
    
    def _on_select_all(self):
        """Select all outings."""
        self._excluded_outings.clear()
        self._filter_table.blockSignals(True)
        for row in range(self._filter_table.rowCount()):
            item = self._filter_table.item(row, 0)
            if item:
                item.setCheckState(Qt.Checked)
        self._filter_table.blockSignals(False)
        self.render()
    
    def _on_deselect_all(self):
        """Deselect all outings."""
        self._filter_table.blockSignals(True)
        for row in range(self._filter_table.rowCount()):
            item = self._filter_table.item(row, 0)
            if item:
                outing_name = item.data(Qt.UserRole)
                if outing_name:
                    self._excluded_outings.add(outing_name)
                item.setCheckState(Qt.Unchecked)
        self._filter_table.blockSignals(False)
        self.render()
    
    def _on_viz_change(self, _index: int = 0):
        """Handle visualization option changes."""
        self.render()
    
    def _get_filtered_data(self) -> List[Tuple[str, Dict[str, any]]]:
        """Get outing data filtered by the exclusion set, sorted by date."""
        return sorted(
            [(k, v) for k, v in self._outing_data.items() if k not in self._excluded_outings],
            key=lambda x: (x[1]["date"] or "", x[0])
        )
    
    def render(self):
        """Render the chart based on current settings."""
        self._chart.clear()
        ax = self._chart.ax
        
        filtered = self._get_filtered_data()
        
        if not filtered:
            ax.text(0.5, 0.5, "No outing data available (or all filtered out)",
                    ha="center", va="center", fontsize=12)
            self._chart.draw()
            self.lbl_hint.setText("0 outings, 0 observations")
            return
        
        chart_type = self.cmb_chart_type.currentText()
        xaxis_mode = self.cmb_xaxis.currentText()
        regression_mode = self.cmb_regression.currentText()
        
        outing_names = [o[0] for o in filtered]
        dates = [o[1]["date"] for o in filtered]
        matched = [o[1]["matched"] for o in filtered]
        attempted = [o[1]["attempted"] for o in filtered]
        not_attempted = [o[1]["not_attempted"] for o in filtered]
        totals = [m + a + na for m, a, na in zip(matched, attempted, not_attempted)]
        
        # Determine x-axis values
        if xaxis_mode == "Time-proportional":
            # Parse dates for time-proportional spacing
            try:
                import matplotlib.dates as mdates
                x_dates = [datetime.strptime(d, "%Y-%m-%d") if d else None for d in dates]
                # Filter out None dates
                valid_indices = [i for i, d in enumerate(x_dates) if d is not None]
                if not valid_indices:
                    # Fall back to categorical if no valid dates
                    xaxis_mode = "Categorical"
                else:
                    x_values = [x_dates[i] for i in valid_indices]
                    outing_names = [outing_names[i] for i in valid_indices]
                    matched = [matched[i] for i in valid_indices]
                    attempted = [attempted[i] for i in valid_indices]
                    not_attempted = [not_attempted[i] for i in valid_indices]
                    totals = [totals[i] for i in valid_indices]
            except Exception:
                xaxis_mode = "Categorical"
        
        if xaxis_mode == "Categorical":
            x_values = list(range(len(outing_names)))
        
        if chart_type == "Line":
            self._render_line_chart(ax, x_values, outing_names, matched, attempted, 
                                   not_attempted, totals, xaxis_mode, regression_mode)
        else:  # Stacked Bar
            self._render_bar_chart(ax, x_values, outing_names, matched, attempted,
                                  not_attempted, totals, xaxis_mode, regression_mode)
        
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        self._chart.fig.tight_layout()
        self._chart.draw()
        
        total_obs = sum(totals)
        self.lbl_hint.setText(f"{len(outing_names)} outings, {total_obs} observations")
    
    def _render_line_chart(self, ax, x_values, outing_names, matched, attempted, 
                           not_attempted, totals, xaxis_mode, regression_mode):
        """Render a line chart."""
        import matplotlib.dates as mdates
        
        ax.plot(x_values, matched, marker="o", label="Matched",
                color=INFLOW_COLORS["matched"], linewidth=2, markersize=5)
        ax.plot(x_values, attempted, marker="s", label="Attempted",
                color=INFLOW_COLORS["attempted"], linewidth=2, markersize=5)
        ax.plot(x_values, not_attempted, marker="^", label="Not Attempted",
                color=INFLOW_COLORS["not_attempted"], linewidth=2, markersize=5)
        
        # Add regression line if requested
        self._add_regression(ax, x_values, matched, totals, regression_mode, xaxis_mode)
        
        ax.set_ylabel("Observation Count")
        ax.set_xlabel("Outing Date" if xaxis_mode == "Time-proportional" else "Outing")
        ax.set_title("Observations per Outing by Identification Status")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
        ax.margins(x=0.02)
        
        if xaxis_mode == "Time-proportional":
            # Set ticks at actual outing dates, label every other one
            ax.set_xticks(x_values)
            labels = [d.strftime('%Y-%m-%d') if i % 2 == 0 else "" for i, d in enumerate(x_values)]
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        else:
            ax.set_xticks(x_values)
            ax.set_xticklabels(outing_names, rotation=45, ha="right", fontsize=7)
    
    def _render_bar_chart(self, ax, x_values, outing_names, matched, attempted,
                          not_attempted, totals, xaxis_mode, regression_mode):
        """Render a stacked bar chart."""
        import matplotlib.dates as mdates
        
        bar_width = 0.8 if xaxis_mode == "Categorical" else 0.8
        
        if xaxis_mode == "Time-proportional":
            # For time-proportional, use narrower bars based on date spacing
            bar_width = 0.5  # days
        
        ax.bar(x_values, matched, bar_width, label="Matched",
               color=INFLOW_COLORS["matched"], edgecolor="#333")
        
        ax.bar(x_values, attempted, bar_width, bottom=matched,
               label="Attempted", color=INFLOW_COLORS["attempted"], edgecolor="#333")
        
        bottom_na = [m + a for m, a in zip(matched, attempted)]
        ax.bar(x_values, not_attempted, bar_width, bottom=bottom_na,
               label="Not Attempted", color=INFLOW_COLORS["not_attempted"], edgecolor="#333")
        
        # Total labels
        for i, (x, total) in enumerate(zip(x_values, totals)):
            if total > 0:
                ax.text(x, total + 0.2, str(total), ha="center", va="bottom", fontsize=8)
        
        # Add regression line if requested
        self._add_regression(ax, x_values, matched, totals, regression_mode, xaxis_mode)
        
        ax.set_ylabel("Observation Count")
        ax.set_xlabel("Outing Date" if xaxis_mode == "Time-proportional" else "Outing")
        ax.set_title("Observations per Outing by Identification Status")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
        ax.margins(x=0.02)
        
        if xaxis_mode == "Time-proportional":
            # Set ticks at actual outing dates, label every other one
            ax.set_xticks(x_values)
            labels = [d.strftime('%Y-%m-%d') if i % 2 == 0 else "" for i, d in enumerate(x_values)]
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        else:
            ax.set_xticks(x_values)
            ax.set_xticklabels(outing_names, rotation=45, ha="right", fontsize=7)
    
    def _add_regression(self, ax, x_values, matched, totals, regression_mode, xaxis_mode):
        """Add regression line overlay."""
        if regression_mode == "None" or len(x_values) < 2:
            return
        
        try:
            import numpy as np
            
            # Convert x_values to numeric for regression
            if xaxis_mode == "Time-proportional":
                # Convert datetime to ordinal
                x_numeric = np.array([d.toordinal() for d in x_values])
            else:
                x_numeric = np.array(x_values, dtype=float)
            
            # Determine data source and regression type from mode string
            is_total = "Total" in regression_mode
            y_data = np.array(totals if is_total else matched, dtype=float)
            label_prefix = "Total" if is_total else "Matched"
            color = "#333333" if is_total else "#1b9e77"
            
            # Determine regression type
            if "Linear" in regression_mode:
                reg_type = "Linear"
                degree = 1
            elif "Quadratic" in regression_mode:
                reg_type = "Quadratic"
                degree = 2
            elif "Cubic" in regression_mode:
                reg_type = "Cubic"
                degree = 3
            elif "Exponential" in regression_mode:
                reg_type = "Exponential"
                degree = None  # Special handling
            else:
                return
            
            x_line = np.linspace(x_numeric.min(), x_numeric.max(), 100)
            
            if reg_type == "Exponential":
                # Exponential fit: y = a * exp(b * x)
                # Use log transform for linear fit: ln(y) = ln(a) + b*x
                # Filter out zero/negative values
                valid_mask = y_data > 0
                if np.sum(valid_mask) < 2:
                    return
                x_valid = x_numeric[valid_mask]
                y_valid = y_data[valid_mask]
                
                # Normalize x to avoid overflow
                x_norm = (x_valid - x_valid.min()) / max(1, x_valid.max() - x_valid.min())
                x_line_norm = (x_line - x_valid.min()) / max(1, x_valid.max() - x_valid.min())
                
                log_y = np.log(y_valid)
                coeffs = np.polyfit(x_norm, log_y, 1)
                a = np.exp(coeffs[1])
                b = coeffs[0]
                
                y_pred = a * np.exp(b * x_norm)
                y_line = a * np.exp(b * x_line_norm)
                
                # Clip to reasonable values
                y_line = np.clip(y_line, 0, y_data.max() * 2)
                
                # R² calculation
                ss_res = np.sum((y_valid - y_pred) ** 2)
                ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            else:
                # Polynomial fit
                # Normalize x to avoid numerical issues with high-degree polynomials
                x_norm = (x_numeric - x_numeric.min()) / max(1, x_numeric.max() - x_numeric.min())
                x_line_norm = (x_line - x_numeric.min()) / max(1, x_numeric.max() - x_numeric.min())
                
                coeffs = np.polyfit(x_norm, y_data, degree)
                poly = np.poly1d(coeffs)
                
                y_pred = poly(x_norm)
                y_line = poly(x_line_norm)
                
                # Clip to reasonable values (no negative counts)
                y_line = np.maximum(y_line, 0)
                
                # R² calculation
                ss_res = np.sum((y_data - y_pred) ** 2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Convert x back to datetime for plotting if needed
            if xaxis_mode == "Time-proportional":
                from datetime import datetime as dt
                x_plot = [dt.fromordinal(int(x)) for x in x_line]
            else:
                x_plot = x_line
            
            label = f"{label_prefix} {reg_type} (R²={r_squared:.3f})"
            ax.plot(x_plot, y_line, '--', color=color, linewidth=2, alpha=0.7, label=label)
            
        except ImportError:
            # numpy not available, skip regression
            pass
        except Exception:
            # Skip regression on any error
            pass
    
    def _on_refresh(self):
        self._ilog.log("button_click", "btn_refresh_outing_stats", value="clicked")
        self._load_data()
        self._filter_table.blockSignals(True)
        self._populate_filter_table()
        self._filter_table.blockSignals(False)
        self.render()
    
    def _on_export_png(self):
        self._ilog.log("button_click", "btn_export_png_outing_stats", value="clicked")
        if not getattr(self._chart, "fig", None):
            return
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        fname = f"outing_stats_{ts}.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save figure", str(self._reports_dir / fname), "PNG (*.png)"
        )
        if not path:
            return
        try:
            self._chart.fig.savefig(path, dpi=150, bbox_inches="tight")
            self.lbl_hint.setText(f"Saved {os.path.basename(path)}")
        except Exception as e:
            self.lbl_hint.setText(f"Save failed: {e}")
    
    def _on_export_csv(self):
        self._ilog.log("button_click", "btn_export_csv_outing_stats", value="clicked")
        filtered = self._get_filtered_data()
        if not filtered:
            return
        
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        fname = f"outing_stats_{ts}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", str(self._reports_dir / fname), "CSV (*.csv)"
        )
        if not path:
            return
        
        try:
            import csv
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["outing", "date", "matched", "attempted", "not_attempted", "total"])
                for outing_name, stats in filtered:
                    total = stats["matched"] + stats["attempted"] + stats["not_attempted"]
                    w.writerow([
                        outing_name,
                        stats["date"],
                        stats["matched"],
                        stats["attempted"],
                        stats["not_attempted"],
                        total,
                    ])
            self.lbl_hint.setText(f"Saved {os.path.basename(path)}")
        except Exception as e:
            self.lbl_hint.setText(f"Save failed: {e}")


# ---------------------------------------------------------------------------
# Query Grid Dialog
# ---------------------------------------------------------------------------

def _get_best_image_path(query_id: str) -> Optional[Path]:
    """
    Get the best image path for a query.
    
    Uses precomputed cached images from DL pipeline if available,
    otherwise falls back to original full-size images.
    """
    # Try to use cached images first (faster, already resized)
    try:
        from src.dl.image_cache import get_cached_images
        cached = get_cached_images("Queries", query_id)
        if cached:
            # Get original files to determine best image
            original_files = list_image_files("Queries", query_id)
            if original_files:
                # Reorder originals to find best
                reordered = reorder_files_with_best("Queries", query_id, original_files)
                best_original = reordered[0] if reordered else None
                
                if best_original:
                    # Find matching cached file by stem
                    best_stem = best_original.stem
                    for cp in cached:
                        if cp.stem == best_stem:
                            return cp
                    
                # If no match found, return first cached image
                return cached[0]
            return cached[0]
    except ImportError:
        pass
    
    # Fall back to original full-size images
    files = list_image_files("Queries", query_id)
    if not files:
        return None
    files = reorder_files_with_best("Queries", query_id, files)
    return files[0] if files else None


def _group_queries_by_outing() -> Dict[str, Dict[str, any]]:
    """
    Group all query IDs by their outing.
    
    Returns:
        {outing_name: {"date": "YYYY-MM-DD", "queries": [query_id1, query_id2, ...]}}
    """
    outings: Dict[str, Dict[str, any]] = defaultdict(lambda: {"date": "", "queries": []})
    query_ids = list_ids("Queries")
    
    for qid in query_ids:
        parsed = _parse_query_folder_to_outing(qid, query_id=qid)
        if parsed is None:
            continue
        date_str, outing_name = parsed
        outings[outing_name]["date"] = date_str
        outings[outing_name]["queries"].append(qid)
    
    # Sort queries within each outing
    for outing_name in outings:
        outings[outing_name]["queries"].sort()
    
    return dict(outings)


def _get_outing_date_from_data(outing_data: Dict[str, any]) -> str:
    """Get ISO date string from outing data for sorting."""
    return outing_data.get("date", "") or ""


class QueryGridDialog(QDialog):
    """
    Dialog showing a grid of best images for queries, organized by outing.
    
    Features:
    - Rows = outings (date + descriptor)
    - Columns = queries within each outing
    - Outing filter table to toggle outings on/off
    - Optional labels below images (query ID)
    - Progress bar for loading images
    - Uses precomputed cached images when available
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Query Grid")
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.resize(1200, 800)
        self._ilog = get_interaction_logger()
        self._reports_dir = archive_root() / "reports" / "figures"
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        
        self._outings_data: Dict[str, List[str]] = {}  # {outing_name: [query_ids]}
        self._excluded_outings: set = set()
        self._image_cache: Dict[str, any] = {}  # Cache loaded images
        self._loading = False
        
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(8, 8, 8, 8)
        self._root.setSpacing(6)
        
        # ---- Top bar: Refresh, Export, Labels toggle ----
        top = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh")
        self.btn_png = QPushButton("Export PNG")
        self.lbl_hint = QLabel("")
        self.lbl_hint.setStyleSheet("color:#666")
        self.chk_labels = QCheckBox("Show Labels")
        self.chk_labels.setChecked(False)
        self.chk_labels.stateChanged.connect(self._on_viz_change)
        
        top.addWidget(self.btn_refresh)
        top.addSpacing(6)
        top.addWidget(self.btn_png)
        top.addStretch(1)
        top.addWidget(self.chk_labels)
        top.addSpacing(12)
        top.addWidget(self.lbl_hint)
        self._root.addLayout(top)
        
        self.btn_refresh.clicked.connect(self._on_refresh)
        self.btn_png.clicked.connect(self._on_export_png)
        
        # Recreate button (renders after filter changes)
        self.btn_recreate = QPushButton("Recreate")
        self.btn_recreate.setToolTip("Render the grid with current filter settings")
        self.btn_recreate.clicked.connect(self.render)
        top.insertWidget(2, self.btn_recreate)
        top.insertSpacing(3, 6)
        
        # ---- Progress bar ----
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self._root.addWidget(self.progress_bar)
        
        # ---- Outing filter table ----
        filter_header = QHBoxLayout()
        filter_header.addWidget(QLabel("Outing Filter:"))
        self.btn_select_all = QPushButton("Select All")
        self.btn_select_all.clicked.connect(self._on_select_all)
        self.btn_deselect_all = QPushButton("Deselect All")
        self.btn_deselect_all.clicked.connect(self._on_deselect_all)
        filter_header.addWidget(self.btn_select_all)
        filter_header.addWidget(self.btn_deselect_all)
        filter_header.addStretch(1)
        self._root.addLayout(filter_header)
        
        self._filter_table = QTableWidget(0, 3)
        self._filter_table.setHorizontalHeaderLabels(["Include", "Outing", "Queries"])
        self._filter_table.setMaximumHeight(150)
        self._filter_table.horizontalHeader().setStretchLastSection(True)
        self._filter_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._root.addWidget(self._filter_table)
        
        # ---- Chart (grid of images) ----
        self._chart = MplWidget(self)
        self._root.addWidget(self._chart, 1)
        
        # Load data and render
        self._load_data()
        self._populate_filter_table()
        self.render()
    
    def _load_data(self):
        """Load outing-to-query mapping."""
        self._outings_data = _group_queries_by_outing()
        self._image_cache.clear()
    
    def _populate_filter_table(self):
        """Populate the filter table with outing data."""
        self._filter_table.setRowCount(0)
        
        # Sort outings by date (using stored ISO date string)
        sorted_outings = sorted(
            self._outings_data.items(),
            key=lambda x: (_get_outing_date_from_data(x[1]), x[0])
        )
        
        self._filter_table.setRowCount(len(sorted_outings))
        
        for row, (outing_name, outing_data) in enumerate(sorted_outings):
            query_ids = outing_data.get("queries", [])
            
            # Checkbox
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk.setCheckState(Qt.Unchecked if outing_name in self._excluded_outings else Qt.Checked)
            chk.setData(Qt.UserRole, outing_name)
            self._filter_table.setItem(row, 0, chk)
            
            # Outing name
            self._filter_table.setItem(row, 1, QTableWidgetItem(outing_name))
            
            # Query count
            self._filter_table.setItem(row, 2, QTableWidgetItem(str(len(query_ids))))
        
        self._filter_table.resizeColumnsToContents()
        self._filter_table.itemChanged.connect(self._on_filter_changed)
    
    def _on_filter_changed(self, item: QTableWidgetItem):
        """Handle checkbox toggle in filter table."""
        if item.column() != 0:
            return
        outing_name = item.data(Qt.UserRole)
        if outing_name:
            if item.checkState() == Qt.Checked:
                self._excluded_outings.discard(outing_name)
            else:
                self._excluded_outings.add(outing_name)
            # Don't auto-render; user clicks "Recreate" when ready
    
    def _on_select_all(self):
        """Select all outings."""
        self._excluded_outings.clear()
        self._filter_table.blockSignals(True)
        for row in range(self._filter_table.rowCount()):
            item = self._filter_table.item(row, 0)
            if item:
                item.setCheckState(Qt.Checked)
        self._filter_table.blockSignals(False)
        # Don't auto-render; user clicks "Recreate" when ready
    
    def _on_deselect_all(self):
        """Deselect all outings."""
        self._filter_table.blockSignals(True)
        for row in range(self._filter_table.rowCount()):
            item = self._filter_table.item(row, 0)
            if item:
                outing_name = item.data(Qt.UserRole)
                if outing_name:
                    self._excluded_outings.add(outing_name)
                item.setCheckState(Qt.Unchecked)
        self._filter_table.blockSignals(False)
        # Don't auto-render; user clicks "Recreate" when ready
    
    def _on_viz_change(self, _state: int = 0):
        """Handle visualization option changes (e.g., Show Labels toggle)."""
        # Only re-render if images are already loaded (no new loading needed)
        if self._image_cache:
            self.render()
    
    def _get_filtered_outings(self) -> List[Tuple[str, List[str]]]:
        """Get outings filtered by exclusion set, sorted by date."""
        # Filter and sort using stored ISO date
        filtered = [
            (k, v) for k, v in self._outings_data.items() 
            if k not in self._excluded_outings
        ]
        sorted_outings = sorted(
            filtered,
            key=lambda x: (_get_outing_date_from_data(x[1]), x[0])
        )
        # Return (outing_name, query_ids) format for compatibility
        return [(outing_name, outing_data.get("queries", [])) for outing_name, outing_data in sorted_outings]
    
    def _load_image(self, query_id: str, size: int = 120):
        """Load and cache a thumbnail image for a query."""
        if query_id in self._image_cache:
            return self._image_cache[query_id]
        
        try:
            from PIL import Image
            import numpy as np
            
            img_path = _get_best_image_path(query_id)
            if img_path is None or not img_path.exists():
                # Return a placeholder
                placeholder = np.ones((size, size, 3), dtype=np.uint8) * 200
                self._image_cache[query_id] = placeholder
                return placeholder
            
            # Load and resize
            img = Image.open(img_path)
            img = img.convert("RGB")
            
            # Resize maintaining aspect ratio
            w, h = img.size
            if w > h:
                new_w = size
                new_h = int(h * size / w)
            else:
                new_h = size
                new_w = int(w * size / h)
            
            img = img.resize((new_w, new_h), Image.LANCZOS)
            
            # Pad to square
            result = Image.new("RGB", (size, size), (200, 200, 200))
            offset = ((size - new_w) // 2, (size - new_h) // 2)
            result.paste(img, offset)
            
            arr = np.array(result)
            self._image_cache[query_id] = arr
            return arr
            
        except Exception:
            # Return placeholder on error
            import numpy as np
            placeholder = np.ones((size, size, 3), dtype=np.uint8) * 200
            self._image_cache[query_id] = placeholder
            return placeholder
    
    def _preload_images(self, query_ids: List[str], size: int = 120):
        """Preload all images with progress bar."""
        # Filter to only queries not already cached
        to_load = [qid for qid in query_ids if qid not in self._image_cache]
        
        if not to_load:
            return
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(to_load))
        self.lbl_hint.setText(f"Loading {len(to_load)} images...")
        
        self._loading = True
        
        for i, qid in enumerate(to_load):
            if not self._loading:
                break
            
            self._load_image(qid, size)
            
            # Update progress every 5 images to reduce overhead
            if i % 5 == 0 or i == len(to_load) - 1:
                self.progress_bar.setValue(i + 1)
                QApplication.processEvents()
        
        self.progress_bar.setVisible(False)
        self._loading = False
    
    def render(self):
        """Render the image grid."""
        self._chart.clear()
        fig = self._chart.fig
        fig.clf()
        
        filtered = self._get_filtered_outings()
        
        if not filtered:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No outings selected",
                    ha="center", va="center", fontsize=12)
            ax.axis("off")
            self._chart.draw()
            self.lbl_hint.setText("0 outings, 0 queries")
            return
        
        show_labels = self.chk_labels.isChecked()
        
        # Calculate grid dimensions
        num_rows = len(filtered)
        max_cols = max(len(queries) for _, queries in filtered)
        
        if max_cols == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No queries in selected outings",
                    ha="center", va="center", fontsize=12)
            ax.axis("off")
            self._chart.draw()
            return
        
        # Collect all query IDs and preload images with progress
        all_query_ids = []
        for _, query_ids in filtered:
            all_query_ids.extend(query_ids)
        
        img_size = 120  # pixels per image
        self._preload_images(all_query_ids, size=img_size)
        
        # Create subplots grid
        # Each row gets its own set of columns based on query count
        # Use GridSpec for flexible layout
        import matplotlib.gridspec as gridspec
        
        # Calculate figure size based on content
        label_height = 20 if show_labels else 0
        row_height = img_size + label_height + 30  # Extra for row label
        
        gs = gridspec.GridSpec(num_rows, max_cols + 1, figure=fig,
                               width_ratios=[0.15] + [1] * max_cols,
                               wspace=0.02, hspace=0.1)
        
        total_queries = 0
        
        for row_idx, (outing_name, query_ids) in enumerate(filtered):
            # Row label on the left (y-axis style label for this outing)
            ax_label = fig.add_subplot(gs[row_idx, 0])
            # Position text at the right edge of the label cell, centered vertically
            ax_label.text(1.0, 0.5, outing_name, 
                         transform=ax_label.transAxes,
                         ha="right", va="center", fontsize=6,
                         fontweight="bold")
            ax_label.axis("off")
            
            # Images for this row
            for col_idx, query_id in enumerate(query_ids):
                ax = fig.add_subplot(gs[row_idx, col_idx + 1])
                
                # Load and display image
                img = self._load_image(query_id, size=img_size)
                ax.imshow(img)
                ax.axis("off")
                
                # Optional label
                if show_labels:
                    # Truncate long query IDs
                    label = query_id if len(query_id) <= 15 else query_id[:12] + "..."
                    ax.set_title(label, fontsize=5, pad=1)
                
                total_queries += 1
            
            # Clear unused columns in this row
            for col_idx in range(len(query_ids), max_cols):
                ax = fig.add_subplot(gs[row_idx, col_idx + 1])
                ax.axis("off")
        
        fig.tight_layout()
        self._chart.draw()
        
        self.lbl_hint.setText(f"{num_rows} outings, {total_queries} queries")
    
    def _on_refresh(self):
        self._ilog.log("button_click", "btn_refresh_query_grid", value="clicked")
        self._load_data()
        self._filter_table.blockSignals(True)
        self._populate_filter_table()
        self._filter_table.blockSignals(False)
        self.render()
    
    def _on_export_png(self):
        self._ilog.log("button_click", "btn_export_png_query_grid", value="clicked")
        if not getattr(self._chart, "fig", None):
            return
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        fname = f"query_grid_{ts}.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save figure", str(self._reports_dir / fname), "PNG (*.png)"
        )
        if not path:
            return
        try:
            self._chart.fig.savefig(path, dpi=200, bbox_inches="tight")
            self.lbl_hint.setText(f"Saved {os.path.basename(path)}")
        except Exception as e:
            self.lbl_hint.setText(f"Save failed: {e}")
