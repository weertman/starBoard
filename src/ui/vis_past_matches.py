# src/ui/vis_past_matches.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
from pathlib import Path
import os
from datetime import datetime

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox,
    QTableWidget, QTableWidgetItem, QWidget, QSpinBox, QFileDialog
)
from PySide6.QtCore import Qt

from src.ui.mpl_embed import MplWidget
from src.data.archive_paths import archive_root
from src.data.past_matches import PastMatchesDataset

COLORS = {"yes": "#1b9e77", "maybe": "#d95f02", "no": "#d73027"}


class PastMatchesDialogBase(QDialog):
    def __init__(self, title: str, ds: PastMatchesDataset, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.resize(880, 600)
        self._ds = ds
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
        self.render()

    def _on_export_png(self):
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
