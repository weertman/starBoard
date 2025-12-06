# src/ui/matrix_matches_dialog.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QSize
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTableView, QHeaderView, QPushButton
)
from src.data.matches_matrix import MatchMatrixData, load_match_matrix

_COLORS = {"yes": QColor(0,140,60), "maybe": QColor(220,160,0), "no": QColor(190,50,50)}

@dataclass
class _Cell:
    verdict: str = ""; notes: str = ""; updated: str = ""

class _MatrixModel(QAbstractTableModel):
    def __init__(self, data: MatchMatrixData):
        super().__init__(); self.d = data
        self._cells: dict[Tuple[int,int], _Cell] = {}
        q_index = {qid:i for i, qid in enumerate(self.d.query_ids)}
        g_index = {gid:j for j, gid in enumerate(self.d.gallery_ids)}
        for (qid,gid), v in self.d.verdict_by_pair.items():
            i = q_index.get(qid); j = g_index.get(gid)
            if i is None or j is None: continue
            self._cells[(i,j)] = _Cell(v, self.d.notes_by_pair.get((qid,gid), ""), self.d.updated_by_pair.get((qid,gid), ""))

    def rowCount(self, parent=QModelIndex()) -> int: return len(self.d.query_ids)
    def columnCount(self, parent=QModelIndex()) -> int: return len(self.d.gallery_ids)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole) -> Any:
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal: return self.d.gallery_ids[section]
            else:
                qid = self.d.query_ids[section]; d = self.d.last_obs_by_query.get(qid)
                return f"{qid}   ({d.isoformat() if d else '—'})"
        if role == Qt.ToolTipRole and orientation == Qt.Vertical:
            qid = self.d.query_ids[section]; d = self.d.last_obs_by_query.get(qid)
            return f"Query: {qid}\nLast observed: {d.isoformat() if d else '—'}"
        return None

    def data(self, idx: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not idx.isValid(): return None
        cell = self._cells.get((idx.row(), idx.column()))
        if role == Qt.DisplayRole: return "•" if cell else ""
        if role == Qt.TextAlignmentRole: return Qt.AlignCenter
        if role == Qt.ForegroundRole and cell: return QBrush(Qt.white)
        if role == Qt.BackgroundRole and cell: return QBrush(_COLORS.get(cell.verdict, QColor(120,120,120)))
        if role == Qt.ToolTipRole and cell:
            qid = self.d.query_ids[idx.row()]; gid = self.d.gallery_ids[idx.column()]
            return (f"{qid} × {gid}\nVerdict: {cell.verdict}\n"
                    f"Updated: {cell.updated or '—'}\nNotes: {cell.notes or '—'}")
        return None

class MatrixMatchesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Past Matches — Queries × Gallery"); self.resize(1100, 700)
        lay = QVBoxLayout(self)
        legend = QHBoxLayout(); legend.addWidget(QLabel("<b>Legend:</b>"))
        for name, col in (("Yes", _COLORS["yes"]), ("Maybe", _COLORS["maybe"]), ("No", _COLORS["no"])):
            sw = QLabel("  "); sw.setAutoFillBackground(True); pal = sw.palette()
            pal.setColor(sw.backgroundRole(), col); sw.setPalette(pal); sw.setMinimumSize(QSize(16,16))
            legend.addWidget(sw); legend.addWidget(QLabel(name)); legend.addSpacing(12)
        legend.addStretch(1)
        btn = QPushButton("Refresh"); btn.clicked.connect(self._reload); legend.addWidget(btn)
        lay.addLayout(legend)

        self.table = QTableView(self)
        self.table.setSortingEnabled(False); self.table.setCornerButtonEnabled(False)
        self.table.setAlternatingRowColors(False); self.table.setShowGrid(True)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setSelectionMode(QTableView.NoSelection); self.table.setWordWrap(False)
        lay.addWidget(self.table, 1)
        self._reload()

    def _reload(self):
        data = load_match_matrix()
        model = _MatrixModel(data)
        self.table.setModel(model)
        self.table.horizontalHeader().setDefaultSectionSize(28)
        self.table.verticalHeader().setDefaultSectionSize(24)
