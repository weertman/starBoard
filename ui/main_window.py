# src/ui/main_window.py
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel, QSizePolicy)

from .tab_setup import TabSetup
from .tab_first_order import TabFirstOrder
from .tab_second_order import TabSecondOrder
from .tab_past_matches import TabPastMatches

class _PlaceholderTab(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lbl = QLabel(f"{title} — coming soon")
        lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(lbl)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("starBoard")
        self.setMinimumSize(480, 360)
        self.resize(1280, 860)

        tabs = QTabWidget()
        tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        tabs.addTab(TabSetup(), "Setup")
        tabs.addTab(TabFirstOrder(), "First-order")
        tabs.addTab(TabSecondOrder(), "Second-order")      # <-- use live tab
        tabs.addTab(TabPastMatches(), "Past Matches")

        self.setCentralWidget(tabs)
        self.statusBar().showMessage("Ready")

