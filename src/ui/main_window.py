# src/ui/main_window.py
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel, QSizePolicy
)

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

        t_setup = TabSetup()
        t_first = TabFirstOrder()

        # NEW: install Prev/Next gallery navigation under the First‑order gallery pane.
        try:
            t_first.add_gallery_nav_toolbar()
        except Exception:
            # Never let a layout tweak break app startup.
            pass

        t_second = TabSecondOrder()

        # Wire the quick jump from Second‑order back to whatever is selected in First‑order
        try:
            t_second.add_first_order_sync(t_first)
        except Exception:
            pass

        t_past = TabPastMatches()

        tabs.addTab(t_setup, "Setup")
        tabs.addTab(t_first, "First-order")
        tabs.addTab(t_second, "Second-order")
        tabs.addTab(t_past, "Past Matches")

        self.setCentralWidget(tabs)
        self.statusBar().showMessage("Ready")
