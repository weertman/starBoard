# src/ui/mpl_embed.py
from __future__ import annotations

from typing import Optional, Callable
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
    from matplotlib.figure import Figure
    _HAVE_MPL = True
except Exception as _e:
    FigureCanvas = object  # type: ignore
    Figure = object        # type: ignore
    NavToolbar = object    # type: ignore
    _HAVE_MPL = False
    _ERR = _e

class MplWidget(QWidget):
    """
    Drop-in Matplotlib canvas + toolbar.
    Gracefully degrades with an explanatory label if Matplotlib isn't available.
    """
    def __init__(self, parent=None, *, tight_layout: bool = True):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        if not _HAVE_MPL:
            msg = QLabel(
                "Matplotlib is not available. Install 'matplotlib' to enable charts.\n\n"
                f"Import error:\n{_ERR}"
            )
            msg.setWordWrap(True)
            msg.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            lay.addWidget(msg)
            self.canvas = None
            self.fig = None
            self.ax = None
            return

        self.fig = Figure()
        if tight_layout:
            self.fig.set_tight_layout(True)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavToolbar(self.canvas, self)

        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas, 1)

        self.ax = self.fig.add_subplot(111)

    def clear(self):
        if self.fig:
            self.fig.clear()
            self.ax = self.fig.add_subplot(111)

    def draw(self):
        if self.canvas:
            self.canvas.draw_idle()
