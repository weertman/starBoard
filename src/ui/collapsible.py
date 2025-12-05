# src/ui/collapsible.py
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QToolButton, QFrame, QSizePolicy
)

class CollapsibleSection(QWidget):
    """
    Small helper to show a titled header with a disclosure arrow and a content area
    that can be expanded/collapsed. Default starts collapsed.
    """
    toggled = Signal(bool)

    def __init__(self, title: str, *, start_collapsed: bool = True, parent=None):
        super().__init__(parent)
        self._expanded = not start_collapsed

        self.toggle = QToolButton(self)
        self.toggle.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(self._expanded)
        self.toggle.setArrowType(Qt.DownArrow if self._expanded else Qt.RightArrow)
        self.toggle.setText(title)
        self.toggle.toggled.connect(self._on_toggled)

        self.line = QFrame(self)
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.content = QWidget(self)
        self.content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(0)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(self.toggle)
        lay.addWidget(self.line)
        lay.addWidget(self.content, 1)

        self._apply_visibility()

    def setContent(self, widget: QWidget) -> None:
        # remove old
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
        self.content_layout.addWidget(widget, 1)  # stretch factor for expansion

    def _on_toggled(self, checked: bool) -> None:
        self._expanded = checked
        self.toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self._apply_visibility()
        self.toggled.emit(checked)

    def _apply_visibility(self) -> None:
        self.content.setVisible(self._expanded)
