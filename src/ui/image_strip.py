# src/ui/image_strip.py
from __future__ import annotations
from PySide6.QtCore import Qt, QSize, QEvent, QPointF, Signal
from PySide6.QtGui import QPixmap, QImageReader, QAction
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QLabel,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)

from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, Dict

# ---------------- image cache (scaled + full-res) ----------------
class _ImageCache:
    def __init__(self, max_items: int = 256):
        self.max_items = max_items
        self._od: OrderedDict[Tuple[str, int], QPixmap] = OrderedDict()
    def get(self, key):
        p = self._od.get(key)
        if p is None: return None
        self._od.move_to_end(key)
        return p
    def put(self, key, pix: QPixmap):
        self._od[key] = pix
        self._od.move_to_end(key)
        while len(self._od) > self.max_items:
            self._od.popitem(last=False)

_PREVIEW_CACHE = _ImageCache(max_items=256)
_FULLRES_CACHE = _ImageCache(max_items=16)
_ORIG_SIZE: Dict[str, QSize] = {}

def _load_pixmap(path: Path, long_edge: int) -> QPixmap:
    key = (str(path), int(long_edge))
    cache = _FULLRES_CACHE if long_edge <= 0 else _PREVIEW_CACHE
    hit = cache.get(key)
    if hit is not None:
        return hit
    reader = QImageReader(str(path))
    if long_edge > 0:
        size = reader.size()
        if size.isValid():
            w, h = size.width(), size.height()
            if w >= h:
                reader.setScaledSize(QSize(long_edge, max(1, int(h * (long_edge / max(1, w))))))
            else:
                reader.setScaledSize(QSize(max(1, int(w * (long_edge / max(1, h)))), long_edge))
    img = reader.read()
    pix = QPixmap.fromImage(img)
    cache.put(key, pix)
    return pix

class ImageStrip(QWidget):
    """
    Mini viewer with scrub / wheel-zoom / continuous rotate (hold R + drag).
    Upgrades to full-res once you zoom in. The view now expands to fill
    the space its parent gives it (no fixed pixel height).
    """

    # NEW: notify listeners when the underlying pixmap size changes
    pixmapResized = Signal(int, int, int, int)   # old_w, old_h, new_w, new_h

    def __init__(self, files: List[Path] | None = None, long_edge: int = 768, parent=None):
        super().__init__(parent)
        self.files: List[Path] = files or []
        self.idx = 0
        self.long_edge = int(long_edge)
        self._rotation = 0.0
        self._scale = 1.0
        self._hires_loaded = False
        self._hires_trigger = 1.2
        self._rotate_key_down = False
        self._rotating = False
        self._drag_start_x = 0.0
        self._rotation_at_press = 0.0

        self.setFocusPolicy(Qt.StrongFocus)
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.btn_prev = QPushButton("◀")
        self.btn_next = QPushButton("▶")
        self.lbl_idx = QLabel("0/0")

        self.view = QGraphicsView()
        self.scene = QGraphicsScene(self.view)
        self.view.setScene(self.scene)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setMinimumHeight(80)

        self.pix_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pix_item)

        self.view.viewport().installEventFilter(self)
        self.view.installEventFilter(self)

        self.btn_prev.clicked.connect(self.prev)
        self.btn_next.clicked.connect(self.next)

        lay.addWidget(self.btn_prev)
        lay.addWidget(self.view, 1)
        lay.addWidget(self.btn_next)
        lay.addWidget(self.lbl_idx)

        self._act_fit = QAction("Fit", self); self._act_fit.triggered.connect(self.fit)
        self.addAction(self._act_fit)
        self.setContextMenuPolicy(Qt.ActionsContextMenu)

        self._show_current(reset_view=True)

    # -------- public API --------
    def set_files(self, files: List[Path]):
        self.files = files or []
        self.idx = 0
        self._rotation = 0.0
        self._scale = 1.0
        self._hires_loaded = False
        self._show_current(reset_view=True)

    def set_view_height(self, h: int):
        """Baseline minimum; parent splitters/cards may grow the view beyond this."""
        h = max(80, int(h))
        self.view.setMinimumHeight(h)
        self.fit()

    def set_view_min_width(self, w: int):
        """Baseline minimum width for the image viewport."""
        w = max(100, int(w))
        self.view.setMinimumWidth(w)
        self.fit()

    def prev(self):
        if not self.files: return
        self.idx = (self.idx - 1) % len(self.files)
        self._rotation = 0.0; self._scale = 1.0; self._hires_loaded = False
        self._show_current(reset_view=True)

    def next(self):
        if not self.files: return
        self.idx = (self.idx + 1) % len(self.files)
        self._rotation = 0.0; self._scale = 1.0; self._hires_loaded = False
        self._show_current(reset_view=True)

    def fit(self):
        self.view.resetTransform()
        self._scale = 1.0
        self.view.fitInView(self.pix_item, Qt.KeepAspectRatio)

    # -------- event handling: wheel-zoom + continuous rotate --------
    def eventFilter(self, obj, ev):
        if obj is self.view.viewport():
            t = ev.type()
            if t == QEvent.Wheel:
                delta = ev.angleDelta().y()
                if delta == 0: return False
                factor = 1.0015 ** float(delta)
                self.view.scale(factor, factor)
                self._scale *= factor
                if not self._hires_loaded and self._scale >= self._hires_trigger:
                    self._upgrade_to_fullres()
                ev.accept(); return True
            if t == QEvent.MouseButtonPress and getattr(ev, "button", lambda: None)() == Qt.LeftButton:
                if self._rotate_key_down:
                    self._rotating = True
                    self._drag_start_x = float(getattr(ev, "position", lambda: QPointF(0, 0))().x())
                    self._rotation_at_press = self._rotation
                    self.view.setDragMode(QGraphicsView.NoDrag)
                    ev.accept(); return True
            if t == QEvent.MouseMove and self._rotating:
                x = float(getattr(ev, "position", lambda: QPointF(0, 0))().x())
                dx = x - self._drag_start_x
                self._rotation = self._rotation_at_press + (dx * 0.3)
                self.pix_item.setRotation(self._rotation)
                ev.accept(); return True
            if t == QEvent.MouseButtonRelease and self._rotating:
                self._rotating = False
                self.view.setDragMode(QGraphicsView.ScrollHandDrag)
                ev.accept(); return True

        if obj is self.view:
            if ev.type() == QEvent.KeyPress and ev.key() == Qt.Key_R:
                self._rotate_key_down = True
            elif ev.type() == QEvent.KeyRelease and ev.key() == Qt.Key_R:
                self._rotate_key_down = False
        return False

    # -------- internals --------
    def _show_current(self, reset_view: bool):
        n = len(self.files)
        self.lbl_idx.setText(f"{(self.idx+1 if n else 0)}/{n}")
        self.btn_prev.setEnabled(n > 1); self.btn_next.setEnabled(n > 1)
        if n == 0:
            # emit only if something was previously shown
            old = self.pix_item.pixmap()
            if not old.isNull():
                self.pixmapResized.emit(old.width(), old.height(), 0, 0)
            self.pix_item.setPixmap(QPixmap())
            return

        p = self.files[self.idx]
        pix = _load_pixmap(p, self.long_edge)

        # EMIT size change if needed
        old = self.pix_item.pixmap()
        ow, oh = (old.width(), old.height()) if not old.isNull() else (0, 0)

        self.pix_item.setPixmap(pix)
        self.pix_item.setTransformOriginPoint(self.pix_item.boundingRect().center())
        self.pix_item.setRotation(self._rotation)
        self.scene.setSceneRect(self.pix_item.mapRectToScene(self.pix_item.boundingRect()))

        nw, nh = pix.width(), pix.height()
        if (ow != nw) or (oh != nh):
            self.pixmapResized.emit(int(ow), int(oh), int(nw), int(nh))

        if reset_view:
            self.fit(); self._hires_loaded = False

    def _upgrade_to_fullres(self):
        if not self.files: return
        p = self.files[self.idx]

        # --- Preserve camera ---
        old_rect = self.pix_item.boundingRect()  # item coords
        center_scene = self.view.mapToScene(self.view.viewport().rect().center())
        center_item = self.pix_item.mapFromScene(center_scene)
        nx = 0.5 if old_rect.width()  <= 0 else (center_item.x() - old_rect.left()) / old_rect.width()
        ny = 0.5 if old_rect.height() <= 0 else (center_item.y() - old_rect.top())  / old_rect.height()

        full = _load_pixmap(p, long_edge=0)
        if full.isNull():
            return

        # EMIT size change if needed
        old = self.pix_item.pixmap()
        ow, oh = (old.width(), old.height()) if not old.isNull() else (0, 0)

        self.pix_item.setPixmap(full)
        self.pix_item.setTransformOriginPoint(self.pix_item.boundingRect().center())
        self.pix_item.setRotation(self._rotation)
        new_rect = self.pix_item.boundingRect()

        # keep on‑screen magnification
        if (old_rect.width() > 0 and old_rect.height() > 0 and
            new_rect.width() > 0 and new_rect.height() > 0):
            s = min(old_rect.width()/new_rect.width(), old_rect.height()/new_rect.height())
            if 0 < s < 1e6:
                self.view.scale(s, s)
                self._scale *= s

        cx = new_rect.left() + nx * new_rect.width()
        cy = new_rect.top()  + ny * new_rect.height()
        self.view.centerOn(self.pix_item.mapToScene(QPointF(cx, cy)))

        self.scene.setSceneRect(self.pix_item.mapRectToScene(self.pix_item.boundingRect()))
        self._hires_loaded = True

        nw, nh = full.width(), full.height()
        if (ow != nw) or (oh != nh):
            self.pixmapResized.emit(int(ow), int(oh), int(nw), int(nh))
