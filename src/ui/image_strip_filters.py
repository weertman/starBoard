from __future__ import annotations
from typing import List
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QEvent, QPointF
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QGraphicsView

from .image_strip import ImageStrip, _load_pixmap


class FilteredImageStrip(ImageStrip):
    """
    ImageStrip with per-viewer filters:
      - brightness in [-1..+1]
      - contrast   in [-1..+1]
      - saturation in [0..2]
    NOTE: Implemented with a PySide6-safe NumPy path (no memoryview.setsize).
    """
    def __init__(self, files: List[Path] | None = None, long_edge: int = 768, parent=None):
        super().__init__(files=files, long_edge=long_edge, parent=parent)
        self._brightness = 0.0
        self._contrast = 0.0
        self._saturation = 1.0

    # ------------- public API -------------
    def set_filters(self, *, brightness: float, contrast: float, saturation: float):
        self._brightness = float(brightness)
        self._contrast = float(contrast)
        self._saturation = float(saturation)
        self._apply_filters_to_current()

    # ------------- hooks that apply filtering -------------
    def _show_current(self, reset_view: bool):
        # Same structure as base, but pass through filter and emit size change
        n = len(self.files)
        self.lbl_idx.setText(f"{(self.idx+1 if n else 0)}/{n}")
        self.btn_prev.setEnabled(n > 1)
        self.btn_next.setEnabled(n > 1)
        if n == 0:
            old = self.pix_item.pixmap()
            if not old.isNull():
                self.pixmapResized.emit(old.width(), old.height(), 0, 0)
            self.pix_item.setPixmap(QPixmap())
            return
        p = self.files[self.idx]
        pix = _load_pixmap(p, self.long_edge)
        pix = self._filtered_pixmap(pix)

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
            self.fit()
            self._hires_loaded = False

    def _upgrade_to_fullres(self):
        # Preserve camera (same as base), but filter and emit size change
        if not self.files:
            return
        p = self.files[self.idx]

        old_rect = self.pix_item.boundingRect()
        center_scene = self.view.mapToScene(self.view.viewport().rect().center())
        center_item = self.pix_item.mapFromScene(center_scene)
        nx = 0.5 if old_rect.width() <= 0 else (center_item.x() - old_rect.left()) / old_rect.width()
        ny = 0.5 if old_rect.height() <= 0 else (center_item.y() - old_rect.top()) / old_rect.height()

        full = _load_pixmap(p, long_edge=0)
        full = self._filtered_pixmap(full)
        if full.isNull():
            return

        old = self.pix_item.pixmap()
        ow, oh = (old.width(), old.height()) if not old.isNull() else (0, 0)

        self.pix_item.setPixmap(full)
        self.pix_item.setTransformOriginPoint(self.pix_item.boundingRect().center())
        self.pix_item.setRotation(self._rotation)
        new_rect = self.pix_item.boundingRect()

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

    # ------------- internals -------------
    def _apply_filters_to_current(self):
        """Re-render the current pix with filters, preserving camera/rotation."""
        if not self.files:
            return
        p = self.files[self.idx]
        base = _load_pixmap(p, long_edge=0 if self._hires_loaded else self.long_edge)
        pix = self._filtered_pixmap(base)

        center_scene = self.view.mapToScene(self.view.viewport().rect().center())
        self.pix_item.setPixmap(pix)
        self.pix_item.setTransformOriginPoint(self.pix_item.boundingRect().center())
        self.pix_item.setRotation(self._rotation)
        self.scene.setSceneRect(self.pix_item.mapRectToScene(self.pix_item.boundingRect()))
        self.view.centerOn(center_scene)

    def _filtered_pixmap(self, pix: QPixmap) -> QPixmap:
        if (abs(self._brightness) < 1e-3 and
            abs(self._contrast)   < 1e-3 and
            abs(self._saturation - 1.0) < 1e-3):
            return pix
        img = pix.toImage()
        adj = self._adjust_qimage(img, self._brightness, self._contrast, self._saturation)
        return QPixmap.fromImage(adj)

    @staticmethod
    def _adjust_qimage(img: QImage, brightness: float, contrast: float, saturation: float) -> QImage:
        """
        brightness in [-1..+1] ; contrast in [-1..+1] ; saturation in [0..2]
        PySide6-safe conversion:
          - reshape via bytesPerLine() (no memoryview.setsize!)
          - respect potential row padding
        """
        # IMPORTANT: QImage is imported at module level; DO NOT re-import here (avoids UnboundLocalError)
        q = img.convertToFormat(QImage.Format_RGBA8888)
        w, h = q.width(), q.height()
        bpl = q.bytesPerLine()

        buf = np.frombuffer(q.bits(), dtype=np.uint8)
        if buf.size != h * bpl:
            buf = buf[: h * bpl]
        arr = buf.reshape((h, bpl))[:, : (4 * w)].reshape((h, w, 4)).copy()

        rgb = arr[..., :3].astype(np.float32) / 255.0
        a = arr[..., 3:4]  # keep alpha

        if abs(contrast) > 1e-6:
            rgb = (rgb - 0.5) * (1.0 + float(contrast)) + 0.5
        if abs(brightness) > 1e-6:
            rgb = rgb + float(brightness)
        if abs(saturation - 1.0) > 1e-6:
            luma = (rgb[..., 0]*0.299 + rgb[..., 1]*0.587 + rgb[..., 2]*0.114)[..., None]
            rgb = luma + (rgb - luma) * float(saturation)

        rgb = np.clip(rgb, 0.0, 1.0)
        out = np.concatenate((np.round(rgb * 255.0).astype(np.uint8), a), axis=-1)

        # Construct a new QImage from contiguous data and return a copy
        res = QImage(out.data, w, h, 4 * w, QImage.Format_RGBA8888)
        return res.copy()
