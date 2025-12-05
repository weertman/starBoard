# src/ui/annotator_view.py
from __future__ import annotations

from pathlib import Path
import json
import uuid
import math
from typing import List, Dict, Tuple, Optional

from PySide6.QtCore import Qt, Signal, QEvent, QPointF
from PySide6.QtGui import (
    QPen, QBrush, QColor, QImageReader, QPolygonF,
    QUndoStack, QUndoCommand, QAction
)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QToolButton, QButtonGroup,
    QGraphicsEllipseItem, QGraphicsSimpleTextItem, QGraphicsPolygonItem, QGraphicsItem,
    QSlider
)

from src.ui.image_strip import ImageStrip
from src.data.archive_paths import roots_for_read


class AnnotatorView(QWidget):
    """
    Image + overlay view built on top of ImageStrip.

    Tools:
      - Select: select a polygon to edit (drag vertices, insert on edge, Delete polygon)
      - Point: click to add a labeled marker (undoable)
      - Polygon: click to add vertices, Enter/Double‑click to finish (undoable)
        - Esc cancels an in‑progress polygon
        - Right‑click also cancels

    Persistence:
      - Sidecar I/O per image: <image>.<ann>.json
      - Coordinates stored in ORIGINAL image pixel space
      - Overlays are parented to the pixmap item (auto‑rotate/zoom correctly)
    """
    currentImageChanged = Signal(Path)

    def __init__(self, target: str, title: str, parent=None):
        super().__init__(parent)
        self._target = "Gallery" if (target or "").lower().startswith("g") else "Queries"

        # ---- Toolbar ----
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        self._title_label = QLabel(f"<b>{title}</b>")
        bar.addWidget(self._title_label)

        # Tools
        self.btn_select = QToolButton(); self.btn_select.setText("Select"); self.btn_select.setCheckable(True)
        self.btn_point  = QToolButton(); self.btn_point.setText("Point");  self.btn_point.setCheckable(True)
        self.btn_poly   = QToolButton(); self.btn_poly.setText("Polygon"); self.btn_poly.setCheckable(True)

        self._tools = QButtonGroup(self)
        self._tools.setExclusive(True)
        for b in (self.btn_select, self.btn_point, self.btn_poly):
            self._tools.addButton(b)
        # Default tool = Select (per request)
        self.btn_select.setChecked(True)

        # Undo/Redo
        self.btn_undo = QPushButton("Undo")
        self.btn_redo = QPushButton("Redo")

        # Save
        self.btn_save = QPushButton("Save ann")
        self.btn_save.setToolTip("Save annotations for the current image")

        bar.addSpacing(8)
        bar.addWidget(self.btn_select)
        bar.addWidget(self.btn_point)
        bar.addWidget(self.btn_poly)
        bar.addSpacing(12)
        bar.addWidget(self.btn_undo)
        bar.addWidget(self.btn_redo)

        # ---- Adjustments (Brightness / Contrast / Saturation) ----
        # 0..200 with 100 = neutral (mapped below to [-1..+1] or [0..2])
        def _mk_slider(tt: str) -> QSlider:
            s = QSlider(Qt.Horizontal)
            s.setRange(0, 200)
            s.setValue(100)
            s.setFixedWidth(110)
            s.setToolTip(tt)
            return s

        bar.addSpacing(18)
        bar.addWidget(QLabel("B"))
        self.sld_bright = _mk_slider("Brightness")
        bar.addWidget(self.sld_bright)

        bar.addSpacing(8)
        bar.addWidget(QLabel("C"))
        self.sld_contrast = _mk_slider("Contrast")
        bar.addWidget(self.sld_contrast)

        bar.addSpacing(8)
        bar.addWidget(QLabel("S"))
        self.sld_sat = _mk_slider("Saturation")
        bar.addWidget(self.sld_sat)

        bar.addStretch(1)
        bar.addWidget(self.btn_save)
        lay.addLayout(bar)

        # ---- Image view (reuse your fast viewer) ----
        self.strip = ImageStrip(files=[], long_edge=768)
        lay.addWidget(self.strip, 1)

        # Intercept mouse on the viewport and keys on the view
        self.strip.view.viewport().installEventFilter(self)
        self.strip.view.installEventFilter(self)

        # Hook image navigation from ImageStrip
        self.strip.btn_prev.clicked.connect(self._after_navigate)
        self.strip.btn_next.clicked.connect(self._after_navigate)

        # Adjustments -> strip filters
        self.sld_bright.valueChanged.connect(self._on_adjust_changed)
        self.sld_contrast.valueChanged.connect(self._on_adjust_changed)
        self.sld_sat.valueChanged.connect(self._on_adjust_changed)
        self._on_adjust_changed()  # initialize

        # Actions / shortcuts
        self.btn_save.clicked.connect(self.save_current_annotations)

        self._act_undo = QAction("Undo", self)
        self._act_undo.setShortcut("Ctrl+Z")
        self._act_undo.triggered.connect(lambda: self._undo_stack.undo())
        self.addAction(self._act_undo)

        self._act_redo = QAction("Redo", self)
        self._act_redo.setShortcut("Ctrl+Shift+Z")
        self._act_redo.triggered.connect(lambda: self._undo_stack.redo())
        self.addAction(self._act_redo)

        # ---- Annotation state ----
        self._points_items: Dict[str, Tuple[QGraphicsEllipseItem, QGraphicsSimpleTextItem]] = {}
        self._points_data: Dict[str, Dict] = {}   # id -> {'label': str, 'xy_item': (x,y)}

        self._polys_items: Dict[str, Tuple[QGraphicsPolygonItem, QGraphicsSimpleTextItem]] = {}
        self._polys_data: Dict[str, Dict] = {}    # id -> {'label': str, 'points_item': [(x,y), ...]}

        self._dirty: bool = False
        self._loaded_for_path: Optional[Path] = None
        self._loaded_pix_size: Tuple[float, float] = (0.0, 0.0)

        # Polygon-in-progress
        self._poly_temp_points: List[Tuple[float, float]] = []
        self._poly_temp_item: Optional[QGraphicsPolygonItem] = None

        # Selection & edit
        self._selected_poly_id: Optional[str] = None
        self._vertex_handles: List[_VertexHandle] = []
        self._press_scene: Optional[QPointF] = None
        self._press_item: Optional[QPointF] = None
        self._press_moved: bool = False

        # Undo stack
        self._undo_stack = QUndoStack(self)
        self.btn_undo.clicked.connect(self._undo_stack.undo)
        self.btn_redo.clicked.connect(self._undo_stack.redo)
        self._undo_stack.indexChanged.connect(self._on_stack_changed)
        self._refresh_undo_buttons()

    # ---------- Public surface ----------
    def set_title(self, title: str) -> None:
        self._title_label.setText(f"<b>{title}</b>")

    def set_files(self, files: List[Path]) -> None:
        self._save_prev_if_dirty()
        self._clear_annotations()
        self.strip.set_files(files or [])
        self._load_sidecar_for_current()
        p = self.current_path()
        if p:
            self.currentImageChanged.emit(p)

    def next(self) -> None:
        self._save_prev_if_dirty()
        self.strip.next()
        self._load_sidecar_for_current()

    def prev(self) -> None:
        self._save_prev_if_dirty()
        self.strip.prev()
        self._load_sidecar_for_current()

    def current_path(self) -> Optional[Path]:
        if not self.strip.files:
            return None
        idx = max(0, min(self.strip.idx, len(self.strip.files) - 1))
        return self.strip.files[idx]

    def save_current_annotations(self) -> None:
        self._save_current_if_dirty(force=True)

    # ---------- Event filter ----------
    def eventFilter(self, obj, ev) -> bool:
        # Mouse on viewport (points/polygon/select); let ImageStrip keep ScrollHandDrag
        if obj is self.strip.view.viewport():
            t = ev.type()

            # ------- Track press/move/release in Select to allow panning -------
            if self.btn_select.isChecked():
                if t == QEvent.MouseButtonPress and getattr(ev, "button", lambda: None)() == Qt.LeftButton:
                    self._press_scene = self.strip.view.mapToScene(int(ev.position().x()), int(ev.position().y()))
                    self._press_item = self.strip.pix_item.mapFromScene(self._press_scene)
                    self._press_moved = False
                    # DO NOT consume: allow ScrollHandDrag
                    return False

                if t == QEvent.MouseMove and self._press_scene is not None:
                    self._press_moved = True
                    return False

                if t == QEvent.MouseButtonRelease and getattr(ev, "button", lambda: None)() == Qt.LeftButton:
                    # If it was effectively a click (not a drag), select polygon under cursor
                    try:
                        release_scene = self.strip.view.mapToScene(int(ev.position().x()), int(ev.position().y()))
                    except Exception:
                        release_scene = None
                    if (self._press_scene is not None) and (release_scene is not None):
                        dx = release_scene.x() - self._press_scene.x()
                        dy = release_scene.y() - self._press_scene.y()
                        if (dx*dx + dy*dy) <= 9.0:  # <=3px movement ⇒ click
                            item_pt = self.strip.pix_item.mapFromScene(release_scene)
                            cid = self._poly_hit_test(item_pt)
                            self._select_polygon(cid)
                    self._press_scene = None
                    self._press_item = None
                    self._press_moved = False
                    return False  # still let view finish its gesture

            # ------- Drawing / points -------
            if t == QEvent.MouseButtonPress and getattr(ev, "button", lambda: None)() == Qt.LeftButton:
                posf = getattr(ev, "position", None)
                if posf is None:
                    return False
                posf = posf()
                scene_pt = self.strip.view.mapToScene(int(posf.x()), int(posf.y()))
                item_pt = self.strip.pix_item.mapFromScene(scene_pt)
                if not self.strip.pix_item.contains(item_pt):
                    return False

                x_item, y_item = float(item_pt.x()), float(item_pt.y())
                if self.btn_point.isChecked():
                    self._push_add_point(x_item, y_item)
                    return True

                if self.btn_poly.isChecked():
                    # add a vertex to the temp polygon
                    self._poly_add_vertex(x_item, y_item)
                    return True

            # Finish polygon by double‑click
            if t == QEvent.MouseButtonDblClick:
                if self.btn_poly.isChecked():
                    self._poly_finalize_if_possible()
                    return True
                # Insert vertex on edge when in Select and a polygon is selected
                if self.btn_select.isChecked() and self._selected_poly_id:
                    posf = getattr(ev, "position", None)
                    if posf is not None:
                        posf = posf()
                        scene_pt = self.strip.view.mapToScene(int(posf.x()), int(posf.y()))
                        item_pt = self.strip.pix_item.mapFromScene(scene_pt)
                        x_item, y_item = float(item_pt.x()), float(item_pt.y())
                        i, d = self._closest_edge_index(self._selected_poly_id, x_item, y_item)
                        if i >= 0 and d <= 6.0:  # ~6 px threshold
                            self._push_insert_vertex(self._selected_poly_id, i + 1, x_item, y_item)
                            return True
                return False  # otherwise let default behavior occur

            # Right click cancels in‑progress polygon
            if t == QEvent.MouseButtonPress and getattr(ev, "button", lambda: None)() == Qt.RightButton:
                if self.btn_poly.isChecked() and self._poly_temp_points:
                    self._poly_cancel_temp()
                    return True

        # Keys on the view (Enter/Esc/Backspace/Delete)
        if obj is self.strip.view and ev.type() == QEvent.KeyPress:
            key = ev.key()
            if self.btn_poly.isChecked() and key in (Qt.Key_Return, Qt.Key_Enter):
                self._poly_finalize_if_possible(); return True
            if self.btn_poly.isChecked() and key == Qt.Key_Escape:
                self._poly_cancel_temp(); return True
            if self.btn_poly.isChecked() and key == Qt.Key_Backspace:
                # remove last vertex while drawing
                if self._poly_temp_points:
                    self._poly_temp_points.pop()
                    self._poly_update_temp_item()
                if not self._poly_temp_points:
                    self._poly_cancel_temp()
                return True
            if self.btn_select.isChecked() and key == Qt.Key_Delete and self._selected_poly_id:
                self._undo_stack.push(_DeletePolygonCmd(self, self._selected_poly_id))
                self._selected_poly_id = None
                return True

        return super().eventFilter(obj, ev)

    # ---------- Adjustments ----------
    def _on_adjust_changed(self, *_):
        # Map UI [0..200] → brightness, contrast in [-1..+1]; saturation in [0..2]
        b = (self.sld_bright.value() - 100) / 100.0
        c = (self.sld_contrast.value() - 100) / 100.0
        s = (self.sld_sat.value()) / 100.0
        try:
            self.strip.set_filters(brightness=b, contrast=c, saturation=s)
        except Exception:
            pass

    # ---------- Internals ----------
    def _on_stack_changed(self, *_):
        self._dirty = True
        self._refresh_undo_buttons()

    def _refresh_undo_buttons(self):
        self.btn_undo.setEnabled(self._undo_stack.canUndo())
        self.btn_redo.setEnabled(self._undo_stack.canRedo())

    def _after_navigate(self) -> None:
        self._save_prev_if_dirty()
        self._load_sidecar_for_current()
        p = self.current_path()
        if p:
            self.currentImageChanged.emit(p)

    # ----- Points -----
    def _push_add_point(self, x_item: float, y_item: float, label: Optional[str] = None) -> None:
        cid = uuid.uuid4().hex[:8]
        label = label if label is not None else f"P{len(self._points_items) + 1}"
        self._undo_stack.push(_AddPointCmd(self, cid, x_item, y_item, label))

    def _create_point_graphics(self, cid: str, x_item: float, y_item: float, label: str):
        r = 4.0
        pen = QPen(QColor(0, 170, 255)); pen.setWidthF(1.5)
        brush = QBrush(Qt.NoBrush)

        ell = QGraphicsEllipseItem(x_item - r, y_item - r, 2 * r, 2 * r, parent=self.strip.pix_item)
        ell.setPen(pen); ell.setBrush(brush)
        txt = QGraphicsSimpleTextItem(label, parent=self.strip.pix_item)
        txt.setBrush(QBrush(QColor(0, 170, 255)))
        txt.setPos(x_item + 6.0, y_item - 6.0)
        return ell, txt

    def _register_point(self, cid: str, x_item: float, y_item: float, label: str,
                        ell: QGraphicsEllipseItem, txt: QGraphicsSimpleTextItem):
        self._points_items[cid] = (ell, txt)
        self._points_data[cid] = {"id": cid, "label": label, "xy_item": (x_item, y_item)}
        self._dirty = True

    def _unregister_point(self, cid: str):
        tup = self._points_items.pop(cid, None)
        if tup:
            ell, txt = tup
            try:
                if ell.scene():
                    ell.scene().removeItem(ell)
                if txt.scene():
                    txt.scene().removeItem(txt)
            except Exception:
                pass
        self._points_data.pop(cid, None)
        self._dirty = True

    # ----- Polygon (in‑progress and finalized) -----
    def _poly_add_vertex(self, x_item: float, y_item: float):
        # Ensure temp item exists
        if self._poly_temp_item is None:
            self._poly_temp_item = QGraphicsPolygonItem(parent=self.strip.pix_item)
            pen = QPen(QColor(255, 170, 0)); pen.setWidthF(1.5)
            self._poly_temp_item.setPen(pen)
            self._poly_temp_item.setBrush(QBrush(QColor(255, 170, 0, 28)))
            self._poly_temp_item.setZValue(800)
        self._poly_temp_points.append((x_item, y_item))
        self._poly_update_temp_item()

    def _poly_update_temp_item(self):
        if self._poly_temp_item is None:
            return
        poly = QPolygonF([QPointF(x, y) for (x, y) in self._poly_temp_points])
        self._poly_temp_item.setPolygon(poly)

    def _poly_finalize_if_possible(self):
        if len(self._poly_temp_points) >= 3:
            points = list(self._poly_temp_points)
            cid = uuid.uuid4().hex[:8]
            label = f"Φ{len(self._polys_items) + 1}"
            self._undo_stack.push(_AddPolygonCmd(self, cid, points, label))
        self._poly_cancel_temp()

    def _poly_cancel_temp(self):
        if self._poly_temp_item is not None:
            try:
                if self._poly_temp_item.scene():
                    self._poly_temp_item.scene().removeItem(self._poly_temp_item)
            except Exception:
                pass
        self._poly_temp_points.clear()
        self._poly_temp_item = None

    def _style_polygon(self, poly_item: QGraphicsPolygonItem, selected: bool):
        pen = QPen(QColor(255, 170, 0)); pen.setWidthF(2.0 if selected else 1.5)
        poly_item.setPen(pen)
        alpha = 64 if selected else 36
        poly_item.setBrush(QBrush(QColor(255, 170, 0, alpha)))

    def _create_polygon_graphics(self, cid: str, points_item: List[Tuple[float, float]], label: str):
        poly = QPolygonF([QPointF(x, y) for (x, y) in points_item])
        poly_item = QGraphicsPolygonItem(poly, parent=self.strip.pix_item)
        poly_item.setZValue(700)
        self._style_polygon(poly_item, selected=(cid == self._selected_poly_id))
        # IMPORTANT: don't let polygons consume mouse presses — preserves hand‑drag
        poly_item.setAcceptedMouseButtons(Qt.NoButton)

        # Label at polygon centroid (rough mean)
        cx = sum(x for x, _ in points_item) / max(1, len(points_item))
        cy = sum(y for _, y in points_item) / max(1, len(points_item))
        txt = QGraphicsSimpleTextItem(label, parent=self.strip.pix_item)
        txt.setBrush(QBrush(QColor(255, 170, 0)))
        txt.setPos(cx + 6.0, cy + 6.0)
        return poly_item, txt

    def _register_polygon(self, cid: str, points_item: List[Tuple[float, float]], label: str,
                          poly_item: QGraphicsPolygonItem, txt: QGraphicsSimpleTextItem):
        self._polys_items[cid] = (poly_item, txt)
        self._polys_data[cid] = {"id": cid, "label": label, "points_item": list(points_item)}
        self._style_polygon(poly_item, selected=(cid == self._selected_poly_id))
        self._dirty = True

    def _unregister_polygon(self, cid: str):
        tup = self._polys_items.pop(cid, None)
        if tup:
            poly_item, txt = tup
            try:
                if poly_item.scene():
                    poly_item.scene().removeItem(poly_item)
                if txt.scene():
                    txt.scene().removeItem(txt)
            except Exception:
                pass
        self._polys_data.pop(cid, None)
        self._dirty = True

    # ----- Selection & vertex editing -----
    def _select_polygon(self, cid: Optional[str]):
        if cid == self._selected_poly_id:
            return
        # clear previous selection
        if self._selected_poly_id and self._selected_poly_id in self._polys_items:
            prev_poly, _ = self._polys_items[self._selected_poly_id]
            self._style_polygon(prev_poly, selected=False)
        self._clear_handles()

        self._selected_poly_id = cid
        if not cid:
            return
        if cid in self._polys_items:
            poly_item, _ = self._polys_items[cid]
            self._style_polygon(poly_item, selected=True)
            self._build_handles_for_selected()

    def _clear_handles(self):
        for h in self._vertex_handles:
            try:
                if h.scene(): h.scene().removeItem(h)
            except Exception:
                pass
        self._vertex_handles.clear()

    def _build_handles_for_selected(self):
        cid = self._selected_poly_id
        if not cid: return
        pts = self._polys_data.get(cid, {}).get("points_item", [])
        self._vertex_handles = []
        for i, (x, y) in enumerate(pts):
            self._vertex_handles.append(_VertexHandle(self, cid, i, float(x), float(y)))

    def _update_polygon_graphics(self, cid: str):
        tup = self._polys_items.get(cid)
        if not tup: return
        poly_item, txt = tup
        pts = [QPointF(float(x), float(y)) for (x, y) in (self._polys_data.get(cid, {}).get("points_item") or [])]
        poly_item.setPolygon(QPolygonF(pts))
        # relabel at centroid
        if pts:
            cx = sum(p.x() for p in pts)/len(pts); cy = sum(p.y() for p in pts)/len(pts)
            txt.setPos(cx + 6.0, cy + 6.0)
        # move handles if selected
        if cid == self._selected_poly_id:
            self._clear_handles()
            self._build_handles_for_selected()

    def _on_vertex_drag(self, cid: str, idx: int, x: float, y: float):
        pts = self._polys_data.get(cid, {}).get("points_item", [])
        if 0 <= idx < len(pts):
            pts[idx] = (float(x), float(y))
            self._update_polygon_graphics(cid)
            self._dirty = True

    def _point_to_segment_dist(self, px, py, x1, y1, x2, y2) -> float:
        vx, vy = x2 - x1, y2 - y1
        wx, wy = px - x1, py - y1
        denom = float(vx*vx + vy*vy)
        t = 0.0 if denom == 0 else max(0.0, min(1.0, (wx*vx + wy*vy) / denom))
        cx, cy = x1 + t*vx, y1 + t*vy
        dx, dy = px - cx, py - cy
        return math.hypot(dx, dy)

    def _closest_edge_index(self, cid: str, x: float, y: float) -> Tuple[int, float]:
        """Return (index_before, distance). Edge is between index and (index+1)%n."""
        pts = self._polys_data.get(cid, {}).get("points_item", [])
        n = len(pts)
        best_i, best_d = -1, 1e9
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            d = self._point_to_segment_dist(x, y, float(x1), float(y1), float(x2), float(y2))
            if d < best_d:
                best_i, best_d = i, d
        return best_i, best_d

    def _poly_hit_test(self, item_pt: QPointF) -> Optional[str]:
        """Return cid of topmost polygon under item_pt (item coords)."""
        for cid, (poly_item, _) in reversed(list(self._polys_items.items())):
            try:
                if poly_item.contains(item_pt):
                    return cid
            except Exception:
                continue
        return None

    def _push_move_vertex(self, cid: str, idx: int, x0: float, y0: float, x1: float, y1: float):
        self._undo_stack.push(_MoveVertexCmd(self, cid, idx, x0, y0, x1, y1))

    def _push_insert_vertex(self, cid: str, idx: int, x: float, y: float):
        self._undo_stack.push(_InsertVertexCmd(self, cid, idx, x, y))

    # ----- Clear / load / save -----
    def _clear_annotations(self) -> None:
        # temp poly
        self._poly_cancel_temp()
        # points
        for cid, (ell, txt) in list(self._points_items.items()):
            try:
                if ell.scene():
                    ell.scene().removeItem(ell)
                if txt.scene():
                    txt.scene().removeItem(txt)
            except Exception:
                pass
        self._points_items.clear()
        self._points_data.clear()
        # polygons
        for cid, (poly, txt) in list(self._polys_items.items()):
            try:
                if poly.scene():
                    poly.scene().removeItem(poly)
                if txt.scene():
                    txt.scene().removeItem(txt)
            except Exception:
                pass
        self._polys_items.clear()
        self._polys_data.clear()
        # handles & selection
        self._select_polygon(None)
        self._clear_handles()
        # undo stack
        self._undo_stack.clear()
        self._dirty = False

    def _sidecar_path_for(self, image_path: Path) -> Path:
        return image_path.with_suffix(image_path.suffix + ".ann.json")

    def _original_image_size(self, image_path: Path) -> Tuple[int, int]:
        try:
            reader = QImageReader(str(image_path))
            sz = reader.size()
            if sz.isValid():
                return int(sz.width()), int(sz.height())
        except Exception:
            pass
        pix = self.strip.pix_item.pixmap()
        return int(pix.width()), int(pix.height())

    def _image_rel(self, image_path: Path) -> str:
        for root in roots_for_read(self._target):
            try:
                rel = image_path.resolve().relative_to(root.resolve())
                return rel.as_posix()
            except Exception:
                continue
        return str(image_path)

    def _load_sidecar_for_current(self) -> None:
        self._clear_annotations()
        p = self.current_path()
        if not p:
            self._loaded_for_path = None
            self._loaded_pix_size = (0.0, 0.0)
            return

        sc = self._sidecar_path_for(p)
        if sc.exists():
            try:
                doc = json.loads(sc.read_text(encoding="utf-8"))
            except Exception:
                doc = {}
        else:
            doc = {}

        pix = self.strip.pix_item.pixmap()
        w_pix = float(pix.width()) or 1.0
        h_pix = float(pix.height()) or 1.0
        W = float((doc.get("image_size", [0, 0]) or [0, 0])[0]) or w_pix
        H = float((doc.get("image_size", [0, 0]) or [0, 0])[1]) or h_pix
        sx = w_pix / W
        sy = h_pix / H

        # Points
        for pt in (doc.get("points", []) or []):
            xy = pt.get("xy", [None, None])
            if not isinstance(xy, (list, tuple)) or len(xy) != 2:
                continue
            x_item = float(xy[0]) * sx
            y_item = float(xy[1]) * sy
            label = pt.get("label", "") or f"P{len(self._points_items)+1}"
            cid = pt.get("id") or uuid.uuid4().hex[:8]
            ell, txt = self._create_point_graphics(cid, x_item, y_item, label)
            self._register_point(cid, x_item, y_item, label, ell, txt)

        # Polygons
        for poly in (doc.get("polygons", []) or []):
            pts = poly.get("points", []) or []
            if len(pts) < 3:
                continue
            points_item = [(float(x) * sx, float(y) * sy) for (x, y) in pts]
            label = poly.get("label", "") or f"Φ{len(self._polys_items)+1}"
            cid = poly.get("id") or uuid.uuid4().hex[:8]
            poly_item, txt = self._create_polygon_graphics(cid, points_item, label)
            self._register_polygon(cid, points_item, label, poly_item, txt)

        self._dirty = False
        self._loaded_for_path = p
        self._loaded_pix_size = (w_pix, h_pix)

    def _save_prev_if_dirty(self) -> None:
        if not self._dirty or not self._loaded_for_path:
            return
        prev_p = self._loaded_for_path
        w_pix, h_pix = self._loaded_pix_size
        if w_pix <= 0 or h_pix <= 0:
            pix = self.strip.pix_item.pixmap()
            w_pix = float(pix.width()) or 1.0
            h_pix = float(pix.height()) or 1.0
        W, H = self._original_image_size(prev_p)
        sx = (float(W) / float(w_pix)) if w_pix > 0 else 1.0
        sy = (float(H) / float(h_pix)) if h_pix > 0 else 1.0
        self._write_sidecar(prev_p, sx, sy)
        self._dirty = False

    def _save_current_if_dirty(self, force: bool = False) -> None:
        if not force and not self._dirty:
            return
        p = self.current_path()
        if not p:
            return
        pix = self.strip.pix_item.pixmap()
        w_pix = float(pix.width()) or 1.0
        h_pix = float(pix.height()) or 1.0
        W, H = self._original_image_size(p)
        sx = (float(W) / float(w_pix)) if w_pix > 0 else 1.0
        sy = (float(H) / float(h_pix)) if h_pix > 0 else 1.0
        self._write_sidecar(p, sx, sy)
        self._dirty = False
        self._loaded_for_path = p
        self._loaded_pix_size = (w_pix, h_pix)

    def _write_sidecar(self, path: Path, sx: float, sy: float) -> None:
        points: List[Dict] = []
        for cid, dat in self._points_data.items():
            x_item, y_item = dat.get("xy_item", (0.0, 0.0))
            points.append({
                "id": cid,
                "label": dat.get("label", ""),
                "xy": [float(x_item) * sx, float(y_item) * sy],
            })

        polygons: List[Dict] = []
        for cid, dat in self._polys_data.items():
            pts = dat.get("points_item", []) or []
            polygons.append({
                "id": cid,
                "label": dat.get("label", ""),
                "points": [[float(x) * sx, float(y) * sy] for (x, y) in pts],
            })

        W, H = self._original_image_size(path)
        doc = {
            "schema": "starBoard.annotation/v1",
            "image_rel": self._image_rel(path),
            "image_size": [int(W), int(H)],
            "points": points,
            "polygons": polygons,
        }
        sc = self._sidecar_path_for(path)
        try:
            sc.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        except Exception:
            pass


# -------------------- Draggable vertex handle --------------------
class _VertexHandle(QGraphicsEllipseItem):
    """
    Small circle that stays constant screen size; dragging updates polygon vertex.
    """
    def __init__(self, view: "AnnotatorView", cid: str, idx: int, x: float, y: float, r: float = 4.0):
        super().__init__(-r, -r, 2*r, 2*r, parent=view.strip.pix_item)
        self.view = view
        self.cid = cid
        self.idx = int(idx)
        self.setPos(QPointF(float(x), float(y)))
        self.setPen(QPen(QColor(255, 170, 0))); self.setBrush(QBrush(QColor(255, 170, 0)))
        self.setZValue(1000)  # above polygons
        # Keep size constant while zooming
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self._drag_start = QPointF()

    def mousePressEvent(self, ev):
        self._drag_start = self.pos()
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        super().mouseMoveEvent(ev)
        p = self.pos()
        self.view._on_vertex_drag(self.cid, self.idx, float(p.x()), float(p.y()))

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        p0 = self._drag_start; p1 = self.pos()
        if (p0 - p1).manhattanLength() > 0.5:
            self.view._push_move_vertex(self.cid, self.idx,
                                        float(p0.x()), float(p0.y()),
                                        float(p1.x()), float(p1.y()))


# -------------------- Undo Commands --------------------
class _AddPointCmd(QUndoCommand):
    def __init__(self, view: AnnotatorView, cid: str, x_item: float, y_item: float, label: str):
        super().__init__(f"Add point {label}")
        self.view = view
        self.cid = cid
        self.x = float(x_item); self.y = float(y_item)
        self.label = label
        self._ell = None
        self._txt = None

    def redo(self):
        if self._ell is None or self._txt is None:
            ell, txt = self.view._create_point_graphics(self.cid, self.x, self.y, self.label)
            self._ell, self._txt = ell, txt
        self.view._register_point(self.cid, self.x, self.y, self.label, self._ell, self._txt)

    def undo(self):
        self.view._unregister_point(self.cid)


class _AddPolygonCmd(QUndoCommand):
    def __init__(self, view: AnnotatorView, cid: str, points_item: List[Tuple[float, float]], label: str):
        super().__init__(f"Add polygon {label}")
        self.view = view
        self.cid = cid
        self.points = [(float(x), float(y)) for (x, y) in points_item]
        self.label = label
        self._poly = None
        self._txt = None

    def redo(self):
        if self._poly is None or self._txt is None:
            poly, txt = self.view._create_polygon_graphics(self.cid, self.points, self.label)
            self._poly, self._txt = poly, txt
        self.view._register_polygon(self.cid, self.points, self.label, self._poly, self._txt)
        # Auto‑select new polygon for immediate editing
        self.view._select_polygon(self.cid)

    def undo(self):
        self.view._unregister_polygon(self.cid)
        if self.view._selected_poly_id == self.cid:
            self.view._select_polygon(None)


class _MoveVertexCmd(QUndoCommand):
    def __init__(self, view: AnnotatorView, cid: str, idx: int, x0: float, y0: float, x1: float, y1: float):
        super().__init__(f"Move vertex {idx}")
        self.view = view; self.cid = cid; self.idx = int(idx)
        self.x0, self.y0, self.x1, self.y1 = float(x0), float(y0), float(x1), float(y1)

    def redo(self):
        pts = self.view._polys_data.get(self.cid, {}).get("points_item", [])
        if 0 <= self.idx < len(pts):
            pts[self.idx] = (self.x1, self.y1)
            self.view._update_polygon_graphics(self.cid); self.view._dirty = True

    def undo(self):
        pts = self.view._polys_data.get(self.cid, {}).get("points_item", [])
        if 0 <= self.idx < len(pts):
            pts[self.idx] = (self.x0, self.y0)
            self.view._update_polygon_graphics(self.cid); self.view._dirty = True


class _InsertVertexCmd(QUndoCommand):
    def __init__(self, view: AnnotatorView, cid: str, idx: int, x: float, y: float):
        super().__init__(f"Insert vertex at {idx}")
        self.view = view; self.cid = cid; self.idx = int(idx); self.x = float(x); self.y = float(y)

    def redo(self):
        pts = self.view._polys_data.get(self.cid, {}).get("points_item", [])
        if pts:
            idx = max(0, min(self.idx, len(pts)))
            pts.insert(idx, (self.x, self.y))
            self.view._update_polygon_graphics(self.cid); self.view._dirty = True

    def undo(self):
        pts = self.view._polys_data.get(self.cid, {}).get("points_item", [])
        if 0 <= self.idx < len(pts):
            del pts[self.idx]
            self.view._update_polygon_graphics(self.cid); self.view._dirty = True


class _DeletePolygonCmd(QUndoCommand):
    def __init__(self, view: AnnotatorView, cid: str):
        super().__init__("Delete polygon")
        self.view = view; self.cid = cid
        dat = dict(view._polys_data.get(cid, {}))
        self.label = dat.get("label", "")
        self.points = list(dat.get("points_item", []))

    def redo(self):
        self.view._unregister_polygon(self.cid)
        if self.view._selected_poly_id == self.cid:
            self.view._select_polygon(None)

    def undo(self):
        poly, txt = self.view._create_polygon_graphics(self.cid, self.points, self.label)
        self.view._register_polygon(self.cid, self.points, self.label, poly, txt)
