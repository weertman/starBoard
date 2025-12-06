from __future__ import annotations

from pathlib import Path
import json
import uuid
import math
from typing import List, Dict, Tuple, Optional

from PySide6.QtCore import Qt, Signal, QEvent, QPointF
from PySide6.QtGui import (
    QPen, QBrush, QColor, QImageReader, QPolygonF,
    QUndoStack, QUndoCommand, QAction,
    QPainterPath, QPainterPathStroker
)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QToolButton, QButtonGroup,
    QGraphicsEllipseItem, QGraphicsSimpleTextItem, QGraphicsPolygonItem, QGraphicsItem, QDoubleSpinBox
)

from src.ui.image_strip_filters import FilteredImageStrip
from src.data.archive_paths import roots_for_read


class AnnotatorViewSecond(QWidget):
    """
    Second-order viewer with improved polygon/point tooling.

    Tools:
      - Select (default): click polygon to edit (drag vertices; insert at edge; Delete to remove)
      - Polygon       : click to add vertices; Enter/double-click to finish; Esc/Right-click to cancel
                        - Close by clicking the first vertex (≈8 px tolerance)
                        - Live ghost preview; hold Shift to constrain segment angles (45° steps)
      - Point         : click to add a labeled marker; markers are draggable with Undo support

    QoL:
      - Space-to-pan (temporary Hand) with cursor feedback
      - Edge-tolerant picking (predictable at any zoom)
      - Mid-edge “+” handles for quick vertex insertion
      - Constant-screen-size handles
      - Undo/Redo for all edits; non-destructive image filters
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
        self.btn_select.setChecked(True)  # default: Select

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

        # ---- Adjustments (now numeric fields) ----
        def _mk_spin(tt: str, minv: float, maxv: float, step: float, decimals: int, default: float) -> QDoubleSpinBox:
            w = QDoubleSpinBox()
            w.setRange(float(minv), float(maxv))
            w.setDecimals(int(decimals))
            w.setSingleStep(float(step))
            w.setValue(float(default))
            w.setMaximumWidth(110)
            w.setToolTip(tt)
            return w

        bar.addSpacing(18)
        bar.addWidget(QLabel("B"))
        self.spn_bright = _mk_spin("Brightness (−1.00 … +1.00)", -1.0, 1.0, 0.05, 2, 0.0)
        bar.addWidget(self.spn_bright)

        bar.addSpacing(8)
        bar.addWidget(QLabel("C"))
        self.spn_contrast = _mk_spin("Contrast (−1.00 … +1.00)", -1.0, 1.0, 0.05, 2, 0.0)
        bar.addWidget(self.spn_contrast)

        bar.addSpacing(8)
        bar.addWidget(QLabel("S"))
        self.spn_sat = _mk_spin("Saturation (0.00 … 2.00)", 0.0, 2.0, 0.05, 2, 1.0)
        bar.addWidget(self.spn_sat)

        bar.addStretch(1)
        bar.addWidget(self.btn_save)
        lay.addLayout(bar)

        # ---- Image view (Filtered) ----
        self.strip = FilteredImageStrip(files=[], long_edge=768)
        lay.addWidget(self.strip, 1)
        self.strip.pixmapResized.connect(self._on_pixmap_resized)

        # Intercept mouse on the viewport and keys on the view
        self.strip.view.viewport().installEventFilter(self)
        self.strip.view.installEventFilter(self)

        # Navigation from the strip
        self.strip.btn_prev.clicked.connect(self._after_navigate)
        self.strip.btn_next.clicked.connect(self._after_navigate)

        # Adjustments -> strip filters
        self.spn_bright.valueChanged.connect(self._on_adjust_changed)
        self.spn_contrast.valueChanged.connect(self._on_adjust_changed)
        self.spn_sat.valueChanged.connect(self._on_adjust_changed)
        self._on_adjust_changed()  # initialize

        # Actions / shortcuts
        self.btn_save.clicked.connect(self.save_current_annotations)
        self._act_undo = QAction("Undo", self); self._act_undo.setShortcut("Ctrl+Z")
        self._act_undo.triggered.connect(lambda: self._undo_stack.undo()); self.addAction(self._act_undo)
        self._act_redo = QAction("Redo", self); self._act_redo.setShortcut("Ctrl+Shift+Z")
        self._act_redo.triggered.connect(lambda: self._undo_stack.redo()); self.addAction(self._act_redo)

        # ---- Annotation state ----
        self._points_items: Dict[str, Tuple[QGraphicsEllipseItem, QGraphicsSimpleTextItem]] = {}
        self._points_data: Dict[str, Dict] = {}

        self._polys_items: Dict[str, Tuple[QGraphicsPolygonItem, QGraphicsSimpleTextItem]] = {}
        self._polys_data: Dict[str, Dict] = {}

        self._dirty: bool = False
        self._loaded_for_path: Optional[Path] = None
        self._loaded_pix_size: Tuple[float, float] = (0.0, 0.0)

        # Polygon-in-progress
        self._poly_temp_points: List[Tuple[float, float]] = []
        self._poly_temp_item: Optional[QGraphicsPolygonItem] = None

        # Selection & edit
        self._selected_poly_id: Optional[str] = None
        self._vertex_handles: List[_VertexHandle] = []
        self._edge_handles: List[_MidpointHandle] = []
        self._press_scene: Optional[QPointF] = None
        self._press_moved: bool = False

        # Space-to-pan
        self._space_down: bool = False
        for b in (self.btn_select, self.btn_point, self.btn_poly):
            b.toggled.connect(self._update_cursor)
        self._update_cursor()

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
        # Mouse on viewport; let the view pan if Space is held
        if obj is self.strip.view.viewport():
            t = ev.type()

            # If Space is down, do not consume mouse events (allow panning)
            if t in (QEvent.MouseButtonPress, QEvent.MouseMove, QEvent.MouseButtonRelease) and self._space_down:
                return False

            # ------- Select = pan by default; click to select polygon -------
            if self.btn_select.isChecked():
                if t == QEvent.MouseButtonPress and getattr(ev, "button", lambda: None)() == Qt.LeftButton:
                    self._press_scene = self.strip.view.mapToScene(int(ev.position().x()), int(ev.position().y()))
                    self._press_moved = False
                    return False  # don't consume → panning works
                if t == QEvent.MouseMove and self._press_scene is not None:
                    self._press_moved = True
                    return False
                if t == QEvent.MouseButtonRelease and getattr(ev, "button", lambda: None)() == Qt.LeftButton:
                    if self._press_scene is not None:
                        release_scene = self.strip.view.mapToScene(int(ev.position().x()), int(ev.position().y()))
                        dx = release_scene.x() - self._press_scene.x()
                        dy = release_scene.y() - self._press_scene.y()
                        if (dx*dx + dy*dy) <= 9.0:  # <=3px -> treat as click
                            item_pt = self.strip.pix_item.mapFromScene(release_scene)
                            cid = self._poly_hit_test(item_pt)
                            self._select_polygon(cid)
                    self._press_scene = None
                    self._press_moved = False
                    return False

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

                # --- NEW: if click is on a vertex handle, let Qt handle the drag and skip adding vertices ---
                for h in self._vertex_handles:
                    try:
                        if h.contains(h.mapFromItem(self.strip.pix_item, item_pt)):
                            # clicked on existing handle — allow Qt drag, don't consume
                            return False
                    except Exception:
                        continue

                x_item, y_item = float(item_pt.x()), float(item_pt.y())
                if self.btn_point.isChecked():
                    self._push_add_point(x_item, y_item)
                    return True

                if self.btn_poly.isChecked():
                    # If close to first vertex, close polygon
                    if len(self._poly_temp_points) >= 3:
                        first = QPointF(self._poly_temp_points[0][0], self._poly_temp_points[0][1])
                        if self._dist_item(item_pt, first) <= self._px_to_item_dx(8.0):
                            self._poly_finalize_if_possible()
                            return True
                    # add a vertex
                    self._poly_add_vertex(x_item, y_item)
                    self._poly_update_temp_item()
                    return True

            # Finish polygon by double-click
            if t == QEvent.MouseButtonDblClick:
                if self.btn_poly.isChecked():
                    self._poly_finalize_if_possible()
                    return True

                # In Select mode, we no longer insert vertices on double-click.
                # Only Shift+Right-click will insert (handled in the RightButton press branch).
                return False

            # Shift + Right-click in Select mode: insert a vertex
            if (t == QEvent.MouseButtonPress
                    and getattr(ev, "button", lambda: None)() == Qt.RightButton
                    and self.btn_select.isChecked()
                    and self._selected_poly_id):

                mods = getattr(ev, "modifiers", lambda: Qt.NoModifier)()
                if mods & Qt.ShiftModifier:
                    posf = getattr(ev, "position", None)
                    if posf:
                        posf = posf()
                        scene_pt = self.strip.view.mapToScene(int(posf.x()), int(posf.y()))
                        item_pt = self.strip.pix_item.mapFromScene(scene_pt)
                        if self.strip.pix_item.contains(item_pt):
                            x_item, y_item = float(item_pt.x()), float(item_pt.y())

                            # Prefer inserting "next to" the nearest vertex if within a small pixel tolerance;
                            # otherwise, insert on the closest edge at the click point.
                            vidx, vdist = self._closest_vertex_index(self._selected_poly_id, x_item, y_item)
                            if vidx >= 0 and vdist <= self._px_to_item_dx(8.0):
                                # insert after this vertex
                                self._push_insert_vertex(self._selected_poly_id, vidx + 1, x_item, y_item)
                                return True

                            eidx, edist = self._closest_edge_index(self._selected_poly_id, x_item, y_item)
                            if eidx >= 0 and edist <= self._px_to_item_dx(8.0):
                                self._push_insert_vertex(self._selected_poly_id, eidx + 1, x_item, y_item)
                                return True

            # Right click cancels in‑progress polygon
            if t == QEvent.MouseButtonPress and getattr(ev, "button", lambda: None)() == Qt.RightButton:
                if self.btn_poly.isChecked() and self._poly_temp_points:
                    self._poly_cancel_temp()
                    return True

            # Live ghost preview while moving the mouse (Polygon tool)
            if t == QEvent.MouseMove and self.btn_poly.isChecked() and self._poly_temp_points:
                posf = getattr(ev, "position", None)
                if posf:
                    posf = posf()
                    scene_pt = self.strip.view.mapToScene(int(posf.x()), int(posf.y()))
                    item_pt = self.strip.pix_item.mapFromScene(scene_pt)
                    x_item, y_item = float(item_pt.x()), float(item_pt.y())
                    # Shift = constrain angle to 45° steps
                    if getattr(ev, "modifiers", lambda: Qt.NoModifier)() & Qt.ShiftModifier:
                        x_item, y_item = self._constrain_by_angle(self._poly_temp_points[-1], (x_item, y_item))
                    self._poly_update_temp_item(preview=(x_item, y_item))
                    return True

        # Keys on the view (Enter/Esc/Backspace/Delete/Space)
        if obj is self.strip.view and ev.type() == QEvent.KeyPress:
            key = ev.key()
            if key == Qt.Key_Space:
                self._space_down = True
                self.strip.view.viewport().setCursor(Qt.ClosedHandCursor)
                return False
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

        if obj is self.strip.view and ev.type() == QEvent.KeyRelease:
            if ev.key() == Qt.Key_Space:
                self._space_down = False
                self._update_cursor()
                return False

        return super().eventFilter(obj, ev)

    # ---------- Adjustments ----------
    def _on_adjust_changed(self, *_):
        # Numeric fields already in target ranges:
        # brightness ∈ [-1, +1], contrast ∈ [-1, +1], saturation ∈ [0, 2]
        b = float(self.spn_bright.value())
        c = float(self.spn_contrast.value())
        s = float(self.spn_sat.value())
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

    def _update_cursor(self, *_):
        if self._space_down:
            self.strip.view.viewport().setCursor(Qt.ClosedHandCursor); return
        if self.btn_select.isChecked():
            self.strip.view.viewport().setCursor(Qt.OpenHandCursor)
        elif self.btn_point.isChecked() or self.btn_poly.isChecked():
            self.strip.view.viewport().setCursor(Qt.CrossCursor)
        else:
            self.strip.view.viewport().unsetCursor()

    # ----- Points -----
    def _push_add_point(self, x_item: float, y_item: float, label: Optional[str] = None) -> None:
        cid = uuid.uuid4().hex[:8]
        label = label if label is not None else f"P{len(self._points_items) + 1}"
        self._undo_stack.push(_AddPointCmd(self, cid, x_item, y_item, label))

    def _create_point_graphics(self, cid: str, x_item: float, y_item: float, label: str):
        # Draggable point handle (constant screen size)
        ell = _PointHandle(self, cid, x_item, y_item, r=5.0)
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
                if ell.scene(): ell.scene().removeItem(ell)
                if txt.scene(): txt.scene().removeItem(txt)
            except Exception:
                pass
        self._points_data.pop(cid, None)
        self._dirty = True

    def _on_point_drag(self, cid: str, x: float, y: float):
        dat = self._points_data.get(cid)
        if not dat: return
        dat["xy_item"] = (float(x), float(y))
        ell, txt = self._points_items.get(cid, (None, None))
        if txt:
            txt.setPos(float(x) + 6.0, float(y) - 6.0)
        self._dirty = True

    def _push_move_point(self, cid: str, x0: float, y0: float, x1: float, y1: float):
        self._undo_stack.push(_MovePointCmd(self, cid, x0, y0, x1, y1))

    # ----- Polygon (in‑progress and finalized) -----
    def _closest_vertex_index(self, cid: str, x: float, y: float) -> Tuple[int, float]:
        pts = self._polys_data.get(cid, {}).get("points_item", []) or []
        best_i, best_d = -1, 1e9
        for i, (vx, vy) in enumerate(pts):
            d = math.hypot(float(x) - float(vx), float(y) - float(vy))
            if d < best_d:
                best_i, best_d = i, d
        return best_i, best_d

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

    def _poly_update_temp_item(self, preview: Optional[Tuple[float, float]] = None):
        if self._poly_temp_item is None:
            return
        pts = list(self._poly_temp_points)
        if preview is not None and pts:
            pts = pts + [preview]
        poly = QPolygonF([QPointF(x, y) for (x, y) in pts])
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
        # don't steal mouse → panning works
        poly_item.setAcceptedMouseButtons(Qt.NoButton)

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
                if poly_item.scene(): poly_item.scene().removeItem(poly_item)
                if txt.scene(): txt.scene().removeItem(txt)
            except Exception:
                pass
        self._polys_data.pop(cid, None)
        self._dirty = True

    # ----- Selection & vertex editing -----
    def _select_polygon(self, cid: Optional[str]):
        if cid == self._selected_poly_id:
            return
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
        for h in getattr(self, "_edge_handles", []):
            try:
                if h.scene(): h.scene().removeItem(h)
            except Exception:
                pass
        self._edge_handles.clear()

    def _build_handles_for_selected(self):
        cid = self._selected_poly_id
        if not cid: return
        pts = self._polys_data.get(cid, {}).get("points_item", [])
        self._vertex_handles = []
        for i, (x, y) in enumerate(pts):
            self._vertex_handles.append(_VertexHandle(self, cid, i, float(x), float(y)))
        # mid-edge handles for quick insertion
        self._edge_handles = []
        n = len(pts)
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            mx, my = 0.5*(float(x1)+float(x2)), 0.5*(float(y1)+float(y2))
            self._edge_handles.append(_MidpointHandle(self, cid, i, mx, my))

    def _update_polygon_graphics(self, cid: str, *, refresh_handles: bool = True) -> None:
        tup = self._polys_items.get(cid)
        if not tup:
            return
        poly_item, txt = tup

        pts_list = (self._polys_data.get(cid, {}).get("points_item") or [])
        qpts = [QPointF(float(x), float(y)) for (x, y) in pts_list]
        poly_item.setPolygon(QPolygonF(qpts))

        # Reposition the label at the centroid-ish
        if qpts:
            cx = sum(p.x() for p in qpts) / len(qpts)
            cy = sum(p.y() for p in qpts) / len(qpts)
            txt.setPos(cx + 6.0, cy + 6.0)

        # IMPORTANT: do not tear down the active handle if we're mid-drag
        if cid == self._selected_poly_id:
            if refresh_handles:
                self._clear_handles()
                self._build_handles_for_selected()
            else:
                # Update existing handles in place
                if len(self._vertex_handles) == len(pts_list):
                    for h, (x, y) in zip(self._vertex_handles, pts_list):
                        try:
                            h.setPos(QPointF(float(x), float(y)))
                        except Exception:
                            pass

    def _on_vertex_drag(self, cid: str, idx: int, x: float, y: float):
        pts = self._polys_data.get(cid, {}).get("points_item", [])
        if 0 <= idx < len(pts):
            pts[idx] = (float(x), float(y))
            # Do NOT rebuild handles while dragging — keeps the grabbed item alive
            self._update_polygon_graphics(cid, refresh_handles=False)
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

    def _dist_item(self, p: QPointF, q: QPointF) -> float:
        return math.hypot(float(p.x()-q.x()), float(p.y()-q.y()))

    def _px_to_item_dx(self, px: float) -> float:
        """Convert screen pixels to item-space distance at current zoom."""
        v = self.strip.view
        s1 = v.mapToScene(0, 0); s2 = v.mapToScene(int(px), 0)
        i1 = self.strip.pix_item.mapFromScene(s1); i2 = self.strip.pix_item.mapFromScene(s2)
        return math.hypot(float(i2.x()-i1.x()), float(i2.y()-i1.y()))

    def _constrain_by_angle(self, p0_xy: Tuple[float,float], p1_xy: Tuple[float,float], step_deg: float = 45.0) -> Tuple[float,float]:
        x0, y0 = p0_xy; x1, y1 = p1_xy
        dx, dy = (x1 - x0), (y1 - y0)
        r = math.hypot(dx, dy)
        if r <= 1e-6: return x1, y1
        ang = math.degrees(math.atan2(dy, dx))
        k = round(ang / step_deg)
        a = math.radians(k * step_deg)
        return x0 + r*math.cos(a), y0 + r*math.sin(a)

    def _poly_hit_test(self, item_pt: QPointF) -> Optional[str]:
        """Topmost polygon under 'item_pt' using fill OR stroked-edge hit (screen-pixel tolerant)."""
        tol = max(3.0, min(12.0, self._px_to_item_dx(6.0)))  # ~6px in item units, clamped
        for cid, (poly_item, _) in reversed(list(self._polys_items.items())):
            try:
                poly = poly_item.polygon()
                if poly.isEmpty():
                    continue
                # Fill test
                path_fill = QPainterPath(poly[0])
                for i in range(1, poly.count()):
                    path_fill.lineTo(poly[i])
                path_fill.closeSubpath()
                if path_fill.contains(item_pt):
                    return cid
                # Edge (stroke) test
                path_edge = QPainterPath(poly[0])
                for i in range(1, poly.count()):
                    path_edge.lineTo(poly[i])
                path_edge.closeSubpath()
                stroker = QPainterPathStroker()
                stroker.setWidth(tol)
                if stroker.createStroke(path_edge).contains(item_pt):
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
        self._poly_cancel_temp()
        for cid, (ell, txt) in list(self._points_items.items()):
            try:
                if ell.scene(): ell.scene().removeItem(ell)
                if txt.scene(): txt.scene().removeItem(txt)
            except Exception:
                pass
        self._points_items.clear()
        self._points_data.clear()
        for cid, (poly, txt) in list(self._polys_items.items()):
            try:
                if poly.scene(): poly.scene().removeItem(poly)
                if txt.scene(): txt.scene().removeItem(txt)
            except Exception:
                pass
        self._polys_items.clear()
        self._polys_data.clear()
        self._select_polygon(None)
        self._clear_handles()
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

    def _on_pixmap_resized(self, old_w: int, old_h: int, new_w: int, new_h: int):
        """
        When the underlying pixmap changes size (e.g., preview -> full-res),
        rescale all overlay coordinates so geometry stays glued to the image.
        Also update the cached loaded-pix size used by save-on-navigate.
        """
        if old_w <= 0 or old_h <= 0:
            # Nothing to rescale from
            self._loaded_pix_size = (float(new_w), float(new_h))
            return
        if new_w <= 0 or new_h <= 0:
            return
        if old_w == new_w and old_h == new_h:
            return

        kx = float(new_w) / float(old_w)
        ky = float(new_h) / float(old_h)

        # --- rescale in-progress polygon
        if self._poly_temp_points:
            self._poly_temp_points = [(x * kx, y * ky) for (x, y) in self._poly_temp_points]
            self._poly_update_temp_item()

        # --- rescale points
        for cid, dat in list(self._points_data.items()):
            x, y = dat.get("xy_item", (0.0, 0.0))
            nx, ny = (float(x) * kx, float(y) * ky)
            dat["xy_item"] = (nx, ny)
            it = self._points_items.get(cid)
            if it:
                ell, txt = it
                # keep marker radius constant in item units
                r = 4.0
                ell.setRect(nx - r, ny - r, 2 * r, 2 * r)
                txt.setPos(nx + 6.0, ny - 6.0)

        # --- rescale polygons
        for cid, dat in list(self._polys_data.items()):
            pts = dat.get("points_item", []) or []
            if not pts:
                continue
            dat["points_item"] = [(float(x) * kx, float(y) * ky) for (x, y) in pts]
            self._update_polygon_graphics(cid)

        # this ensures save-on-navigate writes correct scale
        self._loaded_pix_size = (float(new_w), float(new_h))


# -------------------- Draggable vertex handle --------------------
class _VertexHandle(QGraphicsEllipseItem):
    def __init__(self, view: "AnnotatorViewSecond", cid: str, idx: int, x: float, y: float, r: float = 4.0):
        super().__init__(-r, -r, 2*r, 2*r, parent=view.strip.pix_item)
        self.view = view
        self.cid = cid
        self.idx = int(idx)
        self.setPos(QPointF(float(x), float(y)))
        self.setPen(QPen(QColor(255, 170, 0))); self.setBrush(QBrush(QColor(255, 170, 0)))
        self.setZValue(1000)
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


class _MidpointHandle(QGraphicsEllipseItem):
    """Clickable '+' hotspot on polygon edges to insert a vertex."""
    def __init__(self, view: "AnnotatorViewSecond", cid: str, idx_before: int, x: float, y: float, r: float = 4.0):
        super().__init__(-r, -r, 2*r, 2*r, parent=view.strip.pix_item)
        self.view = view; self.cid = cid; self.idx_before = int(idx_before)
        self.setPos(QPointF(float(x), float(y)))
        c = QColor(255, 170, 0)
        self.setPen(QPen(c)); self.setBrush(QBrush(QColor(c.red(), c.green(), c.blue(), 180)))
        self.setZValue(900)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setToolTip("Insert vertex")
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, ev):
        """
        Only insert a new vertex if the user Shift + Right-clicks on this handle.
        Any other click (Left click, no Shift, etc.) is ignored so we don’t add
        unintended vertices.
        """
        try:
            if ev.button() == Qt.RightButton and (ev.modifiers() & Qt.ShiftModifier):
                p = self.pos()
                self.view._push_insert_vertex(self.cid, self.idx_before + 1, float(p.x()), float(p.y()))
                ev.accept()
                return
        except Exception:
            pass
        # Otherwise, ignore the event completely
        ev.ignore()


# -------------------- Point handle + Undo for moving points --------------------
class _PointHandle(QGraphicsEllipseItem):
    def __init__(self, view: "AnnotatorViewSecond", cid: str, x: float, y: float, r: float = 5.0):
        super().__init__(-r, -r, 2*r, 2*r, parent=view.strip.pix_item)
        self.view = view; self.cid = cid
        self.setPos(QPointF(float(x), float(y)))
        pen = QPen(QColor(0, 170, 255)); pen.setWidthF(1.5)
        self.setPen(pen); self.setBrush(QBrush(Qt.NoBrush))
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setZValue(950)
        self.setCursor(Qt.SizeAllCursor)
        self._drag_start = QPointF()

    def mousePressEvent(self, ev):
        self._drag_start = self.pos()
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        super().mouseMoveEvent(ev)
        p = self.pos()
        self.view._on_point_drag(self.cid, float(p.x()), float(p.y()))

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        p0 = self._drag_start; p1 = self.pos()
        if (p0 - p1).manhattanLength() > 0.5:
            self.view._push_move_point(self.cid,
                float(p0.x()), float(p0.y()),
                float(p1.x()), float(p1.y()))


# -------------------- Undo Commands --------------------
class _AddPointCmd(QUndoCommand):
    def __init__(self, view: AnnotatorViewSecond, cid: str, x_item: float, y_item: float, label: str):
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
    def __init__(self, view: AnnotatorViewSecond, cid: str, points_item: List[Tuple[float, float]], label: str):
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
        self.view._select_polygon(self.cid)

    def undo(self):
        self.view._unregister_polygon(self.cid)
        if self.view._selected_poly_id == self.cid:
            self.view._select_polygon(None)


class _MoveVertexCmd(QUndoCommand):
    def __init__(self, view: AnnotatorViewSecond, cid: str, idx: int, x0: float, y0: float, x1: float, y1: float):
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
    def __init__(self, view: AnnotatorViewSecond, cid: str, idx: int, x: float, y: float):
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


class _MovePointCmd(QUndoCommand):
    def __init__(self, view: "AnnotatorViewSecond", cid: str, x0: float, y0: float, x1: float, y1: float):
        super().__init__("Move point")
        self.view = view; self.cid = cid
        self.x0, self.y0, self.x1, self.y1 = float(x0), float(y0), float(x1), float(y1)

    def redo(self):
        self._apply(self.x1, self.y1)

    def undo(self):
        self._apply(self.x0, self.y0)

    def _apply(self, x, y):
        dat = self.view._points_data.get(self.cid)
        if not dat: return
        dat["xy_item"] = (float(x), float(y))
        ell, txt = self.view._points_items.get(self.cid, (None, None))
        if ell:
            ell.setPos(QPointF(float(x), float(y)))
        if txt:
            txt.setPos(float(x) + 6.0, float(y) - 6.0)
        self.view._dirty = True

# -------------------- Undo: Delete polygon --------------------
class _DeletePolygonCmd(QUndoCommand):
    def __init__(self, view: "AnnotatorViewSecond", cid: str):
        super().__init__("Delete polygon")
        self.view = view
        self.cid = cid
        dat = dict(view._polys_data.get(cid, {}))
        self.label = dat.get("label", "")
        self.points = list(dat.get("points_item", []))
        self._poly = None
        self._txt = None

    def redo(self):
        self.view._unregister_polygon(self.cid)
        if self.view._selected_poly_id == self.cid:
            self.view._select_polygon(None)

    def undo(self):
        if self._poly is None or self._txt is None:
            self._poly, self._txt = self.view._create_polygon_graphics(self.cid, self.points, self.label)
        self.view._register_polygon(self.cid, self.points, self.label, self._poly, self._txt)
        self.view._select_polygon(self.cid)