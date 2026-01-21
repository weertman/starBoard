# src/ui/image_strip.py
from __future__ import annotations
from PySide6.QtCore import Qt, QSize, QEvent, QPointF, Signal
from PySide6.QtGui import QPixmap, QImageReader, QAction, QColor, QCursor
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QLabel,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QToolTip
)

from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, Dict, Optional

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
    
    Also supports eyedropper mode for picking colors from the image.
    """

    # Notify listeners when the underlying pixmap size changes
    pixmapResized = Signal(int, int, int, int)   # old_w, old_h, new_w, new_h
    
    # Eyedropper signals
    eyedropperColorPicked = Signal(QColor)
    eyedropperCancelled = Signal()

    def __init__(self, files: List[Path] | None = None, long_edge: int = 768, initial_idx: int = 0, best_idx: int = 0, closest_idx: int | None = None, parent=None):
        super().__init__(parent)
        self.files: List[Path] = files or []
        # Clamp initial_idx to valid range
        if self.files and initial_idx > 0:
            self.idx = min(initial_idx, len(self.files) - 1)
        else:
            self.idx = 0
        # Track best photo index (default 0, can be set externally)
        self._best_idx = max(0, min(best_idx, len(self.files) - 1)) if self.files else 0
        # Track closest match index (for toggle behavior in roll-to-closest mode)
        # None means toggle is disabled; button always goes to best
        self._closest_idx: int | None = None
        if closest_idx is not None and self.files:
            self._closest_idx = max(0, min(closest_idx, len(self.files) - 1))
        self.long_edge = int(long_edge)
        self._rotation = 0.0
        self._scale = 1.0
        self._hires_loaded = False
        self._hires_trigger = 1.2
        self._rotate_key_down = False
        self._rotating = False
        self._drag_start_x = 0.0
        self._rotation_at_press = 0.0
        
        # Track whether user has ever interacted with this view (zoom/pan/rotate)
        # If True, we preserve their state during resize operations
        self._user_interacted = False
        
        # Eyedropper state
        self._eyedropper_active = False
        self._eyedropper_color = QColor()
        self._old_cursor: Optional[QCursor] = None
        self._old_drag_mode = None

        self.setFocusPolicy(Qt.StrongFocus)
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.btn_prev = QPushButton("◀")
        self.btn_next = QPushButton("▶")
        self.btn_best = QPushButton("★")
        self.btn_best.setFixedWidth(28)
        self.lbl_idx = QLabel("0/0")
        self._update_best_button_tooltip()

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
        self.btn_best.clicked.connect(self.go_to_best)

        lay.addWidget(self.btn_prev)
        lay.addWidget(self.view, 1)
        lay.addWidget(self.btn_next)
        lay.addWidget(self.btn_best)
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
        self._user_interacted = False  # Reset on new file set
        self._show_current(reset_view=True)

    def roll_to_index(self, target_idx: int) -> bool:
        """
        Set the current image to the specified index (O(1) operation).
        
        Args:
            target_idx: The index of the image to display
            
        Returns:
            True if the index was valid and applied, False otherwise
        """
        if not self.files or target_idx < 0 or target_idx >= len(self.files):
            return False
        
        if self.idx != target_idx:
            self.idx = target_idx
            self._rotation = 0.0
            self._scale = 1.0
            self._hires_loaded = False
            self._user_interacted = False  # Reset on image change
            self._show_current(reset_view=True)
        
        return True

    def set_best_idx(self, idx: int) -> None:
        """
        Set the index of the best photo for this strip.
        
        Args:
            idx: The index of the best photo in the files list
        """
        if self.files:
            self._best_idx = max(0, min(idx, len(self.files) - 1))
        else:
            self._best_idx = 0
        self._update_best_button_tooltip()

    def set_closest_idx(self, idx: int | None) -> None:
        """
        Set the index of the closest matching image (enables toggle mode).
        
        Args:
            idx: The index of the closest image, or None to disable toggle
        """
        if idx is not None and self.files:
            self._closest_idx = max(0, min(idx, len(self.files) - 1))
        else:
            self._closest_idx = None
        self._update_best_button_tooltip()

    def _update_best_button_tooltip(self) -> None:
        """Update the ★ button tooltip based on current mode."""
        if self._closest_idx is not None:
            self.btn_best.setToolTip("Toggle between best photo and closest match")
        else:
            self.btn_best.setToolTip("Go to best photo")

    def go_to_best(self) -> bool:
        """
        Navigate to the best photo, or toggle between best and closest.
        
        When closest_idx is set (roll-to-closest mode), this toggles:
        - If at best → go to closest
        - If at closest (or anywhere else) → go to best
        
        Returns:
            True if navigation succeeded, False otherwise
        """
        if self._closest_idx is not None:
            # Toggle mode: alternate between best and closest
            if self.idx == self._best_idx:
                return self.roll_to_index(self._closest_idx)
            else:
                return self.roll_to_index(self._best_idx)
        else:
            # Normal mode: always go to best
            return self.roll_to_index(self._best_idx)

    def roll_to_path(self, target_path: str) -> bool:
        """
        Set the current image to the one matching target_path.
        
        Matching is done by filename stem (without extension) to handle
        cases where cache paths use .png but originals use .jpg/.jpeg.
        
        Note: Prefer roll_to_index() when you have a precomputed index,
        as it's O(1) vs O(n) for this method.
        
        Args:
            target_path: Path to the target image (can be cache or original path)
            
        Returns:
            True if a matching image was found and selected, False otherwise
        """
        if not self.files or not target_path:
            return False
        
        target_stem = Path(target_path).stem
        
        for i, f in enumerate(self.files):
            if Path(f).stem == target_stem:
                if self.idx != i:
                    self.idx = i
                    self._rotation = 0.0
                    self._scale = 1.0
                    self._hires_loaded = False
                    self._user_interacted = False  # Reset on image change
                    self._show_current(reset_view=True)
                return True
        
        return False

    def set_view_height(self, h: int):
        """
        Set baseline minimum height.
        
        - If user has interacted with this view (zoom/pan/rotate), preserves their state
        - If user hasn't touched it, refits to the new size for optimal display
        """
        self.view.setMinimumHeight(max(80, int(h)))
        if not self._user_interacted:
            self.fit()

    def set_view_min_width(self, w: int):
        """
        Set baseline minimum width.
        
        - If user has interacted with this view (zoom/pan/rotate), preserves their state
        - If user hasn't touched it, refits to the new size for optimal display
        """
        self.view.setMinimumWidth(max(100, int(w)))
        if not self._user_interacted:
            self.fit()

    def prev(self):
        if not self.files: return
        self.idx = (self.idx - 1) % len(self.files)
        self._rotation = 0.0; self._scale = 1.0; self._hires_loaded = False
        self._user_interacted = False  # Reset on image change
        self._show_current(reset_view=True)

    def next(self):
        if not self.files: return
        self.idx = (self.idx + 1) % len(self.files)
        self._rotation = 0.0; self._scale = 1.0; self._hires_loaded = False
        self._user_interacted = False  # Reset on image change
        self._show_current(reset_view=True)

    def fit(self):
        self.view.resetTransform()
        self._scale = 1.0
        self.view.fitInView(self.pix_item, Qt.KeepAspectRatio)

    # -------- eyedropper mode --------
    def start_eyedropper(self) -> None:
        """Enter eyedropper mode for picking colors from the image."""
        print(f"[ImageStrip] start_eyedropper called, files={len(self.files)}")
        self._eyedropper_active = True
        self._old_cursor = self.cursor()
        self._old_drag_mode = self.view.dragMode()
        
        # Set cursor on multiple levels to ensure it shows
        self.setCursor(Qt.CrossCursor)
        self.view.setCursor(Qt.CrossCursor)
        self.view.viewport().setCursor(Qt.CrossCursor)
        
        # Disable drag mode so clicks aren't consumed
        self.view.setDragMode(QGraphicsView.NoDrag)
        
        # Enable mouse tracking at all levels
        self.setMouseTracking(True)
        self.view.setMouseTracking(True)
        self.view.viewport().setMouseTracking(True)
        
        # Grab focus and raise to front
        self.raise_()
        self.activateWindow()
        self.view.setFocus()
        
        print(f"[ImageStrip] eyedropper mode active, widget visible={self.isVisible()}, viewport size={self.view.viewport().size().width()}x{self.view.viewport().size().height()}")
    
    def stop_eyedropper(self) -> None:
        """Exit eyedropper mode."""
        self._eyedropper_active = False
        if self._old_cursor:
            self.view.setCursor(self._old_cursor)
        if self._old_drag_mode is not None:
            self.view.setDragMode(self._old_drag_mode)
        QToolTip.hideText()
    
    def is_eyedropper_active(self) -> bool:
        """Check if eyedropper mode is active."""
        return self._eyedropper_active
    
    def _get_color_at_viewport_pos(self, viewport_pos: QPointF) -> Optional[QColor]:
        """Get the color at a viewport position, or None if outside image."""
        # Map viewport position to scene coordinates
        scene_pos = self.view.mapToScene(int(viewport_pos.x()), int(viewport_pos.y()))
        # Map scene coordinates to item coordinates
        item_pos = self.pix_item.mapFromScene(scene_pos)
        
        # Get the pixmap
        pix = self.pix_item.pixmap()
        if pix.isNull():
            return None
        
        # Check bounds
        x, y = int(item_pos.x()), int(item_pos.y())
        if x < 0 or y < 0 or x >= pix.width() or y >= pix.height():
            return None
        
        # Get pixel color
        image = pix.toImage()
        return QColor(image.pixel(x, y))
    
    def _handle_eyedropper_move(self, ev) -> bool:
        """Handle mouse move in eyedropper mode. Returns True if handled."""
        if not self._eyedropper_active:
            return False
        
        pos = getattr(ev, "position", lambda: QPointF(0, 0))()
        color = self._get_color_at_viewport_pos(pos)
        print(f"[Eyedropper] move at {pos.x():.0f},{pos.y():.0f} color={color.name() if color else 'None'}")
        
        if color and color.isValid():
            self._eyedropper_color = color
            
            # Show tooltip with color preview
            hex_code = color.name().upper()
            r, g, b = color.red(), color.green(), color.blue()
            
            tooltip = f"""<div style="padding: 8px; background: #333; border-radius: 4px;">
                <div style="background-color: {hex_code}; 
                            width: 80px; height: 50px; 
                            border: 2px solid white; 
                            border-radius: 3px;
                            margin-bottom: 6px;">
                </div>
                <div style="color: white; font-family: monospace; font-size: 13px; font-weight: bold;">
                    {hex_code}
                </div>
                <div style="color: #ccc; font-family: monospace; font-size: 11px;">
                    R:{r} G:{g} B:{b}
                </div>
                <div style="color: #888; font-size: 10px; margin-top: 4px;">
                    Click to select
                </div>
            </div>"""
            
            global_pos = self.view.viewport().mapToGlobal(pos.toPoint())
            QToolTip.showText(global_pos, tooltip, self.view.viewport())
        else:
            QToolTip.hideText()
        
        return True
    
    def _handle_eyedropper_click(self, ev) -> bool:
        """Handle mouse click in eyedropper mode. Returns True if handled."""
        if not self._eyedropper_active:
            return False
        
        button = getattr(ev, "button", lambda: None)()
        
        if button == Qt.LeftButton:
            self.stop_eyedropper()
            if self._eyedropper_color.isValid():
                self.eyedropperColorPicked.emit(self._eyedropper_color)
            return True
        elif button == Qt.RightButton:
            self.stop_eyedropper()
            self.eyedropperCancelled.emit()
            return True
        
        return False

    # -------- event handling: wheel-zoom + continuous rotate + eyedropper --------
    def eventFilter(self, obj, ev):
        if obj is self.view.viewport():
            t = ev.type()
            
            # Eyedropper mode takes priority
            if self._eyedropper_active:
                # Debug: uncomment to see all events
                # if t in (QEvent.MouseMove, QEvent.MouseButtonPress, QEvent.Enter, QEvent.Leave):
                #     print(f"[ImageStrip.eventFilter] eyedropper event type={t}")
                
                if t == QEvent.MouseMove:
                    if self._handle_eyedropper_move(ev):
                        ev.accept()
                        return True
                elif t == QEvent.MouseButtonPress:
                    print(f"[ImageStrip] eyedropper click detected!")
                    if self._handle_eyedropper_click(ev):
                        ev.accept()
                        return True
                elif t == QEvent.Enter:
                    print(f"[ImageStrip] mouse entered viewport during eyedropper mode")
                elif t == QEvent.Leave:
                    print(f"[ImageStrip] mouse left viewport during eyedropper mode")
            
            # Normal wheel zoom
            if t == QEvent.Wheel:
                delta = ev.angleDelta().y()
                if delta == 0: return False
                factor = 1.0015 ** float(delta)
                self.view.scale(factor, factor)
                self._scale *= factor
                self._user_interacted = True  # User zoomed - preserve their state
                if not self._hires_loaded and self._scale >= self._hires_trigger:
                    self._upgrade_to_fullres()
                ev.accept(); return True
            
            # Detect pan via ScrollHandDrag (mouse drag without rotation key)
            if t == QEvent.MouseButtonRelease and not self._rotating:
                if self.view.dragMode() == QGraphicsView.ScrollHandDrag:
                    self._user_interacted = True  # User panned - preserve their state
            
            # Rotation mode
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
                self._user_interacted = True  # User rotated - preserve their state
                self.view.setDragMode(QGraphicsView.ScrollHandDrag)
                ev.accept(); return True

        if obj is self.view:
            t = ev.type()
            # Eyedropper escape key
            if self._eyedropper_active and t == QEvent.KeyPress and ev.key() == Qt.Key_Escape:
                self.stop_eyedropper()
                self.eyedropperCancelled.emit()
                ev.accept()
                return True
            # Rotation key handling
            if t == QEvent.KeyPress and ev.key() == Qt.Key_R:
                self._rotate_key_down = True
            elif t == QEvent.KeyRelease and ev.key() == Qt.Key_R:
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

    def get_view_state(self) -> dict:
        """
        Snapshot the current viewing state in coordinates that are resolution-agnostic.
        Returns a dict with:
          - idx: current image index within `self.files`
          - rotation: current rotation in degrees
          - scale: user zoom factor (1.0 = as loaded by `fit()`)
          - nx, ny: normalized center (0..1) within the image
        """
        state = {
            "idx": int(getattr(self, "idx", 0)),
            "rotation": float(getattr(self, "_rotation", 0.0)),
            "scale": float(getattr(self, "_scale", 1.0)),
            "nx": 0.5,
            "ny": 0.5,
        }
        try:
            rect = self.pix_item.boundingRect()
            center_scene = self.view.mapToScene(self.view.viewport().rect().center())
            center_item = self.pix_item.mapFromScene(center_scene)
            nx = 0.5 if rect.width() <= 0 else (center_item.x() - rect.left()) / rect.width()
            ny = 0.5 if rect.height() <= 0 else (center_item.y() - rect.top()) / rect.height()
            state["nx"] = float(nx)
            state["ny"] = float(ny)
        except Exception:
            pass
        return state

    def set_view_state(self, state: dict) -> None:
        """
        Restore a viewing state saved by `get_view_state`. Missing keys are ignored.
        Safe to call immediately after `set_files()`.
        """
        if not self.files:
            return

        # 1) index
        idx = int(state.get("idx", getattr(self, "idx", 0)))
        self.idx = max(0, min(idx, len(self.files) - 1))

        # Ensure pixmap exists
        self._show_current(reset_view=True)

        # 2) rotation + normalized center
        nx = float(state.get("nx", 0.5))
        ny = float(state.get("ny", 0.5))
        self.pix_item.setTransformOriginPoint(self.pix_item.boundingRect().center())
        self._rotation = float(state.get("rotation", getattr(self, "_rotation", 0.0)))
        self.pix_item.setRotation(self._rotation)

        # 3) zoom (user scale on top of fit)
        target_scale = float(state.get("scale", getattr(self, "_scale", 1.0)))
        try:
            if target_scale > 0 and abs(target_scale - self._scale) > 1e-6:
                s = target_scale / float(self._scale or 1.0)
                if 0 < s < 1e6:
                    self.view.scale(s, s)
                    self._scale *= s
        except Exception:
            pass

        # 4) center to the same point
        rect = self.pix_item.boundingRect()
        cx = rect.left() + nx * rect.width()
        cy = rect.top() + ny * rect.height()
        self.view.centerOn(self.pix_item.mapToScene(QPointF(cx, cy)))

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
