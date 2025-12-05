# src/ui/color_picker.py
"""
Color picker dialog with color wheel, eyedropper, and automatic name suggestions.

Features:
- Color wheel (HSV-based hue/saturation selection)
- Brightness slider
- RGB and Hex input fields
- Dynamic color name suggestions based on closest match
- Simple eyedropper for picking from images (PowerPoint-style)
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, List

from PySide6.QtCore import Qt, Signal, QPoint, QTimer, QRect, QSize, QEvent, QObject
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QSlider, QWidget,
    QFrame, QDialogButtonBox, QApplication, QSizePolicy, QToolTip,
)
from PySide6.QtGui import (
    QColor, QPainter, QPixmap, QImage, QPen, QCursor,
    QMouseEvent, QPaintEvent, QFont,
)


# =============================================================================
# COLOR WHEEL WIDGET
# =============================================================================

class ColorWheelWidget(QWidget):
    """
    HSV color wheel for selecting hue and saturation.
    """
    colorChanged = Signal(QColor)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self._hue = 0
        self._saturation = 255
        self._value = 255
        
        self._wheel_pixmap: Optional[QPixmap] = None
        self._last_size = QSize()
        self._last_value = -1
        self._dragging = False
    
    def setHSV(self, h: int, s: int, v: int) -> None:
        self._hue = max(0, min(359, h))
        self._saturation = max(0, min(255, s))
        old_value = self._value
        self._value = max(0, min(255, v))
        if self._value != old_value:
            self._wheel_pixmap = None  # Force rebuild
        self.update()
    
    def setValue(self, v: int) -> None:
        old_value = self._value
        self._value = max(0, min(255, v))
        if self._value != old_value:
            self._wheel_pixmap = None
        self.update()
    
    def currentColor(self) -> QColor:
        return QColor.fromHsv(self._hue, self._saturation, self._value)
    
    def _rebuild_wheel(self) -> None:
        size = min(self.width(), self.height())
        if size < 10:
            return
        
        # Cache check
        if (self._wheel_pixmap and 
            self._last_size == QSize(size, size) and 
            self._last_value == self._value):
            return
        
        image = QImage(size, size, QImage.Format_ARGB32)
        image.fill(Qt.transparent)
        
        center = size / 2
        radius = center - 2
        
        for y in range(size):
            for x in range(size):
                dx = x - center
                dy = y - center
                dist = math.sqrt(dx * dx + dy * dy)
                
                if dist <= radius:
                    angle = math.degrees(math.atan2(dy, dx))
                    hue = int((angle + 180) % 360)
                    sat = int(255 * dist / radius)
                    color = QColor.fromHsv(hue, sat, self._value)
                    image.setPixelColor(x, y, color)
        
        self._wheel_pixmap = QPixmap.fromImage(image)
        self._last_size = QSize(size, size)
        self._last_value = self._value
    
    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        self._rebuild_wheel()
        
        if self._wheel_pixmap:
            x = (self.width() - self._wheel_pixmap.width()) // 2
            y = (self.height() - self._wheel_pixmap.height()) // 2
            painter.drawPixmap(x, y, self._wheel_pixmap)
        
        # Draw selector
        size = min(self.width(), self.height())
        center = size / 2
        radius = center - 2
        
        angle = math.radians(self._hue - 180)
        dist = radius * self._saturation / 255
        
        sel_x = center + dist * math.cos(angle)
        sel_y = center + dist * math.sin(angle)
        
        off_x = (self.width() - size) // 2
        off_y = (self.height() - size) // 2
        
        # White outline
        painter.setPen(QPen(Qt.white, 3))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPoint(int(sel_x + off_x), int(sel_y + off_y)), 8, 8)
        # Black outline
        painter.setPen(QPen(Qt.black, 1))
        painter.drawEllipse(QPoint(int(sel_x + off_x), int(sel_y + off_y)), 9, 9)
    
    def _update_from_pos(self, pos: QPoint) -> None:
        size = min(self.width(), self.height())
        center = size / 2
        radius = center - 2
        
        off_x = (self.width() - size) // 2
        off_y = (self.height() - size) // 2
        
        dx = pos.x() - off_x - center
        dy = pos.y() - off_y - center
        dist = min(math.sqrt(dx * dx + dy * dy), radius)
        
        angle = math.degrees(math.atan2(dy, dx))
        self._hue = int((angle + 180) % 360)
        self._saturation = int(255 * dist / radius)
        
        self.update()
        self.colorChanged.emit(self.currentColor())
    
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._update_from_pos(event.pos())
    
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging:
            self._update_from_pos(event.pos())
    
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._dragging = False


# =============================================================================
# COLOR PREVIEW SWATCH
# =============================================================================

class ColorSwatchWidget(QWidget):
    """Displays a solid color with border."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._color = QColor(128, 128, 128)
        self.setMinimumSize(60, 60)
    
    def setColor(self, color: QColor) -> None:
        self._color = color
        self.update()
    
    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), self._color)
        painter.setPen(QPen(Qt.black, 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))


# =============================================================================
# MAIN COLOR PICKER DIALOG
# =============================================================================

class ColorPickerDialog(QDialog):
    """
    Dialog for picking/defining a new color.
    
    Features:
    - Color wheel with brightness slider
    - RGB and Hex input
    - Automatic name suggestion based on closest match
    - Eyedropper for picking from images
    
    Can be shown modally (exec()) or non-modally (show()) for eyedropper use.
    When non-modal, use the colorAccepted signal to get results.
    """
    
    # Signal emitted when color is accepted (for non-modal use)
    colorAccepted = Signal(str, tuple)  # (name, (r, g, b))
    
    def __init__(self, parent=None, initial_name: str = "", initial_color: QColor = None):
        super().__init__(parent)
        self.setWindowTitle("Add New Color")
        self.setMinimumSize(500, 420)
        
        self._suppress_updates = False
        self._user_edited_name = False  # Track if user manually typed a name
        self._color_config = None
        
        self._build_ui()
        
        # Set initial values
        if initial_name:
            self.name_edit.setText(initial_name)
            self._user_edited_name = True
        
        if initial_color and initial_color.isValid():
            self._set_color(initial_color)
        else:
            self._set_color(QColor(255, 165, 0))  # Default orange
    
    def _get_color_config(self):
        """Lazy load color config."""
        if self._color_config is None:
            try:
                from src.data.color_config import get_color_config
                self._color_config = get_color_config()
            except ImportError:
                pass
        return self._color_config
    
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        # === Name input with suggestion ===
        name_frame = QFrame()
        name_frame.setFrameStyle(QFrame.StyledPanel)
        name_layout = QVBoxLayout(name_frame)
        name_layout.setContentsMargins(10, 10, 10, 10)
        
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Color Name:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Auto-suggested or type custom name")
        self.name_edit.textEdited.connect(self._on_name_edited)
        name_row.addWidget(self.name_edit)
        name_layout.addLayout(name_row)
        
        # Suggestion label
        self.suggestion_label = QLabel("")
        self.suggestion_label.setStyleSheet("color: #666; font-style: italic;")
        name_layout.addWidget(self.suggestion_label)
        
        layout.addWidget(name_frame)
        
        # === Main content ===
        main_layout = QHBoxLayout()
        main_layout.setSpacing(20)
        
        # Left: Color wheel
        wheel_frame = QFrame()
        wheel_frame.setFrameStyle(QFrame.StyledPanel)
        wheel_layout = QVBoxLayout(wheel_frame)
        
        self.color_wheel = ColorWheelWidget()
        self.color_wheel.setMinimumSize(220, 220)
        self.color_wheel.colorChanged.connect(self._on_wheel_changed)
        wheel_layout.addWidget(self.color_wheel)
        
        # Brightness slider
        bright_layout = QHBoxLayout()
        bright_layout.addWidget(QLabel("Brightness:"))
        self.value_slider = QSlider(Qt.Horizontal)
        self.value_slider.setRange(0, 255)
        self.value_slider.setValue(255)
        self.value_slider.valueChanged.connect(self._on_value_changed)
        bright_layout.addWidget(self.value_slider)
        wheel_layout.addLayout(bright_layout)
        
        main_layout.addWidget(wheel_frame, 2)
        
        # Right: Preview + controls
        right_frame = QFrame()
        right_frame.setFrameStyle(QFrame.StyledPanel)
        right_layout = QVBoxLayout(right_frame)
        
        # Large preview swatch
        right_layout.addWidget(QLabel("Preview:"))
        self.preview_swatch = ColorSwatchWidget()
        self.preview_swatch.setMinimumSize(120, 100)
        right_layout.addWidget(self.preview_swatch)
        
        # RGB inputs
        rgb_layout = QFormLayout()
        rgb_layout.setSpacing(6)
        
        self.spin_r = QSpinBox()
        self.spin_r.setRange(0, 255)
        self.spin_r.valueChanged.connect(self._on_rgb_changed)
        rgb_layout.addRow("R:", self.spin_r)
        
        self.spin_g = QSpinBox()
        self.spin_g.setRange(0, 255)
        self.spin_g.valueChanged.connect(self._on_rgb_changed)
        rgb_layout.addRow("G:", self.spin_g)
        
        self.spin_b = QSpinBox()
        self.spin_b.setRange(0, 255)
        self.spin_b.valueChanged.connect(self._on_rgb_changed)
        rgb_layout.addRow("B:", self.spin_b)
        
        self.hex_edit = QLineEdit()
        self.hex_edit.setMaxLength(7)
        self.hex_edit.setPlaceholderText("#RRGGBB")
        self.hex_edit.textEdited.connect(self._on_hex_edited)
        rgb_layout.addRow("Hex:", self.hex_edit)
        
        right_layout.addLayout(rgb_layout)
        
        # Eyedropper buttons
        eyedropper_layout = QHBoxLayout()
        
        self.btn_eyedropper = QPushButton("🎯 Pick from Image")
        self.btn_eyedropper.setToolTip(
            "Enter eyedropper mode to pick a color from the image.\n"
            "You can still drag, zoom, and change images while picking.\n"
            "RIGHT-CLICK to confirm the color, ESC to cancel."
        )
        self.btn_eyedropper.clicked.connect(self._on_eyedropper_clicked)
        eyedropper_layout.addWidget(self.btn_eyedropper)
        
        self.btn_eyedropper_cancel = QPushButton("✕")
        self.btn_eyedropper_cancel.setFixedWidth(30)
        self.btn_eyedropper_cancel.setToolTip("Cancel eyedropper (ESC)")
        self.btn_eyedropper_cancel.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        self.btn_eyedropper_cancel.clicked.connect(self._on_eyedropper_cancel_clicked)
        self.btn_eyedropper_cancel.hide()  # Hidden until eyedropper mode starts
        eyedropper_layout.addWidget(self.btn_eyedropper_cancel)
        
        right_layout.addLayout(eyedropper_layout)
        
        right_layout.addStretch()
        main_layout.addWidget(right_frame, 1)
        
        layout.addLayout(main_layout)
        
        # === Dialog buttons ===
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def _set_color(self, color: QColor) -> None:
        """Update all controls to show the given color."""
        self._suppress_updates = True
        
        self.preview_swatch.setColor(color)
        self.spin_r.setValue(color.red())
        self.spin_g.setValue(color.green())
        self.spin_b.setValue(color.blue())
        self.hex_edit.setText(color.name().upper())
        
        h, s, v, _ = color.getHsv()
        self.color_wheel.setHSV(h, s, v)
        self.value_slider.setValue(v)
        
        self._suppress_updates = False
        
        # Update name suggestion
        self._update_name_suggestion(color)
    
    def _update_name_suggestion(self, color: QColor) -> None:
        """Update the suggested color name based on closest match.
        
        Always auto-fills the name field with the closest match unless user has edited it.
        This allows users to quickly select existing colors by just clicking OK.
        """
        config = self._get_color_config()
        if not config:
            return
        
        rgb = (color.red(), color.green(), color.blue())
        closest = config.find_closest(rgb, top_k=3)
        
        if closest:
            best_name, best_dist = closest[0]
            
            # Update suggestion label with match quality indicator
            if best_dist < 5:
                self.suggestion_label.setText(f"✓ Exact match: {best_name}")
                self.suggestion_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            elif best_dist < 15:
                self.suggestion_label.setText(f"≈ Very close: {best_name} (ΔE={best_dist:.1f})")
                self.suggestion_label.setStyleSheet("color: #2196F3;")
            elif best_dist < 30:
                self.suggestion_label.setText(f"~ Similar to: {best_name} (ΔE={best_dist:.1f})")
                self.suggestion_label.setStyleSheet("color: #FF9800;")
            else:
                others = ", ".join(n for n, d in closest[1:3])
                self.suggestion_label.setText(f"Nearest: {best_name} (ΔE={best_dist:.1f}) — also: {others}")
                self.suggestion_label.setStyleSheet("color: #666; font-style: italic;")
            
            # Always auto-fill name with closest match (unless user manually edited)
            # This makes it easy to select existing colors - just click OK
            if not self._user_edited_name:
                self.name_edit.setText(best_name)
    
    def _current_color(self) -> QColor:
        return QColor(self.spin_r.value(), self.spin_g.value(), self.spin_b.value())
    
    def _on_wheel_changed(self, color: QColor) -> None:
        if self._suppress_updates:
            return
        self._set_color(color)
    
    def _on_value_changed(self, value: int) -> None:
        if self._suppress_updates:
            return
        self.color_wheel.setValue(value)
        self._set_color(self.color_wheel.currentColor())
    
    def _on_rgb_changed(self) -> None:
        if self._suppress_updates:
            return
        self._set_color(self._current_color())
    
    def _on_hex_edited(self, text: str) -> None:
        if self._suppress_updates:
            return
        if not text.startswith("#"):
            text = "#" + text
        color = QColor(text)
        if color.isValid():
            self._set_color(color)
    
    def _on_name_edited(self, text: str) -> None:
        """User manually edited the name."""
        self._user_edited_name = bool(text.strip())
    
    def _on_eyedropper_clicked(self) -> None:
        """Toggle eyedropper mode, or confirm color if already in eyedropper mode."""
        if getattr(self, '_eyedropper_timer', None) is not None:
            # Already in eyedropper mode - clicking the button confirms the color
            self._stop_eyedropper_mode(accept_color=True)
            return
        
        self._start_eyedropper_mode()
    
    def _on_eyedropper_cancel_clicked(self) -> None:
        """Cancel eyedropper mode without accepting the color."""
        self._stop_eyedropper_mode(accept_color=False)
    
    def _start_eyedropper_mode(self) -> None:
        """Enter eyedropper mode - poll screen color under cursor."""
        # Update button to show confirm action
        self.btn_eyedropper.setText("✓ Use This Color (Right-click)")
        self.btn_eyedropper.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        
        # Show cancel button
        self.btn_eyedropper_cancel.show()
        
        # Move dialog to the side so user can see the image
        screen = QApplication.primaryScreen()
        if screen:
            screen_rect = screen.availableGeometry()
            # Position at right side but not too close to edge
            new_x = screen_rect.right() - self.width() - 20
            new_y = max(screen_rect.top() + 50, self.y())  # Don't move up if already lower
            self.move(new_x, new_y)
        
        # Start polling timer
        self._eyedropper_timer = QTimer(self)
        self._eyedropper_timer.timeout.connect(self._poll_screen_color)
        self._eyedropper_timer.start(50)  # 20fps
        
        # Change cursor application-wide
        QApplication.setOverrideCursor(Qt.CrossCursor)
        
        # Install application-wide event filter to catch right-clicks
        QApplication.instance().installEventFilter(self)
        
        # Raise dialog to front
        self.raise_()
        self.activateWindow()
        
        print("[Eyedropper] Mode started - drag/zoom image, right-click to confirm")
    
    def _stop_eyedropper_mode(self, accept_color: bool = False) -> None:
        """Exit eyedropper mode (safe to call even if not in eyedropper mode)."""
        timer = getattr(self, '_eyedropper_timer', None)
        if timer:
            timer.stop()
            self._eyedropper_timer = None
            
            print(f"[Eyedropper] Mode stopped, accept={accept_color}")
            
            # Remove application event filter
            try:
                QApplication.instance().removeEventFilter(self)
            except:
                pass
            
            try:
                QApplication.restoreOverrideCursor()
            except:
                pass
            
            self.btn_eyedropper.setText("🎯 Pick from Image")
            self.btn_eyedropper.setStyleSheet("")
            self.btn_eyedropper_cancel.hide()
            
            # Reset suggestion label style
            self.suggestion_label.setStyleSheet("color: #666; font-style: italic;")
            
            if accept_color:
                # Update name suggestion for the confirmed color
                self._update_name_suggestion(self._current_color())
    
    def _poll_screen_color(self) -> None:
        """Sample color from screen at current cursor position."""
        try:
            cursor_pos = QCursor.pos()
            screen = QApplication.screenAt(cursor_pos)
            if screen:
                # Grab a 1x1 pixel at cursor position
                pixmap = screen.grabWindow(0, cursor_pos.x(), cursor_pos.y(), 1, 1)
                if not pixmap.isNull():
                    image = pixmap.toImage()
                    color = QColor(image.pixel(0, 0))
                    if color.isValid():
                        # Update preview WITHOUT triggering normal callbacks
                        self._suppress_updates = True
                        self.preview_swatch.setColor(color)
                        self.spin_r.setValue(color.red())
                        self.spin_g.setValue(color.green())
                        self.spin_b.setValue(color.blue())
                        self.hex_edit.setText(color.name().upper())
                        h, s, v, _ = color.getHsv()
                        self.color_wheel.setHSV(h, s, v)
                        self.value_slider.setValue(v)
                        self._suppress_updates = False
                        
                        # Update suggestion label dynamically
                        self._update_suggestion_display(color)
        except Exception as e:
            print(f"[Eyedropper] poll error: {e}")
    
    def _update_suggestion_display(self, color: QColor) -> None:
        """Update suggestion label and auto-fill name (used during eyedropper mode)."""
        config = self._get_color_config()
        if not config:
            return
        
        rgb = (color.red(), color.green(), color.blue())
        closest = config.find_closest(rgb, top_k=3)
        
        if closest:
            best_name, best_dist = closest[0]
            
            if best_dist < 5:
                self.suggestion_label.setText(f"✓ Exact match: {best_name}")
                self.suggestion_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            elif best_dist < 15:
                self.suggestion_label.setText(f"≈ Very close: {best_name} (ΔE={best_dist:.1f})")
                self.suggestion_label.setStyleSheet("color: #2196F3;")
            elif best_dist < 30:
                self.suggestion_label.setText(f"~ Similar: {best_name} (ΔE={best_dist:.1f})")
                self.suggestion_label.setStyleSheet("color: #FF9800;")
            else:
                self.suggestion_label.setText(f"Nearest: {best_name} (ΔE={best_dist:.1f})")
                self.suggestion_label.setStyleSheet("color: #666;")
            
            # Also auto-fill the name field during eyedropper mode
            # (unless user has manually edited it)
            if not self._user_edited_name:
                self.name_edit.setText(best_name)
    
    def keyPressEvent(self, event) -> None:
        """Handle key press - check for eyedropper mode."""
        if getattr(self, '_eyedropper_timer', None):
            key = event.key()
            if key == Qt.Key_Escape:
                print("[Eyedropper] ESC - cancelling")
                self._stop_eyedropper_mode(accept_color=False)
                return
            elif key in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
                print("[Eyedropper] Space/Enter - confirming color")
                self._stop_eyedropper_mode(accept_color=True)
                return
        super().keyPressEvent(event)
    
    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """Application-wide event filter to catch right-clicks during eyedropper mode."""
        if getattr(self, '_eyedropper_timer', None):
            if event.type() == QEvent.MouseButtonPress:
                # Check if it's a right-click
                if hasattr(event, 'button') and event.button() == Qt.RightButton:
                    print("[Eyedropper] Right-click - confirming color")
                    self._stop_eyedropper_mode(accept_color=True)
                    return True  # Consume the event
            elif event.type() == QEvent.KeyPress:
                # Also catch ESC key globally
                if hasattr(event, 'key') and event.key() == Qt.Key_Escape:
                    print("[Eyedropper] ESC (global) - cancelling")
                    self._stop_eyedropper_mode(accept_color=False)
                    return True
        return False  # Don't consume other events
    
    def _on_accept(self) -> None:
        """Validate and accept."""
        # Stop eyedropper if active (accepting the current color)
        self._stop_eyedropper_mode(accept_color=True)
        
        name = self.name_edit.text().strip()
        if not name:
            self.name_edit.setFocus()
            self.name_edit.setStyleSheet("border: 2px solid red;")
            self.suggestion_label.setText("⚠️ Please enter a color name")
            self.suggestion_label.setStyleSheet("color: red; font-weight: bold;")
            return
        self.name_edit.setStyleSheet("")
        
        # Emit signal for non-modal use
        color = self._current_color()
        self.colorAccepted.emit(name.lower(), (color.red(), color.green(), color.blue()))
        
        self.accept()
    
    def reject(self) -> None:
        """Handle dialog rejection (cancel/close)."""
        self._stop_eyedropper_mode(accept_color=False)
        super().reject()
    
    def closeEvent(self, event) -> None:
        """Clean up on close."""
        self._stop_eyedropper_mode(accept_color=False)
        super().closeEvent(event)
    
    def eyedropper_was_requested(self) -> bool:
        """Check if user clicked the eyedropper button (legacy - no longer used)."""
        return False  # Eyedropper now works within the dialog
    
    def get_result(self) -> Optional[Tuple[str, Tuple[int, int, int]]]:
        """Get result if dialog was accepted (not via eyedropper)."""
        if self.result() != QDialog.Accepted or self.eyedropper_was_requested():
            return None
        
        name = self.name_edit.text().strip().lower()
        color = self._current_color()
        return (name, (color.red(), color.green(), color.blue()))
    
    def get_current_color(self) -> QColor:
        """Get the currently selected color."""
        return self._current_color()


# =============================================================================
# IMAGE EYEDROPPER MIXIN (for ImageStrip integration)
# =============================================================================

class ImageEyedropperMixin:
    """
    Mixin to add eyedropper functionality to image viewing widgets.
    
    The widget using this mixin should:
    1. Call `eyedropper_start()` to enter eyedropper mode
    2. Override `eyedropper_get_image_at(pos)` to return the QImage at a point
    3. Connect mouse events to `eyedropper_mouse_move/press`
    """
    eyedropperColorPicked = Signal(QColor)
    eyedropperCancelled = Signal()
    
    def eyedropper_init(self):
        """Initialize eyedropper state."""
        self._eyedropper_active = False
        self._eyedropper_color = QColor()
        self._old_cursor = None
    
    def eyedropper_start(self):
        """Enter eyedropper mode."""
        self._eyedropper_active = True
        self._old_cursor = self.cursor()
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)
    
    def eyedropper_stop(self):
        """Exit eyedropper mode."""
        self._eyedropper_active = False
        if self._old_cursor:
            self.setCursor(self._old_cursor)
        QToolTip.hideText()
    
    def eyedropper_is_active(self) -> bool:
        return getattr(self, '_eyedropper_active', False)
    
    def eyedropper_mouse_move(self, event: QMouseEvent, image: QImage) -> None:
        """Call this from mouseMoveEvent when eyedropper is active."""
        if not self._eyedropper_active or image is None:
            return
        
        pos = event.pos()
        if 0 <= pos.x() < image.width() and 0 <= pos.y() < image.height():
            self._eyedropper_color = QColor(image.pixel(pos.x(), pos.y()))
            
            # Show tooltip with color preview
            hex_code = self._eyedropper_color.name().upper()
            r, g, b = self._eyedropper_color.red(), self._eyedropper_color.green(), self._eyedropper_color.blue()
            
            # Create rich tooltip with color swatch
            tooltip = f"""
            <div style="padding: 5px;">
                <div style="background-color: {hex_code}; 
                            width: 60px; height: 40px; 
                            border: 2px solid black; 
                            margin-bottom: 5px;">
                </div>
                <div style="font-family: monospace; font-size: 12px;">
                    {hex_code}<br>
                    RGB: {r}, {g}, {b}
                </div>
            </div>
            """
            QToolTip.showText(event.globalPos(), tooltip, self)
    
    def eyedropper_mouse_press(self, event: QMouseEvent) -> bool:
        """
        Call this from mousePressEvent. Returns True if event was handled.
        """
        if not self._eyedropper_active:
            return False
        
        if event.button() == Qt.LeftButton:
            self.eyedropper_stop()
            if self._eyedropper_color.isValid():
                self.eyedropperColorPicked.emit(self._eyedropper_color)
            return True
        elif event.button() == Qt.RightButton:
            self.eyedropper_stop()
            self.eyedropperCancelled.emit()
            return True
        
        return False
    
    def eyedropper_key_press(self, event) -> bool:
        """Call this from keyPressEvent. Returns True if event was handled."""
        if not self._eyedropper_active:
            return False
        
        if event.key() == Qt.Key_Escape:
            self.eyedropper_stop()
            self.eyedropperCancelled.emit()
            return True
        
        return False


def pick_color_simple(parent=None) -> Optional[Tuple[str, Tuple[int, int, int]]]:
    """Simple function to show color picker and get result."""
    dialog = ColorPickerDialog(parent)
    dialog.exec()
    return dialog.get_result()
