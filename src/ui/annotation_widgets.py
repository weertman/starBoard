# src/ui/annotation_widgets.py
"""
Custom widgets for the annotation system.

Provides specialized input widgets for each annotation type:
- NumericSpinBox: Integer/float input with validation
- ShortArmCodeEditor: Composite widget for short arm coding
- ColorCategoricalComboBox: Extensible color vocabulary selector
- MorphCategoricalComboBox: Fixed-option categorical selector
- TextHistoryComboBox: Text input with history
"""
from __future__ import annotations

from typing import List, Optional, Union, Callable, Tuple
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit,
    QPushButton, QLabel, QScrollArea, QFrame, QInputDialog,
    QSizePolicy, QLineEdit, QApplication, QStyledItemDelegate, QStyle,
    QToolTip, QCompleter,
)
from PySide6.QtGui import QIntValidator, QDoubleValidator, QColor, QPixmap, QPainter, QIcon, QCursor

from src.data.annotation_schema import (
    FieldDefinition, AnnotationType, CategoricalOption,
    ShortArmEntry, parse_short_arm_code, serialize_short_arm_code,
    SHORT_ARM_SEVERITY_OPTIONS,
)
from src.data.vocabulary_store import get_vocabulary_store
from src.utils.interaction_logger import get_interaction_logger


# =============================================================================
# BASE CLASSES
# =============================================================================

class AnnotationWidget(QWidget):
    """Base class for annotation input widgets."""
    value_changed = Signal()

    def __init__(self, field_def: FieldDefinition, parent=None):
        super().__init__(parent)
        self.field_def = field_def
        self._ilog = get_interaction_logger()
    
    def _log_value_change(self, value: str) -> None:
        """Log a value change for this annotation widget."""
        self._ilog.log("annotation_change", f"annotation_{self.field_def.name}", value=value)

    def get_value(self) -> str:
        """Get the current value as a string for CSV storage."""
        raise NotImplementedError

    def set_value(self, value: str) -> None:
        """Set the value from a string (from CSV)."""
        raise NotImplementedError

    def clear(self) -> None:
        """Clear/reset the widget to empty state."""
        raise NotImplementedError


# =============================================================================
# NUMERIC WIDGETS
# =============================================================================

class NumericIntWidget(AnnotationWidget):
    """Integer input with optional range validation."""

    def __init__(self, field_def: FieldDefinition, parent=None):
        super().__init__(field_def, parent)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.spinbox = QSpinBox()
        self.spinbox.setSpecialValueText("")  # Show empty when at minimum
        self.spinbox.setMinimum(-1)  # -1 represents "not set"
        self.spinbox.setValue(-1)
        
        if field_def.min_value is not None:
            self.spinbox.setMinimum(max(-1, int(field_def.min_value) - 1))
        if field_def.max_value is not None:
            self.spinbox.setMaximum(int(field_def.max_value))
        
        self.spinbox.valueChanged.connect(self._on_change)
        layout.addWidget(self.spinbox)
        layout.addStretch()

    def _on_change(self) -> None:
        self._log_value_change(self.get_value())
        self.value_changed.emit()

    def get_value(self) -> str:
        val = self.spinbox.value()
        if val < 0:  # "not set" state
            return ""
        return str(val)

    def set_value(self, value: str) -> None:
        if not value or not value.strip():
            self.spinbox.setValue(-1)
        else:
            try:
                self.spinbox.setValue(int(float(value)))
            except ValueError:
                self.spinbox.setValue(-1)

    def clear(self) -> None:
        self.spinbox.setValue(-1)


class NumericFloatWidget(AnnotationWidget):
    """Float input with optional range validation."""

    def __init__(self, field_def: FieldDefinition, parent=None):
        super().__init__(field_def, parent)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.spinbox = QDoubleSpinBox()
        self.spinbox.setDecimals(2)
        self.spinbox.setSpecialValueText("")
        self.spinbox.setMinimum(-0.01)  # -0.01 represents "not set"
        self.spinbox.setValue(-0.01)
        
        if field_def.min_value is not None:
            self.spinbox.setMinimum(min(-0.01, field_def.min_value - 0.01))
        if field_def.max_value is not None:
            self.spinbox.setMaximum(field_def.max_value)
        
        self.spinbox.valueChanged.connect(self._on_change)
        layout.addWidget(self.spinbox)
        layout.addStretch()

    def _on_change(self) -> None:
        self._log_value_change(self.get_value())
        self.value_changed.emit()

    def get_value(self) -> str:
        val = self.spinbox.value()
        if val < 0:
            return ""
        return f"{val:.2f}"

    def set_value(self, value: str) -> None:
        if not value or not value.strip():
            self.spinbox.setValue(-0.01)
        else:
            try:
                self.spinbox.setValue(float(value))
            except ValueError:
                self.spinbox.setValue(-0.01)

    def clear(self) -> None:
        self.spinbox.setValue(-0.01)


# =============================================================================
# SHORT ARM CODE EDITOR
# =============================================================================

class ShortArmEntryRow(QWidget):
    """A single row in the short arm code editor."""
    changed = Signal()
    remove_requested = Signal(object)  # emits self

    SEVERITY_OPTIONS = ["tiny", "small", "short"]

    def __init__(self, entry: Optional[ShortArmEntry] = None, parent=None):
        super().__init__(parent)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(4)
        
        # Arm position
        self.position_spin = QSpinBox()
        self.position_spin.setMinimum(1)
        self.position_spin.setMaximum(25)
        self.position_spin.setPrefix("Arm ")
        self.position_spin.setFixedWidth(80)
        self.position_spin.valueChanged.connect(self._emit_changed)
        layout.addWidget(self.position_spin)
        
        # Severity selector
        self.severity_combo = QComboBox()
        self.severity_combo.addItems(self.SEVERITY_OPTIONS)
        self.severity_combo.setFixedWidth(100)
        self.severity_combo.currentTextChanged.connect(self._emit_changed)
        layout.addWidget(self.severity_combo)
        
        # Remove button
        self.remove_btn = QPushButton("✕")
        self.remove_btn.setFixedSize(24, 24)
        self.remove_btn.setToolTip("Remove this entry")
        self.remove_btn.clicked.connect(lambda: self.remove_requested.emit(self))
        layout.addWidget(self.remove_btn)
        
        layout.addStretch()
        
        # Initialize from entry if provided
        if entry:
            self.position_spin.setValue(entry.position)
            idx = self.severity_combo.findText(entry.severity)
            if idx >= 0:
                self.severity_combo.setCurrentIndex(idx)

    def _emit_changed(self) -> None:
        self.changed.emit()

    def get_entry(self) -> ShortArmEntry:
        return ShortArmEntry(
            position=self.position_spin.value(),
            severity=self.severity_combo.currentText()
        )


class ShortArmCodeEditor(AnnotationWidget):
    """
    Composite widget for editing short arm codes.
    
    Displays a variable-length list of (arm_position, severity) pairs
    with add/remove functionality. Dynamically resizes based on content.
    """
    
    ROW_HEIGHT = 30  # Approximate height of each entry row
    MAX_VISIBLE_ROWS = 8  # Show scroll bar after this many rows

    def __init__(self, field_def: FieldDefinition, parent=None):
        super().__init__(field_def, parent)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)
        
        # Container for entries (no scroll area needed for small counts)
        self.entries_container = QWidget()
        self.entries_layout = QVBoxLayout(self.entries_container)
        self.entries_layout.setContentsMargins(0, 0, 0, 0)
        self.entries_layout.setSpacing(2)
        
        # Scrollable area - only shows scrollbar when needed
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setWidget(self.entries_container)
        self.scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        
        main_layout.addWidget(self.scroll)
        
        # Add button
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        self.add_btn = QPushButton("+ Add short arm")
        self.add_btn.clicked.connect(self._add_entry)
        btn_layout.addWidget(self.add_btn)
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)
        
        self._rows: List[ShortArmEntryRow] = []
        
        # Set initial size (just button height when empty)
        self._update_size()

    def _add_entry(self, entry: Optional[ShortArmEntry] = None) -> ShortArmEntryRow:
        row = ShortArmEntryRow(entry, self.entries_container)
        row.changed.connect(self._on_change)
        row.remove_requested.connect(self._remove_entry)
        
        # Add to layout
        self.entries_layout.addWidget(row)
        self._rows.append(row)
        self._update_size()
        self._on_change()
        return row

    def _remove_entry(self, row: ShortArmEntryRow) -> None:
        if row in self._rows:
            self._rows.remove(row)
            row.setParent(None)
            row.deleteLater()
            self._update_size()
            self._on_change()

    def _on_change(self) -> None:
        self.value_changed.emit()
    
    def _update_size(self) -> None:
        """Update the scroll area height based on number of entries."""
        num_rows = len(self._rows)
        
        if num_rows == 0:
            # No entries - minimize height
            self.scroll.setMinimumHeight(0)
            self.scroll.setMaximumHeight(0)
        else:
            # Calculate height needed for all rows
            content_height = num_rows * self.ROW_HEIGHT + 4  # +4 for margins
            max_height = self.MAX_VISIBLE_ROWS * self.ROW_HEIGHT
            
            # Set height to content or max, whichever is smaller
            display_height = min(content_height, max_height)
            self.scroll.setMinimumHeight(display_height)
            self.scroll.setMaximumHeight(display_height)
        
        # Force layout update
        self.updateGeometry()

    def get_value(self) -> str:
        entries = [row.get_entry() for row in self._rows]
        return serialize_short_arm_code(entries)

    def set_value(self, value: str) -> None:
        # Clear existing
        for row in self._rows[:]:
            self._remove_entry(row)
        
        # Parse and add new entries
        entries = parse_short_arm_code(value)
        for entry in entries:
            self._add_entry(entry)

    def clear(self) -> None:
        for row in self._rows[:]:
            self._remove_entry(row)


# =============================================================================
# COLOR CATEGORICAL COMBOBOX (with color swatches and picker)
# =============================================================================

class ColorSwatchDelegate(QStyledItemDelegate):
    """
    Item delegate that draws a color swatch next to text in combo box.
    """
    SWATCH_SIZE = 16
    SWATCH_MARGIN = 4
    
    def __init__(self, get_color_rgb_func, parent=None):
        super().__init__(parent)
        self._get_rgb = get_color_rgb_func  # Callable[[str], Optional[Tuple[int,int,int]]]
    
    def paint(self, painter, option, index):
        # Get the color name
        text = index.data(Qt.DisplayRole) or ""
        
        # Draw background (selection, hover, etc.)
        self.initStyleOption(option, index)
        style = option.widget.style() if option.widget else QApplication.style()
        
        # Draw the background
        style.drawPrimitive(QStyle.PE_PanelItemViewItem, option, painter, option.widget)
        
        # Calculate positions
        rect = option.rect
        swatch_rect = rect.adjusted(
            self.SWATCH_MARGIN, 
            (rect.height() - self.SWATCH_SIZE) // 2,
            0, 0
        )
        swatch_rect.setWidth(self.SWATCH_SIZE)
        swatch_rect.setHeight(self.SWATCH_SIZE)
        
        # Draw swatch if we have RGB for this color
        rgb = self._get_rgb(text) if text else None
        if rgb:
            color = QColor(rgb[0], rgb[1], rgb[2])
            painter.fillRect(swatch_rect, color)
            painter.setPen(QColor(0, 0, 0))
            painter.drawRect(swatch_rect.adjusted(0, 0, -1, -1))
            text_offset = self.SWATCH_SIZE + self.SWATCH_MARGIN * 2
        else:
            text_offset = self.SWATCH_MARGIN
        
        # Draw text
        text_rect = rect.adjusted(text_offset, 0, 0, 0)
        painter.setPen(option.palette.text().color())
        painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, text)
    
    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        # Add space for swatch
        size.setWidth(size.width() + self.SWATCH_SIZE + self.SWATCH_MARGIN * 2)
        size.setHeight(max(size.height(), self.SWATCH_SIZE + self.SWATCH_MARGIN))
        return size


class ColorCategoricalComboBox(AnnotationWidget):
    """
    ComboBox with extensible color vocabulary and visual color swatches.
    
    Features:
    - Color swatch displayed next to each color name
    - "New..." option opens a full color picker dialog with:
      - Color wheel (HSV selection)
      - RGB/Hex input
      - Eyedropper tool for sampling colors from screen
    - Colors are stored with RGB values in color_config.yaml
    """
    NEW_ITEM_TEXT = "✚ New..."

    def __init__(self, field_def: FieldDefinition, parent=None):
        super().__init__(field_def, parent)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        
        self.combo = QComboBox()
        self.combo.setMinimumWidth(180)
        
        # Make combo editable for type-to-search functionality
        self.combo.setEditable(True)
        self.combo.setInsertPolicy(QComboBox.NoInsert)  # Don't add typed text as items
        self.combo.completer().setFilterMode(Qt.MatchContains)  # Match anywhere in string
        self.combo.completer().setCompletionMode(QCompleter.PopupCompletion)
        
        # Set up custom delegate for color swatches
        self._delegate = ColorSwatchDelegate(self._get_color_rgb, self.combo)
        self.combo.setItemDelegate(self._delegate)
        
        # Use activated signal (user explicitly selected) instead of currentTextChanged
        # This prevents firing on every keystroke while typing to search
        self.combo.activated.connect(self._on_item_activated)
        # Also handle when user presses Enter on typed text
        self.combo.lineEdit().returnPressed.connect(self._on_return_pressed)
        
        layout.addWidget(self.combo)
        
        # Color picker button for quick access to color picker
        self.btn_picker = QPushButton("🎨")
        self.btn_picker.setFixedSize(28, 28)
        self.btn_picker.setToolTip("Open color picker / eyedropper")
        self.btn_picker.clicked.connect(self._on_picker_button_clicked)
        layout.addWidget(self.btn_picker)
        
        layout.addStretch()
        
        self._vocab = get_vocabulary_store()
        self._color_config = None  # Lazy load
        self._suppress_signal = False
        self._refresh_items()
    
    def _get_color_config(self):
        """Lazy load color config to avoid circular imports."""
        if self._color_config is None:
            try:
                from src.data.color_config import get_color_config
                self._color_config = get_color_config()
            except ImportError:
                pass
        return self._color_config
    
    def _get_color_rgb(self, color_name: str) -> Optional[tuple]:
        """Get RGB tuple for a color name, or None if unknown."""
        if not color_name or color_name == self.NEW_ITEM_TEXT:
            return None
        
        config = self._get_color_config()
        if config:
            color_def = config.get_color(color_name)
            if color_def:
                return color_def.rgb
        return None

    def _refresh_items(self) -> None:
        """Refresh the combo box items from vocabulary, sorted by color hue."""
        self._suppress_signal = True
        current = self.combo.currentText()
        
        self.combo.clear()
        self.combo.addItem("")  # Empty option
        
        colors = self._vocab.get_colors(self.field_def.name)
        sorted_colors = self._sort_colors_by_hue(colors)
        
        for color in sorted_colors:
            self.combo.addItem(color)
        
        self.combo.addItem(self.NEW_ITEM_TEXT)
        
        # Restore selection if possible
        if current and current != self.NEW_ITEM_TEXT:
            idx = self.combo.findText(current, Qt.MatchFixedString)
            if idx >= 0:
                self.combo.setCurrentIndex(idx)
        
        self._suppress_signal = False
    
    def _sort_colors_by_hue(self, color_names: List[str]) -> List[str]:
        """Sort color names by their hue/saturation/lightness values.
        
        Creates a perceptual ordering where similar colors appear near each other:
        - Achromatic colors (white, gray, black) first
        - Then chromatic colors sorted by hue (red → orange → yellow → green → blue → purple)
        - Within same hue, sorted by saturation then lightness
        """
        config = self._get_color_config()
        if not config:
            return sorted(color_names)  # Fallback to alphabetical
        
        def color_sort_key(name: str):
            color_def = config.get_color(name)
            if not color_def or not color_def.rgb:
                return (2, 0, 0, 0, name)  # Unknown colors at end, then alphabetical
            
            r, g, b = color_def.rgb
            qcolor = QColor(r, g, b)
            h, s, l, _ = qcolor.getHsl()
            
            # Achromatic colors (very low saturation) go first
            # Sort by lightness (white first, black last)
            if s < 20:
                return (0, -l, 0, 0, name)  # Group 0 = achromatic, -l so white comes first
            
            # Chromatic colors sorted by hue, then saturation (vivid first), then lightness
            return (1, h, -s, -l, name)  # Group 1 = chromatic
        
        return sorted(color_names, key=color_sort_key)

    def _on_item_activated(self, index: int) -> None:
        """Handle when user explicitly selects an item from dropdown."""
        if self._suppress_signal:
            return
        
        text = self.combo.itemText(index)
        if text == self.NEW_ITEM_TEXT:
            self._add_new_color()
        else:
            self.value_changed.emit()
    
    def _on_return_pressed(self) -> None:
        """Handle when user presses Enter in the editable combo."""
        if self._suppress_signal:
            return
        
        text = self.combo.currentText().strip()
        if not text:
            return
        
        # Check if text matches an existing item
        idx = self.combo.findText(text, Qt.MatchFixedString)
        if idx >= 0:
            self.combo.setCurrentIndex(idx)
            item_text = self.combo.itemText(idx)
            if item_text == self.NEW_ITEM_TEXT:
                self._add_new_color()
            else:
                self.value_changed.emit()

    def _add_new_color(self) -> None:
        """Open color picker dialog to add a new color."""
        self._show_color_picker()
    
    def _on_picker_button_clicked(self) -> None:
        """Open color picker directly from the eyedropper button."""
        self._show_color_picker()
    
    def _show_color_picker(self, initial_color: QColor = None) -> None:
        """Show the color picker dialog non-modally.
        
        The dialog is always non-modal so the user can interact with the image
        viewer while using the eyedropper. Results are handled via signals.
        
        Args:
            initial_color: Pre-selected color to start with
        """
        try:
            from src.ui.color_picker import ColorPickerDialog
            
            # Clean up any existing dialog
            if hasattr(self, '_color_picker_dialog') and self._color_picker_dialog:
                try:
                    self._color_picker_dialog.close()
                except:
                    pass
            
            dialog = ColorPickerDialog(self, initial_color=initial_color)
            self._color_picker_dialog = dialog
            
            # Connect signal for when color is accepted
            dialog.colorAccepted.connect(self._on_color_picker_accepted)
            dialog.rejected.connect(self._on_color_picker_rejected)
            
            # Show non-modally - this allows interaction with the image viewer
            # The dialog stays on top via WindowStaysOnTopHint
            dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowStaysOnTopHint)
            dialog.setModal(False)
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
                
        except ImportError as e:
            print(f"Color picker import error: {e}")
            # Fallback: simple text input
            self._add_new_color_simple()
    
    def _on_color_picker_accepted(self, name: str, rgb: tuple) -> None:
        """Handle color accepted from color picker.
        
        If the color name already exists, just select it.
        If it's a new name, add it to the config and vocabulary.
        """
        from src.data.color_config import get_color_config
        
        config = get_color_config()
        
        # Check if this color name already exists
        existing_color = config.get_color(name)
        
        if existing_color is None:
            # New color - add to config and vocabulary
            config.add_color(name, rgb, source="user")
            self._vocab.add_color(self.field_def.name, name)
            print(f"[ColorPicker] Added new color: {name} = {rgb}")
        else:
            # Existing color - just make sure it's in this field's vocabulary
            self._vocab.add_color(self.field_def.name, name)
            print(f"[ColorPicker] Using existing color: {name}")
        
        # Refresh and select
        self._refresh_items()
        idx = self.combo.findText(name)
        if idx >= 0:
            self.combo.setCurrentIndex(idx)
        
        # Clean up dialog reference
        self._color_picker_dialog = None
    
    def _on_color_picker_rejected(self) -> None:
        """Handle color picker cancelled/closed."""
        # Reset combo to empty if it was on "New..."
        if self.combo.currentText() == self.NEW_ITEM_TEXT:
            self.combo.setCurrentIndex(0)
        self._color_picker_dialog = None
    
    def _add_new_color_simple(self) -> None:
        """Fallback simple text input for adding colors."""
        color, ok = QInputDialog.getText(
            self, 
            "Add New Color",
            f"Enter new color for '{self.field_def.display_name}':",
            QLineEdit.Normal,
            ""
        )
        
        if ok and color.strip():
            color = color.strip().lower()
            if self._vocab.add_color(self.field_def.name, color):
                self._refresh_items()
            idx = self.combo.findText(color)
            if idx >= 0:
                self.combo.setCurrentIndex(idx)
        else:
            self.combo.setCurrentIndex(0)
    
    def _start_image_eyedropper(self, current_color: QColor = None) -> None:
        """
        Start eyedropper mode on the nearest image strip.
        After picking, reopen the color picker with the selected color.
        """
        from PySide6.QtWidgets import QMessageBox
        
        # Store current color to pass back to dialog
        self._eyedropper_base_color = current_color
        
        # Find the ImageStrip widget
        image_strip = self._find_image_strip()
        
        if image_strip is None:
            QMessageBox.information(
                self,
                "Eyedropper",
                "Could not find an image viewer.\n"
                "Please make sure an image is loaded and try again."
            )
            self._show_color_picker(current_color)
            return
        
        # Store reference to disconnect later
        self._eyedropper_strip = image_strip
        
        # Connect signals (using try/except for robustness)
        try:
            image_strip.eyedropperColorPicked.connect(self._on_eyedropper_color_picked)
            image_strip.eyedropperCancelled.connect(self._on_eyedropper_cancelled)
        except Exception as e:
            print(f"Failed to connect eyedropper signals: {e}")
            QMessageBox.warning(
                self,
                "Eyedropper Error",
                f"Could not set up eyedropper: {e}"
            )
            self._show_color_picker(current_color)
            return
        
        # Start eyedropper mode
        try:
            image_strip.start_eyedropper()
            
            # Show instruction to user
            QToolTip.showText(
                QCursor.pos(),
                "Hover over the image to see colors.\nClick to select, ESC to cancel.",
                image_strip,
                image_strip.rect(),
                3000  # Show for 3 seconds
            )
        except Exception as e:
            print(f"Failed to start eyedropper: {e}")
            self._show_color_picker(current_color)
    
    def _find_image_strip(self):
        """Find the best ImageStrip widget - one that's actually visible and has images."""
        from src.ui.image_strip import ImageStrip
        
        candidates = []
        
        # Collect all ImageStrips from all windows
        for window in QApplication.topLevelWidgets():
            strips = window.findChildren(ImageStrip)
            for strip in strips:
                # Check if this strip is actually visible on screen
                # isVisible() alone isn't enough - need to check if it's really showing
                if strip.files and len(strip.files) > 0:
                    # Check effective visibility by seeing if it has a non-zero size
                    # and its viewport is mapped to the screen
                    try:
                        viewport = strip.view.viewport()
                        global_pos = viewport.mapToGlobal(viewport.rect().center())
                        size = viewport.size()
                        # If viewport has reasonable size and is within screen bounds, it's likely visible
                        if size.width() > 50 and size.height() > 50:
                            # Score by size - larger is probably more prominent
                            score = size.width() * size.height()
                            candidates.append((score, strip))
                    except:
                        pass
        
        print(f"[ColorCombo] Found {len(candidates)} candidate ImageStrips with files")
        
        if candidates:
            # Sort by score (size) descending - pick the largest visible one
            candidates.sort(key=lambda x: x[0], reverse=True)
            best = candidates[0][1]
            print(f"[ColorCombo] Selected ImageStrip with {len(best.files)} files, viewport size {best.view.viewport().size().width()}x{best.view.viewport().size().height()}")
            return best
        
        print("[ColorCombo] No suitable ImageStrip found!")
        return None
    
    def _disconnect_eyedropper_signals(self) -> None:
        """Disconnect eyedropper signals from the image strip."""
        image_strip = getattr(self, '_eyedropper_strip', None)
        if image_strip:
            try:
                image_strip.eyedropperColorPicked.disconnect(self._on_eyedropper_color_picked)
            except (RuntimeError, TypeError):
                pass
            try:
                image_strip.eyedropperCancelled.disconnect(self._on_eyedropper_cancelled)
            except (RuntimeError, TypeError):
                pass
        self._eyedropper_strip = None
    
    def _on_eyedropper_color_picked(self, color: QColor) -> None:
        """Handle color picked from eyedropper."""
        self._disconnect_eyedropper_signals()
        # Reopen color picker with the picked color
        self._show_color_picker(color)
    
    def _on_eyedropper_cancelled(self) -> None:
        """Handle eyedropper cancelled."""
        self._disconnect_eyedropper_signals()
        # Reopen color picker with original color
        base_color = getattr(self, '_eyedropper_base_color', None)
        self._show_color_picker(base_color)

    def get_value(self) -> str:
        text = self.combo.currentText()
        if text == self.NEW_ITEM_TEXT:
            return ""
        return text

    def set_value(self, value: str) -> None:
        self._suppress_signal = True
        value = value.strip() if value else ""
        
        if not value:
            self.combo.setCurrentIndex(0)
        else:
            idx = self.combo.findText(value, Qt.MatchFixedString)
            if idx >= 0:
                self.combo.setCurrentIndex(idx)
            else:
                # Color not in vocabulary - add it
                if self._vocab.add_color(self.field_def.name, value):
                    self._refresh_items()
                idx = self.combo.findText(value, Qt.MatchFixedString)
                if idx >= 0:
                    self.combo.setCurrentIndex(idx)
        
        self._suppress_signal = False

    def clear(self) -> None:
        self._suppress_signal = True
        self.combo.setCurrentIndex(0)
        self._suppress_signal = False


# =============================================================================
# MORPHOLOGY CATEGORICAL COMBOBOX
# =============================================================================

class MorphCategoricalComboBox(AnnotationWidget):
    """
    ComboBox with fixed categorical options.
    
    Displays labels but stores ordinal values.
    """

    def __init__(self, field_def: FieldDefinition, parent=None):
        super().__init__(field_def, parent)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.combo = QComboBox()
        self.combo.setMinimumWidth(120)
        
        # Add empty option first
        self.combo.addItem("", None)
        
        # Add categorical options
        for opt in field_def.options:
            self.combo.addItem(opt.label, opt.value)
        
        self.combo.currentIndexChanged.connect(self._on_change)
        layout.addWidget(self.combo)
        layout.addStretch()

    def _on_change(self) -> None:
        self.value_changed.emit()

    def get_value(self) -> str:
        data = self.combo.currentData()
        if data is None:
            return ""
        return str(data)

    def set_value(self, value: str) -> None:
        if not value or not value.strip():
            self.combo.setCurrentIndex(0)
            return
            
        try:
            # Try to match by value
            val = float(value)
            for i in range(self.combo.count()):
                data = self.combo.itemData(i)
                if data is not None and float(data) == val:
                    self.combo.setCurrentIndex(i)
                    return
        except ValueError:
            pass
        
        # Try to match by label
        idx = self.combo.findText(value, Qt.MatchFixedString)
        if idx >= 0:
            self.combo.setCurrentIndex(idx)
        else:
            self.combo.setCurrentIndex(0)

    def clear(self) -> None:
        self.combo.setCurrentIndex(0)


# =============================================================================
# TEXT WIDGETS
# =============================================================================

class TextHistoryComboBox(AnnotationWidget):
    """
    Editable ComboBox with history of previous entries.
    
    Shows all previous values plus "New..." option.
    """
    NEW_ITEM_TEXT = "New..."

    def __init__(self, field_def: FieldDefinition, parent=None):
        super().__init__(field_def, parent)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.combo = QComboBox()
        self.combo.setEditable(False)
        self.combo.setMinimumWidth(200)
        self.combo.currentTextChanged.connect(self._on_selection_changed)
        layout.addWidget(self.combo)
        layout.addStretch()
        
        self._vocab = get_vocabulary_store()
        self._suppress_signal = False
        self._refresh_items()

    def _refresh_items(self) -> None:
        """Refresh items from vocabulary."""
        self._suppress_signal = True
        current = self.combo.currentText()
        
        self.combo.clear()
        self.combo.addItem("")
        
        for loc in self._vocab.get_locations():
            self.combo.addItem(loc)
        
        self.combo.addItem(self.NEW_ITEM_TEXT)
        
        if current and current != self.NEW_ITEM_TEXT:
            idx = self.combo.findText(current)
            if idx >= 0:
                self.combo.setCurrentIndex(idx)
        
        self._suppress_signal = False

    def _on_selection_changed(self, text: str) -> None:
        if self._suppress_signal:
            return
            
        if text == self.NEW_ITEM_TEXT:
            self._add_new_entry()
        else:
            self.value_changed.emit()

    def _add_new_entry(self) -> None:
        """Open dialog to add new location."""
        text, ok = QInputDialog.getText(
            self,
            "Add New Location",
            "Enter new location:",
            QLineEdit.Normal,
            ""
        )
        
        if ok and text.strip():
            text = text.strip()
            self._vocab.add_location(text)
            self._refresh_items()
            idx = self.combo.findText(text)
            if idx >= 0:
                self.combo.setCurrentIndex(idx)
        else:
            self.combo.setCurrentIndex(0)

    def get_value(self) -> str:
        text = self.combo.currentText()
        if text == self.NEW_ITEM_TEXT:
            return ""
        return text

    def set_value(self, value: str) -> None:
        self._suppress_signal = True
        value = value.strip() if value else ""
        
        if not value:
            self.combo.setCurrentIndex(0)
        else:
            idx = self.combo.findText(value, Qt.MatchFixedString)
            if idx >= 0:
                self.combo.setCurrentIndex(idx)
            else:
                # Add to vocabulary
                self._vocab.add_location(value)
                self._refresh_items()
                idx = self.combo.findText(value, Qt.MatchFixedString)
                if idx >= 0:
                    self.combo.setCurrentIndex(idx)
        
        self._suppress_signal = False

    def clear(self) -> None:
        self._suppress_signal = True
        self.combo.setCurrentIndex(0)
        self._suppress_signal = False


class TextFreeWidget(AnnotationWidget):
    """Multi-line free-form text input."""

    def __init__(self, field_def: FieldDefinition, parent=None):
        super().__init__(field_def, parent)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.text_edit = QPlainTextEdit()
        self.text_edit.setMaximumHeight(80)
        self.text_edit.setPlaceholderText(field_def.tooltip or "")
        self.text_edit.textChanged.connect(self._on_change)
        layout.addWidget(self.text_edit)

    def _on_change(self) -> None:
        self.value_changed.emit()

    def get_value(self) -> str:
        return self.text_edit.toPlainText().strip()

    def set_value(self, value: str) -> None:
        self.text_edit.setPlainText(value or "")

    def clear(self) -> None:
        self.text_edit.clear()


# =============================================================================
# WIDGET FACTORY
# =============================================================================

def create_widget_for_field(field_def: FieldDefinition, parent=None) -> AnnotationWidget:
    """
    Factory function to create the appropriate widget for a field definition.
    """
    widget_map = {
        AnnotationType.NUMERIC_INT: NumericIntWidget,
        AnnotationType.NUMERIC_FLOAT: NumericFloatWidget,
        AnnotationType.MORPHOMETRIC_CODE: ShortArmCodeEditor,
        AnnotationType.COLOR_CATEGORICAL: ColorCategoricalComboBox,
        AnnotationType.MORPH_CATEGORICAL: MorphCategoricalComboBox,
        AnnotationType.TEXT_HISTORY: TextHistoryComboBox,
        AnnotationType.TEXT_FREE: TextFreeWidget,
    }
    
    widget_class = widget_map.get(field_def.annotation_type)
    if widget_class is None:
        raise ValueError(f"Unknown annotation type: {field_def.annotation_type}")
    
    return widget_class(field_def, parent)

