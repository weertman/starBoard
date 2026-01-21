# src/ui/image_quality_panel.py
"""
Compact panel for annotating image sequence quality.

This widget provides comboboxes for the three image quality fields:
- Madreporite visibility (marker of bilateral symmetry)
- Anus visibility (marker of bilateral symmetry)
- Postural visibility (quality of posture for re-identification)

Designed to be embedded in First-order, Second-order, and Metadata tabs.
"""
from __future__ import annotations

from typing import Dict, Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QLabel, QComboBox,
    QPushButton, QGroupBox, QSizePolicy, QFrame,
)

from src.data.annotation_schema import (
    FIELD_BY_NAME, FieldDefinition, AnnotationType,
    MARKER_VISIBILITY_OPTIONS, POSTURAL_VISIBILITY_OPTIONS,
)
from src.data import archive_paths as ap
from src.data.csv_io import append_row, read_rows_multi, last_row_per_id, normalize_id_value
from src.utils.interaction_logger import get_interaction_logger


# The three image quality field names
IMAGE_QUALITY_FIELDS = [
    "madreporite_visibility",
    "anus_visibility",
    "postural_visibility",
]


class ImageQualityPanel(QWidget):
    """
    Compact panel with three comboboxes for image sequence quality annotation.
    
    Features:
    - Horizontal layout with labeled comboboxes
    - Optional Save button for standalone use
    - Emits value_changed signal when any combo changes
    - Can load/save values for a specific target+ID
    """
    value_changed = Signal()
    saved = Signal(str, str)  # Emits (target, id_value) after save
    
    def __init__(
        self,
        parent: Optional[QWidget] = None,
        show_save_button: bool = True,
        compact: bool = True,
        title: str = "Image Quality",
    ):
        super().__init__(parent)
        self._target: str = "Queries"
        self._id_value: str = ""
        self._show_save_button = show_save_button
        self._compact = compact
        self._combos: Dict[str, QComboBox] = {}
        self._loaded_values: Dict[str, str] = {}
        self._ilog = get_interaction_logger()
        
        self._build_ui(title)
    
    def _build_ui(self, title: str) -> None:
        """Build the panel UI."""
        if self._compact:
            self._build_compact_ui(title)
        else:
            self._build_form_ui(title)
    
    def _build_compact_ui(self, title: str) -> None:
        """Build a compact horizontal layout."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(8)
        
        # Title label (only if title is provided)
        if title:
            lbl_title = QLabel(f"<b>{title}:</b>")
            main_layout.addWidget(lbl_title)
        
        # Add comboboxes for each field
        for field_name in IMAGE_QUALITY_FIELDS:
            field_def = FIELD_BY_NAME.get(field_name)
            if field_def is None:
                continue
            
            # Short label (abbreviated)
            short_label = self._get_short_label(field_name)
            lbl = QLabel(short_label + ":")
            lbl.setToolTip(field_def.tooltip)
            main_layout.addWidget(lbl)
            
            # Combobox
            combo = self._create_combo_for_field(field_def)
            combo.setToolTip(field_def.tooltip)
            combo.currentIndexChanged.connect(self._on_value_changed)
            self._combos[field_name] = combo
            main_layout.addWidget(combo)
        
        main_layout.addStretch(1)
        
        # Optional save button
        if self._show_save_button:
            self.btn_save = QPushButton("Save")
            self.btn_save.setToolTip("Save image quality annotations")
            self.btn_save.clicked.connect(self._on_save_clicked)
            main_layout.addWidget(self.btn_save)
    
    def _build_form_ui(self, title: str) -> None:
        """Build a vertical form layout (for use in metadata tab)."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)
        
        # Group box
        group = QGroupBox(title)
        form = QFormLayout(group)
        form.setContentsMargins(8, 8, 8, 8)
        form.setSpacing(6)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        for field_name in IMAGE_QUALITY_FIELDS:
            field_def = FIELD_BY_NAME.get(field_name)
            if field_def is None:
                continue
            
            lbl = QLabel(field_def.display_name + ":")
            lbl.setToolTip(field_def.tooltip)
            
            combo = self._create_combo_for_field(field_def)
            combo.setToolTip(field_def.tooltip)
            combo.currentIndexChanged.connect(self._on_value_changed)
            self._combos[field_name] = combo
            
            form.addRow(lbl, combo)
        
        main_layout.addWidget(group)
        
        # Optional save button
        if self._show_save_button:
            btn_row = QHBoxLayout()
            btn_row.setContentsMargins(0, 0, 0, 0)
            btn_row.addStretch(1)
            self.btn_save = QPushButton("Save Quality Annotations")
            self.btn_save.setToolTip("Save image quality annotations")
            self.btn_save.clicked.connect(self._on_save_clicked)
            btn_row.addWidget(self.btn_save)
            main_layout.addLayout(btn_row)
    
    def _get_short_label(self, field_name: str) -> str:
        """Get abbreviated label for compact mode."""
        labels = {
            "madreporite_visibility": "Madreporite",
            "anus_visibility": "Anus",
            "postural_visibility": "Posture",
        }
        return labels.get(field_name, field_name)
    
    def _create_combo_for_field(self, field_def: FieldDefinition) -> QComboBox:
        """Create a combobox with options from the field definition."""
        combo = QComboBox()
        combo.setMinimumWidth(140)
        
        # Add empty option
        combo.addItem("", None)
        
        # Add categorical options
        for opt in field_def.options:
            combo.addItem(opt.label, opt.value)
        
        return combo
    
    def _on_value_changed(self) -> None:
        """Handle value change in any combo."""
        self.value_changed.emit()
    
    def _on_save_clicked(self) -> None:
        """Save the current values."""
        if not self._id_value:
            return
        self._ilog.log("button_click", "btn_save_image_quality", value=self._id_value,
                      context={"target": self._target, "values": self.get_values()})
        self.save_for_id(self._target, self._id_value)
    
    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    
    def set_target(self, target: str) -> None:
        """Set the target archive (Gallery or Queries)."""
        self._target = target
    
    def set_id_value(self, id_value: str) -> None:
        """Set the current ID being annotated."""
        self._id_value = id_value or ""
    
    def set_context(self, target: str, id_value: str) -> None:
        """Set both target and ID in one call."""
        self._target = target
        self._id_value = id_value or ""
    
    def get_values(self) -> Dict[str, str]:
        """Get current values from all comboboxes."""
        values: Dict[str, str] = {}
        for field_name, combo in self._combos.items():
            data = combo.currentData()
            if data is None:
                values[field_name] = ""
            else:
                values[field_name] = str(data)
        return values
    
    def set_values(self, values: Dict[str, str]) -> None:
        """Set values in all comboboxes."""
        for field_name, combo in self._combos.items():
            combo.blockSignals(True)
            value = values.get(field_name, "")
            
            if not value or not value.strip():
                combo.setCurrentIndex(0)
            else:
                # Try to match by value (numeric)
                found = False
                try:
                    val = float(value)
                    for i in range(combo.count()):
                        data = combo.itemData(i)
                        if data is not None and float(data) == val:
                            combo.setCurrentIndex(i)
                            found = True
                            break
                except ValueError:
                    pass
                
                # Try to match by label
                if not found:
                    idx = combo.findText(value, Qt.MatchFixedString)
                    if idx >= 0:
                        combo.setCurrentIndex(idx)
                    else:
                        combo.setCurrentIndex(0)
            
            combo.blockSignals(False)
        
        self._loaded_values = self.get_values()
    
    def clear(self) -> None:
        """Clear all comboboxes to empty state."""
        for combo in self._combos.values():
            combo.blockSignals(True)
            combo.setCurrentIndex(0)
            combo.blockSignals(False)
        self._loaded_values = {}
    
    def is_dirty(self) -> bool:
        """Check if values have changed since last load/save."""
        return self.get_values() != self._loaded_values
    
    def load_for_id(self, target: str, id_value: str) -> None:
        """Load values from CSV for a specific ID."""
        self.set_context(target, id_value)
        
        if not id_value:
            self.clear()
            return
        
        # Read latest row for this ID
        try:
            id_col = ap.id_column_name(target)
            csv_paths = self._get_csv_paths_for_read(target)
            rows = read_rows_multi(csv_paths)
            latest_map = last_row_per_id(rows, id_col)
            data = latest_map.get(normalize_id_value(id_value), {})
            self.set_values(data)
        except Exception:
            self.clear()
    
    def save_for_id(self, target: str, id_value: str) -> None:
        """Save current values to CSV for a specific ID."""
        if not id_value:
            return
        
        self.set_context(target, id_value)
        
        # Get the full row data (load existing, merge with our values)
        id_col = ap.id_column_name(target)
        csv_path, header = ap.metadata_csv_for(target)
        
        # Start with existing data
        try:
            csv_paths = self._get_csv_paths_for_read(target)
            rows = read_rows_multi(csv_paths)
            latest_map = last_row_per_id(rows, id_col)
            row = dict(latest_map.get(normalize_id_value(id_value), {}))
        except Exception:
            row = {}
        
        # Ensure all header columns exist
        for col in header:
            if col not in row:
                row[col] = ""
        
        # Set ID
        row[id_col] = id_value
        
        # Merge in our quality values
        for field_name, value in self.get_values().items():
            row[field_name] = value
        
        # Append to CSV
        append_row(csv_path, header, row)
        
        # Update loaded state
        self._loaded_values = self.get_values()
        
        # Emit saved signal
        self.saved.emit(target, id_value)
    
    def _get_csv_paths_for_read(self, target: str):
        """Get CSV paths for reading metadata."""
        try:
            return ap.metadata_csv_paths_for_read(target)
        except Exception:
            root = ap.archive_root()
            if target.lower() == "gallery":
                return [ap.gallery_root() / "gallery_metadata.csv"]
            candidates = [
                root / "querries" / "querries_metadata.csv",  # legacy
                root / "queries" / "queries_metadata.csv",    # new
            ]
            return [p for p in candidates if p.exists()]
    
    def setEnabled(self, enabled: bool) -> None:
        """Override to enable/disable all combos and save button."""
        super().setEnabled(enabled)
        for combo in self._combos.values():
            combo.setEnabled(enabled)
        if hasattr(self, 'btn_save'):
            self.btn_save.setEnabled(enabled and bool(self._id_value))

