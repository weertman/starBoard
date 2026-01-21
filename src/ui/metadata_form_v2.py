# src/ui/metadata_form_v2.py
"""
New metadata annotation form with typed fields and collapsible groups.

Replaces the old free-form metadata_form.py with a structured annotation system.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger("starBoard.ui.metadata_form_v2")
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QFormLayout, QLabel, QFrame,
    QSizePolicy,
)

from src.data.annotation_schema import (
    FIELD_DEFINITIONS, FIELD_GROUPS, FIELD_BY_NAME, GROUP_BY_NAME,
    FieldDefinition, FieldGroup, AnnotationType,
    GALLERY_HEADER_V2, QUERIES_HEADER_V2,
)
from src.ui.annotation_widgets import create_widget_for_field, AnnotationWidget
from src.ui.collapsible import CollapsibleSection


class MetadataFormV2(QWidget):
    """
    Structured metadata annotation form with typed fields.
    
    Features:
    - Collapsible field groups organized by annotation workflow
    - Type-specific input widgets (spinboxes, combo boxes, etc.)
    - Extensible vocabularies for colors and locations
    - Custom short arm code editor
    - Dirty state tracking for unsaved changes
    """
    value_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._target = "Gallery"
        self._id_value = ""
        self._loaded_snapshot: Dict[str, str] = {}
        self._widgets: Dict[str, AnnotationWidget] = {}
        self._groups: Dict[str, CollapsibleSection] = {}
        
        self._build_ui()
        self._loaded_snapshot = self.collect_row_dict()

    def _build_ui(self) -> None:
        """Build the form UI with collapsible groups."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Scrollable container
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(4, 4, 4, 4)
        container_layout.setSpacing(8)
        
        # Build each field group
        for group in FIELD_GROUPS:
            section = self._build_group_section(group)
            self._groups[group.name] = section
            container_layout.addWidget(section)
        
        container_layout.addStretch()
        scroll.setWidget(container)
        main_layout.addWidget(scroll, 1)  # stretch factor for vertical expansion

    def _build_group_section(self, group: FieldGroup) -> CollapsibleSection:
        """Build a collapsible section for a field group."""
        section = CollapsibleSection(
            group.display_name,
            start_collapsed=not group.start_expanded,
        )
        
        # Content widget with form layout
        content = QWidget()
        form = QFormLayout(content)
        form.setContentsMargins(8, 8, 8, 8)
        form.setSpacing(8)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        
        # Add fields for this group
        fields_created = []
        for field_name in group.fields:
            field_def = FIELD_BY_NAME.get(field_name)
            if field_def is None:
                logger.warning("Field definition not found: %s", field_name)
                continue
            
            # Create label
            label = QLabel(field_def.display_name + ":")
            label.setToolTip(field_def.tooltip)
            
            # Create widget
            widget = create_widget_for_field(field_def, content)
            widget.setToolTip(field_def.tooltip)
            widget.value_changed.connect(self._on_value_changed)
            self._widgets[field_name] = widget
            fields_created.append(field_name)
            
            form.addRow(label, widget)
        
        # Debug: log morph widget creation
        if group.name == "morphometric_auto":
            logger.debug("Created morphometric_auto widgets: %s", fields_created)
        
        section.setContent(content)
        return section

    def _on_value_changed(self) -> None:
        """Handle value changes from any widget."""
        self.value_changed.emit()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def set_target(self, target: str) -> None:
        """
        Set the target type (Gallery or Queries).
        
        Note: In v2, both targets use the same field schema.
        """
        self._target = target

    def set_id_value(self, value: str) -> None:
        """Set the ID value (injected at save time)."""
        self._id_value = value or ""

    def populate(self, values: Dict[str, str]) -> None:
        """
        Populate form from a dictionary of field values.
        
        Updates the loaded snapshot so is_dirty() returns False.
        """
        for field_name, widget in self._widgets.items():
            widget.set_value(values.get(field_name, ""))
        self._loaded_snapshot = self.collect_row_dict()

    def apply_values(self, values: Dict[str, str]) -> None:
        """
        Apply values without updating the loaded snapshot.
        
        Use this for carry-over values where is_dirty() should remain True.
        """
        # Debug: log morph_* fields being applied
        morph_fields = {k: v for k, v in values.items() if k.startswith('morph_')}
        if morph_fields:
            logger.debug("apply_values: Applying morph fields: %s", morph_fields)
        
        applied_count = 0
        for field_name, widget in self._widgets.items():
            if field_name in values:
                widget.set_value(values[field_name])
                if field_name.startswith('morph_'):
                    logger.debug("apply_values: Set %s = %s", field_name, values[field_name])
                applied_count += 1
        
        # Debug: check if morph widgets exist
        morph_widgets = [k for k in self._widgets.keys() if k.startswith('morph_')]
        if morph_fields and not morph_widgets:
            logger.warning("apply_values: No morph_* widgets found in form! Widgets: %s", list(self._widgets.keys()))

    def collect_row_dict(self) -> Dict[str, str]:
        """
        Collect all field values as a dictionary.
        
        Includes the ID column.
        """
        id_col = "gallery_id" if self._target == "Gallery" else "query_id"
        header = GALLERY_HEADER_V2 if self._target == "Gallery" else QUERIES_HEADER_V2
        
        out: Dict[str, str] = {k: "" for k in header}
        out[id_col] = self._id_value
        
        for field_name, widget in self._widgets.items():
            out[field_name] = widget.get_value()
        
        # Debug: log morph_* fields being collected
        morph_collected = {k: v for k, v in out.items() if k.startswith('morph_') and v}
        if morph_collected:
            logger.debug("collect_row_dict: Collected morph fields: %s", morph_collected)
        
        return out

    def collect_row(self) -> Dict[str, str]:
        """Alias for collect_row_dict() for API compatibility."""
        return self.collect_row_dict()

    def is_dirty(self) -> bool:
        """Check if any values have changed since last populate()."""
        return self.collect_row_dict() != self._loaded_snapshot

    def mark_clean(self) -> None:
        """Mark current values as the clean baseline (e.g., after saving)."""
        self._loaded_snapshot = self.collect_row_dict()

    def revert_to_loaded(self) -> None:
        """Revert all values to the last populated state."""
        if self._loaded_snapshot:
            self.populate(self._loaded_snapshot)

    def clear_all(self) -> None:
        """Clear all field values."""
        for widget in self._widgets.values():
            widget.clear()

    def get_widget(self, field_name: str) -> Optional[AnnotationWidget]:
        """Get the widget for a specific field."""
        return self._widgets.get(field_name)

    def expand_all_groups(self) -> None:
        """Expand all collapsible sections."""
        for section in self._groups.values():
            section.toggle.setChecked(True)

    def collapse_all_groups(self) -> None:
        """Collapse all collapsible sections."""
        for section in self._groups.values():
            section.toggle.setChecked(False)

    def _header(self):
        """Get the appropriate header for current target."""
        return GALLERY_HEADER_V2 if self._target == "Gallery" else QUERIES_HEADER_V2

    def _id_col(self) -> str:
        """Get the ID column name for current target."""
        return "gallery_id" if self._target == "Gallery" else "query_id"


# =============================================================================
# BACKWARD COMPATIBILITY WRAPPER
# =============================================================================

class MetadataForm(MetadataFormV2):
    """
    Alias for MetadataFormV2 providing backward compatibility.
    
    Drop-in replacement for the old MetadataForm class.
    """
    pass

