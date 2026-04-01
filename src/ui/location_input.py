# src/ui/location_input.py
"""
Compound widget for location input: name + latitude + longitude + map picker.

Designed to replace the individual location/latitude/longitude fields in the
metadata form when the "location" group is rendered.

Implements the AnnotationWidget interface so it can be managed by MetadataFormV2.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QLineEdit, QPushButton, QCompleter, QSizePolicy,
)
from PySide6.QtGui import QDoubleValidator

from src.data.annotation_schema import FieldDefinition, AnnotationType, FIELD_BY_NAME
from src.data.vocabulary_store import get_vocabulary_store

log = logging.getLogger(__name__)


class LocationInputGroup(QWidget):
    """
    Compound widget for location input.

    Contains:
    - Location name (text with autocomplete from vocabulary)
    - Latitude (text field with numeric validation, accepts negatives)
    - Longitude (text field with numeric validation, accepts negatives)
    - "Pick on Map" button (opens MapPickerDialog if available)

    Emits value_changed when any sub-field changes.
    """
    value_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Row 1: Location name with autocomplete
        name_row = QHBoxLayout()
        name_row.setSpacing(6)
        lbl_name = QLabel("Name:")
        lbl_name.setFixedWidth(70)
        lbl_name.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        name_row.addWidget(lbl_name)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Location name (e.g. Eagle Point)")
        self._name_edit.setToolTip("Written description of the star's location")

        # Set up autocomplete from vocabulary
        try:
            store = get_vocabulary_store()
            locations = store.get_vocabulary("locations")
            if locations:
                completer = QCompleter(sorted(locations))
                completer.setCaseSensitivity(Qt.CaseInsensitive)
                completer.setFilterMode(Qt.MatchContains)
                self._name_edit.setCompleter(completer)
        except Exception:
            pass  # vocabulary not available — no autocomplete

        name_row.addWidget(self._name_edit, 1)
        layout.addLayout(name_row)

        # Row 2: Latitude + Longitude + Map button
        coord_row = QHBoxLayout()
        coord_row.setSpacing(6)

        lbl_lat = QLabel("Lat:")
        lbl_lat.setFixedWidth(70)
        lbl_lat.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        coord_row.addWidget(lbl_lat)

        self._lat_edit = QLineEdit()
        self._lat_edit.setPlaceholderText("e.g. 48.546")
        self._lat_edit.setToolTip("Latitude in decimal degrees (WGS84), range [-90, 90]")
        self._lat_edit.setMaximumWidth(130)
        lat_validator = QDoubleValidator(-90.0, 90.0, 8)
        lat_validator.setNotation(QDoubleValidator.StandardNotation)
        self._lat_edit.setValidator(lat_validator)
        coord_row.addWidget(self._lat_edit)

        lbl_lon = QLabel("Lon:")
        lbl_lon.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        coord_row.addWidget(lbl_lon)

        self._lon_edit = QLineEdit()
        self._lon_edit.setPlaceholderText("e.g. -123.013")
        self._lon_edit.setToolTip("Longitude in decimal degrees (WGS84), range [-180, 180]")
        self._lon_edit.setMaximumWidth(130)
        lon_validator = QDoubleValidator(-180.0, 180.0, 8)
        lon_validator.setNotation(QDoubleValidator.StandardNotation)
        self._lon_edit.setValidator(lon_validator)
        coord_row.addWidget(self._lon_edit)

        # Map picker button
        self._btn_map = QPushButton("🗺 Pick on Map")
        self._btn_map.setToolTip("Open an interactive map to pick coordinates")
        self._btn_map.setMaximumWidth(130)

        # Hide the button if map picker is not available
        try:
            from src.ui.map_picker import is_map_picker_available
            if not is_map_picker_available():
                self._btn_map.setVisible(False)
        except ImportError:
            self._btn_map.setVisible(False)

        coord_row.addWidget(self._btn_map)
        coord_row.addStretch()

        layout.addLayout(coord_row)

    def _connect_signals(self) -> None:
        self._name_edit.textChanged.connect(self._on_any_changed)
        self._lat_edit.textChanged.connect(self._on_any_changed)
        self._lon_edit.textChanged.connect(self._on_any_changed)
        self._btn_map.clicked.connect(self._open_map_picker)

    def _on_any_changed(self) -> None:
        self.value_changed.emit()

    def _open_map_picker(self) -> None:
        """Open the MapPickerDialog and populate lat/lon from the result."""
        try:
            from src.ui.map_picker import MapPickerDialog
        except ImportError:
            return

        current_lat = self.latitude()
        current_lon = self.longitude()

        dialog = MapPickerDialog(
            self,
            latitude=current_lat,
            longitude=current_lon,
        )
        from PySide6.QtWidgets import QDialog
        if dialog.exec() == QDialog.Accepted:
            lat, lon = dialog.get_coordinates()
            if lat is not None:
                self._lat_edit.setText(f"{lat:.6f}")
            else:
                self._lat_edit.clear()
            if lon is not None:
                self._lon_edit.setText(f"{lon:.6f}")
            else:
                self._lon_edit.clear()

    # -------------------------------------------------------------------------
    # Public API: individual field access
    # -------------------------------------------------------------------------

    def location_name(self) -> str:
        """Get the location name string."""
        return self._name_edit.text().strip()

    def set_location_name(self, name: str) -> None:
        """Set the location name string."""
        self._name_edit.setText(name)

    def latitude(self) -> Optional[float]:
        """Get latitude as a float, or None if empty/invalid."""
        text = self._lat_edit.text().strip()
        if not text:
            return None
        try:
            val = float(text)
            if -90.0 <= val <= 90.0:
                return val
        except (ValueError, TypeError):
            pass
        return None

    def set_latitude(self, lat: Optional[float]) -> None:
        """Set latitude from a float or None."""
        if lat is not None:
            self._lat_edit.setText(f"{lat:.6f}")
        else:
            self._lat_edit.clear()

    def longitude(self) -> Optional[float]:
        """Get longitude as a float, or None if empty/invalid."""
        text = self._lon_edit.text().strip()
        if not text:
            return None
        try:
            val = float(text)
            if -180.0 <= val <= 180.0:
                return val
        except (ValueError, TypeError):
            pass
        return None

    def set_longitude(self, lon: Optional[float]) -> None:
        """Set longitude from a float or None."""
        if lon is not None:
            self._lon_edit.setText(f"{lon:.6f}")
        else:
            self._lon_edit.clear()

    # -------------------------------------------------------------------------
    # Public API: bulk access (for MetadataFormV2 integration)
    # -------------------------------------------------------------------------

    def get_values(self) -> Dict[str, str]:
        """
        Get all three field values as a dict suitable for CSV row merging.

        Returns:
            {"location": "...", "latitude": "...", "longitude": "..."}
        """
        lat = self.latitude()
        lon = self.longitude()
        return {
            "location": self.location_name(),
            "latitude": f"{lat:.6f}" if lat is not None else "",
            "longitude": f"{lon:.6f}" if lon is not None else "",
        }

    def set_values(self, values: Dict[str, str]) -> None:
        """
        Set all three fields from a dict (e.g., from a CSV row).

        Expected keys: "location", "latitude", "longitude"
        Missing keys are ignored (not cleared).
        """
        if "location" in values:
            self._name_edit.setText((values["location"] or "").strip())
        if "latitude" in values:
            text = (values["latitude"] or "").strip()
            self._lat_edit.setText(text)
        if "longitude" in values:
            text = (values["longitude"] or "").strip()
            self._lon_edit.setText(text)

    def clear_all(self) -> None:
        """Clear all three fields."""
        self._name_edit.clear()
        self._lat_edit.clear()
        self._lon_edit.clear()

    # -------------------------------------------------------------------------
    # Public API: convenience tuple access
    # -------------------------------------------------------------------------

    def get_location(self) -> Tuple[str, Optional[float], Optional[float]]:
        """Return (name, latitude, longitude)."""
        return self.location_name(), self.latitude(), self.longitude()

    def set_location(
        self,
        name: str = "",
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> None:
        """Set all three values at once."""
        self.set_location_name(name)
        self.set_latitude(lat)
        self.set_longitude(lon)
