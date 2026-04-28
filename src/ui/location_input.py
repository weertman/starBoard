# src/ui/location_input.py
"""
Compound widget for location input: name + latitude + longitude + map picker.

Designed to replace the individual location/latitude/longitude fields in the
metadata form when the "location" group is rendered.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QComboBox, QCompleter,
)

from src.data.location_registry import add_or_update_location, get_location, list_known_locations

log = logging.getLogger(__name__)


class LocationInputGroup(QWidget):
    """
    Compound widget for location input.

    Contains:
    - Editable location-name combo box populated from known archive locations
    - Latitude and longitude text fields with numeric validation
    - "Pick on Map" button that can select existing locations or coordinates

    Emits value_changed when any sub-field changes.
    """

    value_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._suppress_location_lookup = False
        self._build_ui()
        self._connect_signals()
        self.refresh_locations()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        name_row = QHBoxLayout()
        name_row.setSpacing(6)
        lbl_name = QLabel("Name:")
        lbl_name.setFixedWidth(70)
        lbl_name.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        name_row.addWidget(lbl_name)

        self._name_combo = QComboBox()
        self._name_combo.setEditable(True)
        self._name_combo.setInsertPolicy(QComboBox.NoInsert)
        self._name_combo.setToolTip("Written description of the star's location")
        if self._name_combo.lineEdit() is not None:
            self._name_combo.lineEdit().setPlaceholderText("Location name (e.g. Eagle Point)")
        name_row.addWidget(self._name_combo, 1)
        layout.addLayout(name_row)

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

        self._btn_map = QPushButton("🗺 Pick on Map")
        self._btn_map.setToolTip("Open an interactive map to pick coordinates or an existing location")
        self._btn_map.setMaximumWidth(130)
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
        self._name_combo.currentTextChanged.connect(self._on_location_text_changed)
        self._lat_edit.textChanged.connect(self._on_any_changed)
        self._lon_edit.textChanged.connect(self._on_any_changed)
        self._btn_map.clicked.connect(self._open_map_picker)

    def refresh_locations(self) -> None:
        """Reload location choices from the canonical location registry."""
        current = self.location_name()
        try:
            records = list_known_locations()
        except Exception:
            records = []
        names = [record.name for record in records]

        self._suppress_location_lookup = True
        self._name_combo.blockSignals(True)
        try:
            self._name_combo.clear()
            self._name_combo.addItem("")
            self._name_combo.addItems(names)
            completer = self._name_combo.completer()
            if completer is not None:
                completer.setCaseSensitivity(Qt.CaseInsensitive)
                completer.setFilterMode(Qt.MatchContains)
                completer.setCompletionMode(QCompleter.PopupCompletion)
            self._name_combo.setCurrentText(current)
        finally:
            self._name_combo.blockSignals(False)
            self._suppress_location_lookup = False

    def _on_any_changed(self) -> None:
        self.value_changed.emit()

    def _on_location_text_changed(self, text: str) -> None:
        if self._suppress_location_lookup:
            self.value_changed.emit()
            return
        record = get_location(text.strip())
        if record is not None and record.latitude is not None and record.longitude is not None:
            self._lat_edit.blockSignals(True)
            self._lon_edit.blockSignals(True)
            try:
                self.set_latitude(record.latitude)
                self.set_longitude(record.longitude)
            finally:
                self._lat_edit.blockSignals(False)
                self._lon_edit.blockSignals(False)
        self.value_changed.emit()

    def _open_map_picker(self) -> None:
        """Open the MapPickerDialog and populate name/lat/lon from the result."""
        try:
            from src.ui.map_picker import MapPickerDialog
        except ImportError:
            return

        dialog = MapPickerDialog(
            self,
            location_name=self.location_name(),
            latitude=self.latitude(),
            longitude=self.longitude(),
            locations=list_known_locations(),
        )
        from PySide6.QtWidgets import QDialog
        if dialog.exec() == QDialog.Accepted:
            if hasattr(dialog, "get_location"):
                name, lat, lon = dialog.get_location()
                self.set_location(name, lat, lon)
            else:
                lat, lon = dialog.get_coordinates()
                self.set_location(self.location_name(), lat, lon)
            self.persist_current_location()

    def location_name(self) -> str:
        """Get the location name string."""
        return self._name_combo.currentText().strip()

    def set_location_name(self, name: str) -> None:
        """Set the location name string and populate known coordinates when available."""
        self._name_combo.setCurrentText((name or "").strip())
        self._on_location_text_changed((name or "").strip())

    def latitude(self) -> Optional[float]:
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
        if lat is not None:
            self._lat_edit.setText(f"{lat:.6f}")
        else:
            self._lat_edit.clear()

    def longitude(self) -> Optional[float]:
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
        if lon is not None:
            self._lon_edit.setText(f"{lon:.6f}")
        else:
            self._lon_edit.clear()

    def persist_current_location(self) -> None:
        """Persist the current non-empty location name and optional coordinates."""
        name = self.location_name()
        if not name:
            return
        add_or_update_location(name, self.latitude(), self.longitude())
        self.refresh_locations()

    def get_values(self) -> Dict[str, str]:
        lat = self.latitude()
        lon = self.longitude()
        return {
            "location": self.location_name(),
            "latitude": f"{lat:.6f}" if lat is not None else "",
            "longitude": f"{lon:.6f}" if lon is not None else "",
        }

    def set_values(self, values: Dict[str, str]) -> None:
        self._suppress_location_lookup = True
        try:
            if "location" in values:
                self._name_combo.setCurrentText((values["location"] or "").strip())
            if "latitude" in values:
                self._lat_edit.setText((values["latitude"] or "").strip())
            if "longitude" in values:
                self._lon_edit.setText((values["longitude"] or "").strip())
        finally:
            self._suppress_location_lookup = False

    def clear_all(self) -> None:
        self._name_combo.setCurrentText("")
        self._lat_edit.clear()
        self._lon_edit.clear()

    def get_location(self) -> Tuple[str, Optional[float], Optional[float]]:
        return self.location_name(), self.latitude(), self.longitude()

    def set_location(
        self,
        name: str = "",
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> None:
        self._suppress_location_lookup = True
        try:
            self._name_combo.setCurrentText((name or "").strip())
            self.set_latitude(lat)
            self.set_longitude(lon)
        finally:
            self._suppress_location_lookup = False
        self.value_changed.emit()
