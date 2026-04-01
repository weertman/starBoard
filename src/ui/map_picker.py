# src/ui/map_picker.py
"""
Map picker dialog for selecting latitude/longitude coordinates.

Uses Leaflet + OpenStreetMap in a QWebEngineView.
Falls back gracefully if QtWebEngine is not available.
"""
from __future__ import annotations

import json
import logging
from typing import Optional, Tuple

from PySide6.QtCore import Qt, QObject, Slot, Signal, QUrl
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QDialogButtonBox, QMessageBox, QSizePolicy,
)

log = logging.getLogger(__name__)

# Lazy import of WebEngine — may not be installed everywhere
_webengine_available = False
try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWebChannel import QWebChannel
    _webengine_available = True
except ImportError:
    log.info("QtWebEngine not available — map picker will be disabled")


# Default center: San Juan Islands / FHL area
DEFAULT_LAT = 48.546
DEFAULT_LON = -123.013
DEFAULT_ZOOM = 12


def is_map_picker_available() -> bool:
    """Return True if the map picker dialog can be used."""
    return _webengine_available


# =============================================================================
# JavaScript bridge
# =============================================================================

class _MapBridge(QObject):
    """Bridge between the Leaflet JS map and the Python dialog."""
    coordinatesChanged = Signal(float, float)

    @Slot(float, float)
    def onMapClick(self, lat: float, lon: float) -> None:
        """Called from JavaScript when the user clicks the map."""
        self.coordinatesChanged.emit(lat, lon)


# =============================================================================
# HTML/JS for the Leaflet map
# =============================================================================

def _build_map_html(lat: float, lon: float, zoom: int, has_pin: bool) -> str:
    """Build the self-contained HTML page with Leaflet map."""
    pin_js = ""
    if has_pin:
        pin_js = f"""
        var marker = L.marker([{lat}, {lon}]).addTo(map);
        """
    else:
        pin_js = """
        var marker = null;
        """

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
<style>
    body {{ margin: 0; padding: 0; }}
    #map {{ width: 100%; height: 100vh; }}
</style>
</head>
<body>
<div id="map"></div>
<script>
    var map = L.map('map').setView([{lat}, {lon}], {zoom});
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
        attribution: '&copy; OpenStreetMap contributors',
        maxZoom: 19,
    }}).addTo(map);

    {pin_js}

    var bridge = null;
    new QWebChannel(qt.webChannelTransport, function(channel) {{
        bridge = channel.objects.bridge;
    }});

    map.on('click', function(e) {{
        var lat = e.latlng.lat;
        var lng = e.latlng.lng;
        if (marker) {{
            marker.setLatLng(e.latlng);
        }} else {{
            marker = L.marker(e.latlng).addTo(map);
        }}
        if (bridge) {{
            bridge.onMapClick(lat, lng);
        }}
    }});

    // Allow Python to move the marker programmatically
    window.setMarker = function(lat, lng) {{
        var latlng = L.latLng(lat, lng);
        if (marker) {{
            marker.setLatLng(latlng);
        }} else {{
            marker = L.marker(latlng).addTo(map);
        }}
        map.panTo(latlng);
    }};
</script>
</body>
</html>"""


# =============================================================================
# Dialog
# =============================================================================

class MapPickerDialog(QDialog):
    """
    Dialog for picking coordinates on an interactive map.

    Usage:
        dialog = MapPickerDialog(parent, latitude=48.5, longitude=-123.0)
        if dialog.exec() == QDialog.Accepted:
            lat, lon = dialog.get_coordinates()
    """

    def __init__(
        self,
        parent=None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Pick Location on Map")
        self.setMinimumSize(700, 550)
        self.resize(850, 650)

        self._lat: Optional[float] = latitude
        self._lon: Optional[float] = longitude

        if not _webengine_available:
            self._build_fallback_ui()
            return

        self._build_map_ui()

    def _build_map_ui(self) -> None:
        """Build the full map-based UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Map view
        self._web = QWebEngineView()
        self._web.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Web channel bridge
        self._bridge = _MapBridge()
        self._bridge.coordinatesChanged.connect(self._on_map_click)
        self._channel = QWebChannel()
        self._channel.registerObject("bridge", self._bridge)
        self._web.page().setWebChannel(self._channel)

        layout.addWidget(self._web, 1)

        # Coordinate display row
        coord_row = QHBoxLayout()
        coord_row.addWidget(QLabel("Latitude:"))
        self._lat_edit = QLineEdit()
        self._lat_edit.setPlaceholderText("e.g. 48.546")
        self._lat_edit.setMaximumWidth(140)
        coord_row.addWidget(self._lat_edit)

        coord_row.addWidget(QLabel("Longitude:"))
        self._lon_edit = QLineEdit()
        self._lon_edit.setPlaceholderText("e.g. -123.013")
        self._lon_edit.setMaximumWidth(140)
        coord_row.addWidget(self._lon_edit)

        self._btn_go = QPushButton("Go to coordinates")
        self._btn_go.clicked.connect(self._go_to_typed_coordinates)
        coord_row.addWidget(self._btn_go)

        coord_row.addStretch()

        hint = QLabel("Click the map to place a pin, or type coordinates and click 'Go'.")
        hint.setStyleSheet("color: #888;")
        coord_row.addWidget(hint)

        layout.addLayout(coord_row)

        # Dialog buttons
        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        # Populate initial values
        if self._lat is not None and self._lon is not None:
            self._lat_edit.setText(f"{self._lat:.6f}")
            self._lon_edit.setText(f"{self._lon:.6f}")
            center_lat, center_lon = self._lat, self._lon
            has_pin = True
        else:
            center_lat, center_lon = DEFAULT_LAT, DEFAULT_LON
            has_pin = False

        html = _build_map_html(center_lat, center_lon, DEFAULT_ZOOM, has_pin)
        self._web.setHtml(html, QUrl("about:blank"))

    def _build_fallback_ui(self) -> None:
        """Build a simple text-only UI when WebEngine is not available."""
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            "Map picker is not available (QtWebEngine not installed).\n"
            "Enter coordinates manually below."
        ))

        coord_row = QHBoxLayout()
        coord_row.addWidget(QLabel("Latitude:"))
        self._lat_edit = QLineEdit()
        self._lat_edit.setPlaceholderText("e.g. 48.546")
        coord_row.addWidget(self._lat_edit)

        coord_row.addWidget(QLabel("Longitude:"))
        self._lon_edit = QLineEdit()
        self._lon_edit.setPlaceholderText("e.g. -123.013")
        coord_row.addWidget(self._lon_edit)
        layout.addLayout(coord_row)

        if self._lat is not None:
            self._lat_edit.setText(f"{self._lat:.6f}")
        if self._lon is not None:
            self._lon_edit.setText(f"{self._lon:.6f}")

        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _on_map_click(self, lat: float, lon: float) -> None:
        """Handle a click on the map."""
        self._lat = lat
        self._lon = lon
        self._lat_edit.setText(f"{lat:.6f}")
        self._lon_edit.setText(f"{lon:.6f}")

    def _go_to_typed_coordinates(self) -> None:
        """Move the map pin to the coordinates typed in the text fields."""
        try:
            lat = float(self._lat_edit.text().strip())
            lon = float(self._lon_edit.text().strip())
        except (ValueError, TypeError):
            QMessageBox.warning(
                self, "Invalid Coordinates",
                "Please enter valid numeric latitude and longitude values.",
            )
            return

        if not (-90.0 <= lat <= 90.0):
            QMessageBox.warning(self, "Invalid Latitude", "Latitude must be between -90 and 90.")
            return
        if not (-180.0 <= lon <= 180.0):
            QMessageBox.warning(self, "Invalid Longitude", "Longitude must be between -180 and 180.")
            return

        self._lat = lat
        self._lon = lon

        if _webengine_available and hasattr(self, '_web'):
            self._web.page().runJavaScript(f"window.setMarker({lat}, {lon});")

    def _on_accept(self) -> None:
        """Validate and accept the dialog."""
        lat_text = self._lat_edit.text().strip()
        lon_text = self._lon_edit.text().strip()

        if not lat_text and not lon_text:
            # Accept with no coordinates (user chose not to set any)
            self._lat = None
            self._lon = None
            self.accept()
            return

        try:
            lat = float(lat_text)
            lon = float(lon_text)
        except (ValueError, TypeError):
            QMessageBox.warning(
                self, "Invalid Coordinates",
                "Please enter valid numeric latitude and longitude values, "
                "or clear both fields to accept without coordinates.",
            )
            return

        if not (-90.0 <= lat <= 90.0):
            QMessageBox.warning(self, "Invalid Latitude", "Latitude must be between -90 and 90.")
            return
        if not (-180.0 <= lon <= 180.0):
            QMessageBox.warning(self, "Invalid Longitude", "Longitude must be between -180 and 180.")
            return

        self._lat = lat
        self._lon = lon
        self.accept()

    def get_coordinates(self) -> Tuple[Optional[float], Optional[float]]:
        """Return the selected (latitude, longitude), or (None, None) if not set."""
        return self._lat, self._lon
