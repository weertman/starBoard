# src/ui/map_picker.py
"""Map picker dialog for selecting named locations and coordinates."""
from __future__ import annotations

import json
import logging
from typing import Iterable, List, Optional, Tuple

from PySide6.QtCore import QObject, QUrl, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from src.data.location_registry import LocationRecord

log = logging.getLogger(__name__)

_webengine_available = False
try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWebChannel import QWebChannel
    _webengine_available = True
except ImportError:
    log.info("QtWebEngine not available — map picker will use fallback text fields")

DEFAULT_LAT = 48.546
DEFAULT_LON = -123.013
DEFAULT_ZOOM = 12


def is_map_picker_available() -> bool:
    """Return True if the map picker dialog can show an interactive map."""
    return _webengine_available


class _MapBridge(QObject):
    """Bridge between the Leaflet JS map and the Python dialog."""

    coordinatesChanged = Signal(float, float)
    locationSelected = Signal(str, float, float)

    @Slot(float, float)
    def onMapClick(self, lat: float, lon: float) -> None:
        self.coordinatesChanged.emit(lat, lon)

    @Slot(str, float, float)
    def onLocationSelected(self, name: str, lat: float, lon: float) -> None:
        self.locationSelected.emit(name, lat, lon)


def _location_payload(locations: Optional[Iterable[LocationRecord]]) -> str:
    payload = []
    for record in locations or []:
        if record.latitude is None or record.longitude is None:
            continue
        payload.append({"name": record.name, "latitude": record.latitude, "longitude": record.longitude})
    return json.dumps(payload)


def _build_map_html(
    lat: float,
    lon: float,
    zoom: int,
    has_pin: bool,
    locations: Optional[Iterable[LocationRecord]] = None,
) -> str:
    """Build the self-contained HTML page with Leaflet map and known-location markers."""
    pin_js = ""
    if has_pin:
        pin_js = f"var marker = L.marker([{lat}, {lon}]).addTo(map);"
    else:
        pin_js = "var marker = null;"
    locations_json = _location_payload(locations)

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
    .existing-location-marker {{ filter: hue-rotate(190deg); }}
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
    var knownLocations = {locations_json};

    new QWebChannel(qt.webChannelTransport, function(channel) {{
        bridge = channel.objects.bridge;
    }});

    function setSelection(name, lat, lng) {{
        var latlng = L.latLng(lat, lng);
        if (marker) {{
            marker.setLatLng(latlng);
        }} else {{
            marker = L.marker(latlng).addTo(map);
        }}
        map.panTo(latlng);
        if (bridge) {{
            bridge.onLocationSelected(name || '', lat, lng);
        }}
    }}

    knownLocations.forEach(function(loc) {{
        var locMarker = L.marker([loc.latitude, loc.longitude], {{title: loc.name}}).addTo(map);
        locMarker.getElement()?.classList.add('existing-location-marker');
        locMarker.bindPopup(loc.name);
        locMarker.on('click', function() {{
            setSelection(loc.name, loc.latitude, loc.longitude);
        }});
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


class MapPickerDialog(QDialog):
    """Dialog for picking an existing location or raw coordinates on a map."""

    def __init__(
        self,
        parent=None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        location_name: str = "",
        locations: Optional[List[LocationRecord]] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Pick Location on Map")
        self.setMinimumSize(700, 550)
        self.resize(850, 650)

        self._name = (location_name or "").strip()
        self._lat: Optional[float] = latitude
        self._lon: Optional[float] = longitude
        self._locations = list(locations or [])

        if not _webengine_available:
            self._build_fallback_ui()
            return
        self._build_map_ui()

    def _build_location_combo(self) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(True)
        combo.setInsertPolicy(QComboBox.NoInsert)
        combo.addItem("")
        for record in self._locations:
            combo.addItem(record.name, record)
        combo.setCurrentText(self._name)
        combo.currentTextChanged.connect(self._on_location_combo_changed)
        return combo

    def _build_map_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._web = QWebEngineView()
        self._web.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._bridge = _MapBridge()
        self._bridge.coordinatesChanged.connect(self._on_map_click)
        self._bridge.locationSelected.connect(self._on_location_selected)
        self._channel = QWebChannel()
        self._channel.registerObject("bridge", self._bridge)
        self._web.page().setWebChannel(self._channel)
        layout.addWidget(self._web, 1)

        row = QHBoxLayout()
        row.addWidget(QLabel("Location:"))
        self._name_combo = self._build_location_combo()
        row.addWidget(self._name_combo, 1)
        row.addWidget(QLabel("Latitude:"))
        self._lat_edit = QLineEdit()
        self._lat_edit.setPlaceholderText("e.g. 48.546")
        self._lat_edit.setMaximumWidth(140)
        row.addWidget(self._lat_edit)
        row.addWidget(QLabel("Longitude:"))
        self._lon_edit = QLineEdit()
        self._lon_edit.setPlaceholderText("e.g. -123.013")
        self._lon_edit.setMaximumWidth(140)
        row.addWidget(self._lon_edit)
        self._btn_go = QPushButton("Go")
        self._btn_go.clicked.connect(self._go_to_typed_coordinates)
        row.addWidget(self._btn_go)
        layout.addLayout(row)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        if self._lat is not None and self._lon is not None:
            self._lat_edit.setText(f"{self._lat:.6f}")
            self._lon_edit.setText(f"{self._lon:.6f}")
            center_lat, center_lon = self._lat, self._lon
            has_pin = True
        else:
            center_lat, center_lon = DEFAULT_LAT, DEFAULT_LON
            has_pin = False
        self._web.setHtml(
            _build_map_html(center_lat, center_lon, DEFAULT_ZOOM, has_pin, self._locations),
            QUrl("about:blank"),
        )

    def _build_fallback_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Map picker is not available (QtWebEngine not installed).\n"
            "Select or enter a location and coordinates manually below."
        ))

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Location:"))
        self._name_combo = self._build_location_combo()
        name_row.addWidget(self._name_combo, 1)
        layout.addLayout(name_row)

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

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _on_location_combo_changed(self, text: str) -> None:
        self._name = (text or "").strip()
        for record in self._locations:
            if record.name == self._name and record.latitude is not None and record.longitude is not None:
                self._lat = record.latitude
                self._lon = record.longitude
                self._lat_edit.setText(f"{record.latitude:.6f}")
                self._lon_edit.setText(f"{record.longitude:.6f}")
                if _webengine_available and hasattr(self, "_web"):
                    self._web.page().runJavaScript(f"window.setMarker({record.latitude}, {record.longitude});")
                return

    def _on_map_click(self, lat: float, lon: float) -> None:
        self._lat = lat
        self._lon = lon
        self._lat_edit.setText(f"{lat:.6f}")
        self._lon_edit.setText(f"{lon:.6f}")

    def _on_location_selected(self, name: str, lat: float, lon: float) -> None:
        self._name = (name or "").strip()
        self._lat = lat
        self._lon = lon
        self._name_combo.setCurrentText(self._name)
        self._lat_edit.setText(f"{lat:.6f}")
        self._lon_edit.setText(f"{lon:.6f}")

    def _go_to_typed_coordinates(self) -> None:
        try:
            lat = float(self._lat_edit.text().strip())
            lon = float(self._lon_edit.text().strip())
        except (ValueError, TypeError):
            QMessageBox.warning(self, "Invalid Coordinates", "Please enter valid numeric latitude and longitude values.")
            return
        if not (-90.0 <= lat <= 90.0):
            QMessageBox.warning(self, "Invalid Latitude", "Latitude must be between -90 and 90.")
            return
        if not (-180.0 <= lon <= 180.0):
            QMessageBox.warning(self, "Invalid Longitude", "Longitude must be between -180 and 180.")
            return
        self._lat = lat
        self._lon = lon
        if _webengine_available and hasattr(self, "_web"):
            self._web.page().runJavaScript(f"window.setMarker({lat}, {lon});")

    def _on_accept(self) -> None:
        self._name = self._name_combo.currentText().strip()
        lat_text = self._lat_edit.text().strip()
        lon_text = self._lon_edit.text().strip()
        if not lat_text and not lon_text:
            self._lat = None
            self._lon = None
            self.accept()
            return
        try:
            lat = float(lat_text)
            lon = float(lon_text)
        except (ValueError, TypeError):
            QMessageBox.warning(
                self,
                "Invalid Coordinates",
                "Please enter valid numeric latitude and longitude values, or clear both fields to accept without coordinates.",
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
        return self._lat, self._lon

    def get_location(self) -> Tuple[str, Optional[float], Optional[float]]:
        return self._name, self._lat, self._lon
