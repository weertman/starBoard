from src.data.location_registry import LocationRecord
from src.ui import map_picker
from src.ui.map_picker import _build_map_html


def test_map_html_renders_existing_location_markers_and_select_callback():
    html = _build_map_html(
        48.0,
        -123.0,
        12,
        False,
        locations=[LocationRecord("Eagle Point", 48.1, -123.1)],
    )

    assert "Eagle Point" in html
    assert "existing-location-marker" in html
    assert "bridge.onLocationSelected" in html


def test_map_picker_bridge_emits_location_selection():
    bridge = map_picker._MapBridge()
    captured = []
    bridge.locationSelected.connect(lambda name, lat, lon: captured.append((name, lat, lon)))

    bridge.onLocationSelected("Eagle Point", 48.1, -123.1)

    assert captured == [("Eagle Point", 48.1, -123.1)]
