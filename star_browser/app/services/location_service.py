from __future__ import annotations

from src.data.archive_paths import metadata_csv_paths_for_read
from src.data.csv_io import read_rows_multi
from src.data.field_visits import read_field_visits


def _maybe_float(value: str) -> float | None:
    text = (value or '').strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def get_location_sites() -> dict:
    sites: dict[str, tuple[float, float]] = {}

    for target in ('Gallery', 'Queries'):
        for row in read_rows_multi(metadata_csv_paths_for_read(target)):
            name = (row.get('location') or '').strip()
            lat = _maybe_float(row.get('latitude') or '')
            lon = _maybe_float(row.get('longitude') or '')
            if name and lat is not None and lon is not None:
                sites[name] = (lat, lon)

    for visit in read_field_visits():
        if visit.location and visit.latitude is not None and visit.longitude is not None:
            sites[visit.location] = (visit.latitude, visit.longitude)

    return {
        'sites': [
            {'name': name, 'latitude': lat, 'longitude': lon}
            for name, (lat, lon) in sorted(sites.items())
        ]
    }
