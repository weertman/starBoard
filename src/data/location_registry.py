# src/data/location_registry.py
"""Canonical location discovery and persistence helpers for desktop UI widgets."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from src.data import archive_paths as ap
from src.data.csv_io import last_row_per_id, read_rows, read_rows_multi
from src.data.vocabulary_store import get_vocabulary_store


@dataclass(frozen=True)
class LocationRecord:
    """A named place with optional WGS84 coordinates."""

    name: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None


def _coordinates_path() -> Path:
    return ap.archive_root() / "vocabularies" / "location_coordinates.json"


def _parse_float(value) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _valid_coordinates(lat: Optional[float], lon: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    if lat is None or lon is None:
        return None, None
    if not (-90.0 <= lat <= 90.0):
        return None, None
    if not (-180.0 <= lon <= 180.0):
        return None, None
    return lat, lon


def _record_from_row(row: Dict[str, str]) -> Optional[LocationRecord]:
    name = (row.get("location") or "").strip()
    if not name:
        return None
    lat, lon = _valid_coordinates(_parse_float(row.get("latitude")), _parse_float(row.get("longitude")))
    return LocationRecord(name=name, latitude=lat, longitude=lon)


def _load_persisted_coordinates() -> Dict[str, Tuple[float, float]]:
    path = _coordinates_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    out: Dict[str, Tuple[float, float]] = {}
    for name, coords in payload.items():
        if not isinstance(name, str) or not isinstance(coords, dict):
            continue
        lat, lon = _valid_coordinates(_parse_float(coords.get("latitude")), _parse_float(coords.get("longitude")))
        if lat is not None and lon is not None:
            out[name.strip().lower()] = (lat, lon)
    return out


def _save_persisted_coordinates(updates: Dict[str, Tuple[float, float]]) -> None:
    path = _coordinates_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: Dict[str, Dict[str, float]] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                existing = {str(k): v for k, v in payload.items() if isinstance(v, dict)}
        except (OSError, json.JSONDecodeError):
            existing = {}
    for name, (lat, lon) in updates.items():
        existing[name] = {"latitude": lat, "longitude": lon}
    path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")


def _merge_record(records: Dict[str, LocationRecord], record: LocationRecord) -> None:
    name = record.name.strip()
    if not name:
        return
    key = name.lower()
    current = records.get(key)
    if current is None:
        records[key] = LocationRecord(name, record.latitude, record.longitude)
        return
    if current.latitude is None and record.latitude is not None and record.longitude is not None:
        records[key] = LocationRecord(current.name, record.latitude, record.longitude)


def _iter_metadata_records() -> Iterable[LocationRecord]:
    for target in ("Gallery", "Queries"):
        try:
            paths = ap.metadata_csv_paths_for_read(target)
            id_col = ap.id_column_name(target)
            latest_rows = last_row_per_id(read_rows_multi(paths), id_col).values()
        except Exception:
            latest_rows = []
        for row in latest_rows:
            record = _record_from_row(row)
            if record is not None:
                yield record


def _iter_field_visit_records() -> Iterable[LocationRecord]:
    try:
        from src.data.field_visits import read_field_visits

        visits = read_field_visits()
    except Exception:
        visits = []
    for visit in visits:
        name = (getattr(visit, "location", "") or "").strip()
        if not name:
            continue
        lat, lon = _valid_coordinates(getattr(visit, "latitude", None), getattr(visit, "longitude", None))
        yield LocationRecord(name, lat, lon)


def _iter_location_history_records() -> Iterable[LocationRecord]:
    try:
        gallery_root = ap.gallery_root()
    except Exception:
        return
    if not gallery_root.exists():
        return
    for path in gallery_root.glob("*/_location_history.csv"):
        for row in read_rows(path):
            record = _record_from_row(row)
            if record is not None:
                yield record


def list_known_locations() -> List[LocationRecord]:
    """Return known location names with best available coordinates."""
    records: Dict[str, LocationRecord] = {}

    try:
        vocab = get_vocabulary_store()
        vocab.reload()
        for name in vocab.get_locations():
            _merge_record(records, LocationRecord(name.strip()))
    except Exception:
        pass

    persisted = _load_persisted_coordinates()
    for record in list(records.values()):
        coords = persisted.get(record.name.lower())
        if coords:
            _merge_record(records, LocationRecord(record.name, coords[0], coords[1]))

    for record in _iter_metadata_records():
        _merge_record(records, record)
    for record in _iter_field_visit_records():
        _merge_record(records, record)
    for record in _iter_location_history_records():
        _merge_record(records, record)

    # Re-apply explicitly saved coordinates last so manually curated coordinates win.
    for key, (lat, lon) in persisted.items():
        current = records.get(key)
        if current is not None:
            records[key] = LocationRecord(current.name, lat, lon)

    return sorted(records.values(), key=lambda item: item.name.lower())


def get_location(name: str) -> Optional[LocationRecord]:
    key = (name or "").strip().lower()
    if not key:
        return None
    for record in list_known_locations():
        if record.name.lower() == key:
            return record
    return None


def add_or_update_location(
    name: str,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> Optional[LocationRecord]:
    """Persist a user-entered location name and optional coordinates."""
    clean_name = (name or "").strip()
    if not clean_name:
        return None

    try:
        vocab = get_vocabulary_store()
        vocab.reload()
        vocab.add_location(clean_name)
    except Exception:
        pass

    lat, lon = _valid_coordinates(latitude, longitude)
    if lat is not None and lon is not None:
        _save_persisted_coordinates({clean_name: (lat, lon)})
        return LocationRecord(clean_name, lat, lon)
    return LocationRecord(clean_name)
