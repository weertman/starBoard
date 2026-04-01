# src/data/location_visits.py
"""
Location visit tracking (absence data).

Records visits to locations where no stars were found, providing
negative/absence data that complements the positive sightings
captured through the normal query workflow.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

from .archive_paths import archive_root
from .csv_io import read_rows


# =============================================================================
# CONSTANTS
# =============================================================================

LOCATION_VISITS_FILENAME = "location_visits.csv"

LOCATION_VISITS_HEADER = [
    "visit_date",   # ISO date (YYYY-MM-DD)
    "location",     # Location name (from vocabulary)
    "latitude",     # Decimal degrees (WGS84), optional
    "longitude",    # Decimal degrees (WGS84), optional
    "notes",        # Free-text notes (conditions, visibility, etc.)
    "added_utc",    # Timestamp when this record was created
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class LocationVisit:
    """A single location visit record (no stars found)."""
    visit_date: Optional[date]
    location: str
    notes: str
    added_utc: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    def to_row(self) -> Dict[str, str]:
        return {
            "visit_date": self.visit_date.isoformat() if self.visit_date else "",
            "location": self.location,
            "latitude": f"{self.latitude:.6f}" if self.latitude is not None else "",
            "longitude": f"{self.longitude:.6f}" if self.longitude is not None else "",
            "notes": self.notes,
            "added_utc": self.added_utc,
        }

    @classmethod
    def from_row(cls, row: Dict[str, str]) -> "LocationVisit":
        date_str = (row.get("visit_date") or "").strip()
        visit_date = None
        if date_str:
            try:
                visit_date = date.fromisoformat(date_str)
            except ValueError:
                pass
        lat_str = (row.get("latitude") or "").strip()
        lon_str = (row.get("longitude") or "").strip()
        latitude = None
        longitude = None
        if lat_str:
            try:
                latitude = float(lat_str)
            except ValueError:
                pass
        if lon_str:
            try:
                longitude = float(lon_str)
            except ValueError:
                pass
        return cls(
            visit_date=visit_date,
            location=(row.get("location") or "").strip(),
            notes=(row.get("notes") or "").strip(),
            added_utc=(row.get("added_utc") or "").strip(),
            latitude=latitude,
            longitude=longitude,
        )


# =============================================================================
# FILE OPERATIONS
# =============================================================================

def _csv_path() -> Path:
    return archive_root() / LOCATION_VISITS_FILENAME


def _ensure_header(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOCATION_VISITS_HEADER)
        writer.writeheader()


def _append_row(path: Path, row: Dict[str, str]) -> None:
    _ensure_header(path)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOCATION_VISITS_HEADER)
        writer.writerow(row)


def _write_all(path: Path, visits: List[LocationVisit]) -> None:
    """Rewrite the entire CSV (used by delete)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOCATION_VISITS_HEADER)
        writer.writeheader()
        for v in visits:
            writer.writerow(v.to_row())


# =============================================================================
# PUBLIC API
# =============================================================================

def get_all_location_visits() -> List[LocationVisit]:
    """
    Load all location visit records, sorted by visit_date descending
    (most recent first).
    """
    path = _csv_path()
    if not path.exists():
        return []

    rows = read_rows(path)
    visits = [LocationVisit.from_row(r) for r in rows]
    visits.sort(
        key=lambda v: (v.visit_date is None, v.visit_date or date.min),
        reverse=True,
    )
    return visits


def add_location_visit(
    location: str,
    visit_date: Optional[date],
    notes: str = "",
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> bool:
    """
    Record a location visit where no stars were found.

    Returns True if saved, False if skipped (empty location).
    """
    location = (location or "").strip()
    if not location:
        return False

    visit = LocationVisit(
        visit_date=visit_date,
        location=location,
        notes=(notes or "").strip(),
        added_utc=datetime.utcnow().isoformat() + "Z",
        latitude=latitude,
        longitude=longitude,
    )

    _append_row(_csv_path(), visit.to_row())
    return True


def delete_location_visit(added_utc: str) -> bool:
    """
    Delete a visit by its added_utc timestamp (used as a unique key).

    Rewrites the CSV without the matching row.
    Returns True if a row was removed.
    """
    visits = get_all_location_visits()
    before = len(visits)
    visits = [v for v in visits if v.added_utc != added_utc]
    if len(visits) == before:
        return False
    _write_all(_csv_path(), visits)
    return True
