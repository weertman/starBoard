# src/data/field_visits.py
"""
Field visit logging (unified survey/absence data).

Records visits to locations — whether or not stars were found.
Consolidates the former separate "location visits" and "negative outings"
systems into a single field visit log.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from .archive_paths import archive_root
from .csv_io import append_row, read_rows


FIELD_VISITS_FILENAME = "field_visits.csv"
FIELD_VISITS_HEADER = [
    "visit_date",        # ISO date (YYYY-MM-DD)
    "location",          # Optional location string
    "latitude",          # Decimal degrees (WGS84), optional
    "longitude",         # Decimal degrees (WGS84), optional
    "observers",         # Optional observer names
    "duration_minutes",  # Optional integer effort duration
    "notes",             # Optional notes (conditions, visibility, etc.)
    "created_utc",       # ISO timestamp when record was created
]


@dataclass
class FieldVisit:
    """A single field visit record."""
    visit_date: Optional[date]
    location: str
    observers: str
    duration_minutes: Optional[int]
    notes: str
    created_utc: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    def to_row(self) -> Dict[str, str]:
        return {
            "visit_date": self.visit_date.isoformat() if self.visit_date else "",
            "location": self.location,
            "latitude": f"{self.latitude:.6f}" if self.latitude is not None else "",
            "longitude": f"{self.longitude:.6f}" if self.longitude is not None else "",
            "observers": self.observers,
            "duration_minutes": str(self.duration_minutes) if self.duration_minutes is not None else "",
            "notes": self.notes,
            "created_utc": self.created_utc,
        }

    @classmethod
    def from_row(cls, row: Dict[str, str]) -> "FieldVisit":
        date_str = (row.get("visit_date") or row.get("outing_date") or "").strip()
        visit_date: Optional[date] = None
        if date_str:
            try:
                visit_date = date.fromisoformat(date_str)
            except ValueError:
                visit_date = None

        duration_str = (row.get("duration_minutes") or "").strip()
        duration_minutes: Optional[int] = None
        if duration_str:
            try:
                duration_minutes = int(duration_str)
            except ValueError:
                duration_minutes = None

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
            observers=(row.get("observers") or "").strip(),
            duration_minutes=duration_minutes,
            notes=(row.get("notes") or "").strip(),
            created_utc=(row.get("created_utc") or row.get("added_utc") or "").strip(),
            latitude=latitude,
            longitude=longitude,
        )


def _field_visits_csv_path() -> Path:
    return archive_root() / FIELD_VISITS_FILENAME


def _legacy_negative_outings_path() -> Path:
    return archive_root() / "negative_outings.csv"


def _legacy_location_visits_path() -> Path:
    return archive_root() / "location_visits.csv"


def append_field_visit(
    visit_date: date,
    location: str = "",
    observers: str = "",
    duration_minutes: Optional[int] = None,
    notes: str = "",
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> None:
    """Record a field visit."""
    record = FieldVisit(
        visit_date=visit_date,
        location=(location or "").strip(),
        observers=(observers or "").strip(),
        duration_minutes=duration_minutes,
        notes=(notes or "").strip(),
        created_utc=datetime.utcnow().isoformat() + "Z",
        latitude=latitude,
        longitude=longitude,
    )
    append_row(_field_visits_csv_path(), FIELD_VISITS_HEADER, record.to_row())


def read_field_visits(limit: Optional[int] = None) -> List[FieldVisit]:
    """
    Load all field visit records, sorted by visit_date descending (most recent first).

    Also reads legacy negative_outings.csv and location_visits.csv if they exist,
    for backward compatibility.
    """
    visits: List[FieldVisit] = []

    # Read primary file
    primary = _field_visits_csv_path()
    if primary.exists():
        for row in read_rows(primary):
            visits.append(FieldVisit.from_row(row))

    # Read legacy negative_outings.csv
    legacy_neg = _legacy_negative_outings_path()
    if legacy_neg.exists():
        for row in read_rows(legacy_neg):
            visits.append(FieldVisit.from_row(row))

    # Read legacy location_visits.csv
    legacy_lv = _legacy_location_visits_path()
    if legacy_lv.exists():
        for row in read_rows(legacy_lv):
            visits.append(FieldVisit.from_row(row))

    # Deduplicate by created_utc
    seen: Set[str] = set()
    deduped: List[FieldVisit] = []
    for v in visits:
        if v.created_utc and v.created_utc in seen:
            continue
        if v.created_utc:
            seen.add(v.created_utc)
        deduped.append(v)

    deduped.sort(
        key=lambda o: (
            (o.visit_date or date.min).toordinal(),
            o.created_utc,
        ),
        reverse=True,
    )
    if limit is not None:
        return deduped[: max(limit, 0)]
    return deduped


def delete_field_visit(created_utc: str) -> bool:
    """
    Delete a visit by its created_utc timestamp.

    Rewrites the primary CSV without the matching row.
    Returns True if a row was removed.
    """
    primary = _field_visits_csv_path()
    if not primary.exists():
        return False
    rows = read_rows(primary)
    before = len(rows)
    rows = [r for r in rows if (r.get("created_utc") or "").strip() != created_utc]
    if len(rows) == before:
        return False
    primary.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with primary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_VISITS_HEADER)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in FIELD_VISITS_HEADER})
    return True


def get_field_visit_locations() -> Set[str]:
    """Get all unique location strings from field visits."""
    locations: Set[str] = set()
    for visit in read_field_visits():
        if visit.location:
            locations.add(visit.location)
    return locations


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# These allow old code that hasn't been updated yet to keep working.
NegativeOuting = FieldVisit
append_negative_outing = append_field_visit
read_negative_outings = read_field_visits
get_negative_outing_locations = get_field_visit_locations
