from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from .archive_paths import archive_root
from .csv_io import append_row, read_rows


NEGATIVE_OUTINGS_FILENAME = "negative_outings.csv"
NEGATIVE_OUTINGS_HEADER = [
    "outing_date",       # ISO date (YYYY-MM-DD)
    "location",          # Optional location string
    "latitude",          # Decimal degrees (WGS84), optional
    "longitude",         # Decimal degrees (WGS84), optional
    "observers",         # Optional observer names
    "duration_minutes",  # Optional integer effort duration
    "notes",             # Optional notes (conditions, visibility, etc.)
    "created_utc",       # ISO timestamp when record was created
]


@dataclass
class NegativeOuting:
    outing_date: Optional[date]
    location: str
    observers: str
    duration_minutes: Optional[int]
    notes: str
    created_utc: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    def to_row(self) -> Dict[str, str]:
        return {
            "outing_date": self.outing_date.isoformat() if self.outing_date else "",
            "location": self.location,
            "latitude": f"{self.latitude:.6f}" if self.latitude is not None else "",
            "longitude": f"{self.longitude:.6f}" if self.longitude is not None else "",
            "observers": self.observers,
            "duration_minutes": str(self.duration_minutes) if self.duration_minutes is not None else "",
            "notes": self.notes,
            "created_utc": self.created_utc,
        }

    @classmethod
    def from_row(cls, row: Dict[str, str]) -> "NegativeOuting":
        date_str = (row.get("outing_date") or "").strip()
        outing_date: Optional[date] = None
        if date_str:
            try:
                outing_date = date.fromisoformat(date_str)
            except ValueError:
                outing_date = None

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
            outing_date=outing_date,
            location=(row.get("location") or "").strip(),
            observers=(row.get("observers") or "").strip(),
            duration_minutes=duration_minutes,
            notes=(row.get("notes") or "").strip(),
            created_utc=(row.get("created_utc") or "").strip(),
            latitude=latitude,
            longitude=longitude,
        )


def _negative_outings_csv_path() -> Path:
    return archive_root() / NEGATIVE_OUTINGS_FILENAME


def append_negative_outing(
    outing_date: date,
    location: str = "",
    observers: str = "",
    duration_minutes: Optional[int] = None,
    notes: str = "",
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> None:
    record = NegativeOuting(
        outing_date=outing_date,
        location=(location or "").strip(),
        observers=(observers or "").strip(),
        duration_minutes=duration_minutes,
        notes=(notes or "").strip(),
        created_utc=datetime.utcnow().isoformat() + "Z",
        latitude=latitude,
        longitude=longitude,
    )
    append_row(_negative_outings_csv_path(), NEGATIVE_OUTINGS_HEADER, record.to_row())


def read_negative_outings(limit: Optional[int] = None) -> List[NegativeOuting]:
    rows = read_rows(_negative_outings_csv_path())
    outings = [NegativeOuting.from_row(r) for r in rows]

    outings.sort(
        key=lambda o: (
            (o.outing_date or date.min).toordinal(),
            o.created_utc,
        ),
        reverse=True,
    )
    if limit is not None:
        return outings[: max(limit, 0)]
    return outings


def get_negative_outing_locations() -> Set[str]:
    locations: Set[str] = set()
    for outing in read_negative_outings():
        if outing.location:
            locations.add(outing.location)
    return locations

