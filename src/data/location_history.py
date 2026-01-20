# src/data/location_history.py
"""
Location history tracking for gallery individuals.

Records where each gallery member has been observed over time.
When queries are merged into a gallery individual, their location
and observation date are recorded as sightings.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from .archive_paths import gallery_root
from .csv_io import read_rows


# =============================================================================
# CONSTANTS
# =============================================================================

LOCATION_HISTORY_FILENAME = "_location_history.csv"

LOCATION_HISTORY_HEADER = [
    "observation_date",  # ISO date (YYYY-MM-DD) when the sighting occurred
    "location",          # Location name from query metadata
    "query_id",          # Source query that was merged
    "added_utc",         # Timestamp when this record was created
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class LocationSighting:
    """A single location sighting record."""
    observation_date: Optional[date]
    location: str
    query_id: str
    added_utc: str

    def to_row(self) -> Dict[str, str]:
        """Convert to CSV row dict."""
        return {
            "observation_date": self.observation_date.isoformat() if self.observation_date else "",
            "location": self.location,
            "query_id": self.query_id,
            "added_utc": self.added_utc,
        }

    @classmethod
    def from_row(cls, row: Dict[str, str]) -> "LocationSighting":
        """Create from CSV row dict."""
        date_str = (row.get("observation_date") or "").strip()
        obs_date = None
        if date_str:
            try:
                obs_date = date.fromisoformat(date_str)
            except ValueError:
                pass
        return cls(
            observation_date=obs_date,
            location=(row.get("location") or "").strip(),
            query_id=(row.get("query_id") or "").strip(),
            added_utc=(row.get("added_utc") or "").strip(),
        )


# =============================================================================
# FILE OPERATIONS
# =============================================================================

def _history_path(gallery_id: str) -> Path:
    """Get the location history CSV path for a gallery individual."""
    return gallery_root() / gallery_id / LOCATION_HISTORY_FILENAME


def _ensure_header(path: Path) -> None:
    """Ensure CSV file exists with header."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOCATION_HISTORY_HEADER)
        writer.writeheader()


def _append_row(path: Path, row: Dict[str, str]) -> None:
    """Append a single row to CSV."""
    _ensure_header(path)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOCATION_HISTORY_HEADER)
        writer.writerow(row)


# =============================================================================
# PUBLIC API
# =============================================================================

def get_location_history(gallery_id: str) -> List[LocationSighting]:
    """
    Load all location sighting records for a gallery individual.
    
    Returns:
        List of LocationSighting objects, ordered by observation_date (oldest first)
    """
    path = _history_path(gallery_id)
    if not path.exists():
        return []
    
    rows = read_rows(path)
    sightings = [LocationSighting.from_row(r) for r in rows]
    
    # Sort by observation date (None dates go last)
    sightings.sort(key=lambda s: (s.observation_date is None, s.observation_date or date.max))
    return sightings


def add_location_sighting(
    gallery_id: str,
    location: str,
    observation_date: Optional[date],
    query_id: str,
) -> bool:
    """
    Add a new location sighting to a gallery individual's history.
    
    Args:
        gallery_id: The gallery individual ID
        location: Location name (skipped if empty)
        observation_date: Date of the sighting
        query_id: Source query ID
        
    Returns:
        True if sighting was added, False if skipped (empty location)
    """
    location = (location or "").strip()
    if not location:
        return False
    
    sighting = LocationSighting(
        observation_date=observation_date,
        location=location,
        query_id=query_id,
        added_utc=datetime.utcnow().isoformat() + "Z",
    )
    
    path = _history_path(gallery_id)
    _append_row(path, sighting.to_row())
    return True


def get_unique_locations_for_gallery(gallery_id: str) -> Set[str]:
    """
    Get the set of all unique locations where a gallery individual has been seen.
    
    Returns:
        Set of location strings (empty locations excluded)
    """
    sightings = get_location_history(gallery_id)
    return {s.location for s in sightings if s.location}


def has_location_in_history(gallery_id: str, location: str) -> bool:
    """
    Check if a gallery individual has ever been seen at a specific location.
    
    Args:
        gallery_id: The gallery individual ID
        location: Location to check (case-sensitive)
        
    Returns:
        True if the individual has been seen at this location
    """
    return location in get_unique_locations_for_gallery(gallery_id)


def find_galleries_with_location_history(
    gallery_ids: List[str],
    required_locations: Set[str],
) -> Set[str]:
    """
    Find gallery IDs that have been seen at ANY of the required locations.
    
    Args:
        gallery_ids: List of gallery IDs to check
        required_locations: Set of location names to match against
        
    Returns:
        Set of gallery IDs that have at least one sighting at any required location
    """
    if not required_locations:
        return set(gallery_ids)
    
    result: Set[str] = set()
    for gid in gallery_ids:
        locations = get_unique_locations_for_gallery(gid)
        if locations & required_locations:  # intersection
            result.add(gid)
    return result


def get_all_historical_locations(gallery_ids: List[str]) -> Set[str]:
    """
    Get all unique locations across all gallery individuals' histories.
    
    Useful for populating filter dropdowns.
    
    Args:
        gallery_ids: List of gallery IDs to scan
        
    Returns:
        Set of all location strings found in any history
    """
    all_locations: Set[str] = set()
    for gid in gallery_ids:
        all_locations.update(get_unique_locations_for_gallery(gid))
    return all_locations



