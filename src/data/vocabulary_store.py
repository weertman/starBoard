# src/data/vocabulary_store.py
"""
Vocabulary persistence layer for extensible categorical fields.

Manages color vocabularies and location history for the annotation system.
Vocabularies are stored as JSON files in the archive/vocabularies directory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Set, Optional
from threading import Lock

from src.data.archive_paths import archive_root
from src.data.annotation_schema import DEFAULT_COLORS


class VocabularyStore:
    """
    Thread-safe singleton store for annotation vocabularies.
    
    Manages:
    - Color vocabularies (per-field, extensible)
    - Location history (extensible)
    """
    _instance: Optional["VocabularyStore"] = None
    _lock = Lock()

    def __new__(cls) -> "VocabularyStore":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        self._vocabularies_dir: Path = archive_root() / "vocabularies"
        self._colors: Dict[str, List[str]] = {}  # field_name -> list of colors
        self._locations: List[str] = []
        self._color_sets: Dict[str, Set[str]] = {}  # for fast lookup
        self._locations_set: Set[str] = set()
        self._dirty: Set[str] = set()  # track modified vocabularies
        
        self._ensure_dir()
        self._load_all()

    def _ensure_dir(self) -> None:
        """Ensure vocabularies directory exists."""
        self._vocabularies_dir.mkdir(parents=True, exist_ok=True)

    def _vocab_path(self, name: str) -> Path:
        """Get path for a vocabulary file."""
        return self._vocabularies_dir / f"{name}.json"

    def _load_all(self) -> None:
        """Load all vocabularies from disk."""
        # Load color vocabularies
        for path in self._vocabularies_dir.glob("colors_*.json"):
            field_name = path.stem.replace("colors_", "")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    colors = json.load(f)
                    if isinstance(colors, list):
                        self._colors[field_name] = colors
                        self._color_sets[field_name] = set(c.lower() for c in colors)
            except (json.JSONDecodeError, IOError):
                pass

        # Load locations
        locations_path = self._vocab_path("locations")
        if locations_path.exists():
            try:
                with open(locations_path, "r", encoding="utf-8") as f:
                    locs = json.load(f)
                    if isinstance(locs, list):
                        self._locations = locs
                        self._locations_set = set(l.lower() for l in locs)
            except (json.JSONDecodeError, IOError):
                pass

    def _save_colors(self, field_name: str) -> None:
        """Save color vocabulary for a field."""
        path = self._vocabularies_dir / f"colors_{field_name}.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._colors.get(field_name, []), f, indent=2)
        except IOError:
            pass

    def _save_locations(self) -> None:
        """Save locations vocabulary."""
        path = self._vocab_path("locations")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._locations, f, indent=2)
        except IOError:
            pass

    # -------------------------------------------------------------------------
    # Public API: Colors
    # -------------------------------------------------------------------------

    def get_colors(self, field_name: str) -> List[str]:
        """
        Get color vocabulary for a field.
        Returns default colors if no custom vocabulary exists.
        """
        if field_name not in self._colors:
            # Initialize with defaults
            self._colors[field_name] = list(DEFAULT_COLORS)
            self._color_sets[field_name] = set(c.lower() for c in DEFAULT_COLORS)
            self._save_colors(field_name)
        return list(self._colors[field_name])

    def add_color(self, field_name: str, color: str) -> bool:
        """
        Add a new color to a field's vocabulary.
        Returns True if added, False if already exists.
        """
        color = color.strip()
        if not color:
            return False

        # Ensure vocabulary exists
        if field_name not in self._colors:
            self.get_colors(field_name)

        # Check if already exists (case-insensitive)
        if color.lower() in self._color_sets[field_name]:
            return False

        self._colors[field_name].append(color)
        self._color_sets[field_name].add(color.lower())
        self._save_colors(field_name)
        return True

    def has_color(self, field_name: str, color: str) -> bool:
        """Check if a color exists in a field's vocabulary."""
        if field_name not in self._color_sets:
            self.get_colors(field_name)
        return color.lower() in self._color_sets[field_name]

    # -------------------------------------------------------------------------
    # Public API: Locations
    # -------------------------------------------------------------------------

    def get_locations(self) -> List[str]:
        """Get all known locations."""
        return list(self._locations)

    def add_location(self, location: str) -> bool:
        """
        Add a new location to the vocabulary.
        Returns True if added, False if already exists.
        """
        location = location.strip()
        if not location:
            return False

        if location.lower() in self._locations_set:
            return False

        self._locations.append(location)
        self._locations_set.add(location.lower())
        self._save_locations()
        return True

    def has_location(self, location: str) -> bool:
        """Check if a location exists in the vocabulary."""
        return location.lower() in self._locations_set

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def reload(self) -> None:
        """Reload all vocabularies from disk."""
        self._colors.clear()
        self._color_sets.clear()
        self._locations.clear()
        self._locations_set.clear()
        self._load_all()


def get_vocabulary_store() -> VocabularyStore:
    """Get the singleton vocabulary store instance."""
    return VocabularyStore()



