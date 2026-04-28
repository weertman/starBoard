import csv
import json
from pathlib import Path

from src.data import archive_paths as ap
from src.data.csv_io import append_row
from src.data.location_history import add_location_sighting
from src.data.location_registry import (
    LocationRecord,
    add_or_update_location,
    get_location,
    list_known_locations,
)
from src.data.vocabulary_store import get_vocabulary_store
from src.data.field_visits import append_field_visit


def _blank_row(header, id_col, id_value, **values):
    row = {col: "" for col in header}
    row[id_col] = id_value
    row.update(values)
    return row


def test_list_known_locations_merges_names_and_coordinates_from_archive_sources(tmp_path, monkeypatch):
    archive_root = tmp_path / "archive"
    monkeypatch.setenv("STARBOARD_ARCHIVE_DIR", str(archive_root))
    get_vocabulary_store().reload()

    add_or_update_location("Vocabulary Cove")

    gallery_csv, gallery_header = ap.metadata_csv_for("Gallery")
    append_row(
        gallery_csv,
        gallery_header,
        _blank_row(
            gallery_header,
            "gallery_id",
            "g1",
            location="Metadata Point",
            latitude="48.100000",
            longitude="-123.100000",
        ),
    )
    append_row(
        gallery_csv,
        gallery_header,
        _blank_row(
            gallery_header,
            "gallery_id",
            "g1",
            location="Metadata Point",
            latitude="48.200000",
            longitude="-123.200000",
        ),
    )

    append_field_visit(
        visit_date=__import__("datetime").date(2026, 4, 28),
        location="Field Visit Reef",
        latitude=48.3,
        longitude=-123.3,
    )
    add_location_sighting("g1", "History Wall", __import__("datetime").date(2026, 4, 27), "q1", 48.4, -123.4)

    records = {record.name: record for record in list_known_locations()}

    assert records["Vocabulary Cove"].latitude is None
    assert records["Metadata Point"].latitude == 48.2
    assert records["Metadata Point"].longitude == -123.2
    assert records["Field Visit Reef"].latitude == 48.3
    assert records["History Wall"].longitude == -123.4


def test_add_or_update_location_persists_name_without_breaking_existing_json_list(tmp_path, monkeypatch):
    archive_root = tmp_path / "archive"
    monkeypatch.setenv("STARBOARD_ARCHIVE_DIR", str(archive_root))
    get_vocabulary_store().reload()

    add_or_update_location("New Kelp Forest", 48.5, -123.5)

    assert get_location("New Kelp Forest") == LocationRecord("New Kelp Forest", 48.5, -123.5)
    locations_json = archive_root / "vocabularies" / "locations.json"
    assert json.loads(locations_json.read_text(encoding="utf-8")) == ["New Kelp Forest"]
