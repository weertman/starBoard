import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import archive_paths as ap
from src.data.csv_io import append_row, last_row_per_id, read_rows
from src.data.image_index import invalidate_image_cache, list_image_files
from src.data.legacy_morphometric_merge import (
    LEGACY_PROVENANCE_KEY,
    LEGACY_SOURCE_ROOT_NAME,
    backfill_legacy_gallery_archive_images,
    build_archive_backfill_plan,
    build_import_plan,
    import_legacy_gallery_measurements,
    list_mfolders,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_bundle(
    source_root: Path,
    *,
    alias: str,
    date_folder: str | None = "10_27_2025",
    mfolder_name: str = "mFolder_1",
    missing: tuple[str, ...] = (),
) -> Path:
    parent = source_root / alias
    if date_folder:
        parent = parent / date_folder
    mfolder = parent / mfolder_name
    mfolder.mkdir(parents=True, exist_ok=True)

    detection = {
        "class_id": 3,
        "class_name": "Pycnopodia_helianthoides",
        "corrected_polygon": [[0, 0], [1, 1]],
        "mm_per_pixel": 0.5,
    }
    morph = {
        "area_mm2": 12.5,
        "num_arms": 5,
        "arm_data": [
            [1, 1.0, 0.0, 10.0],
            [2, -1.0, 0.0, 12.0],
        ],
        "major_axis_mm": 20.0,
        "minor_axis_mm": 10.0,
        "contour_coordinates": [[0, 0], [1, 1]],
        "mm_per_pixel": 0.5,
        "user_initials": "abc",
        "user_notes": "",
    }

    if "corrected_detection.json" not in missing:
        _write_json(mfolder / "corrected_detection.json", detection)
    if "morphometrics.json" not in missing:
        _write_json(mfolder / "morphometrics.json", morph)

    for filename in (
        "corrected_mask.png",
        "corrected_object.png",
        "raw_frame.png",
        "checkerboard_with_object.png",
    ):
        if filename in missing:
            continue
        (mfolder / filename).write_bytes(b"legacy-test")

    return mfolder


def _seed_gallery_csv(csv_path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    for row in rows:
        append_row(csv_path, header, row)


def _blank_row(header: list[str], gallery_id: str, **values: str) -> dict[str, str]:
    row = {column: "" for column in header}
    row["gallery_id"] = gallery_id
    row.update(values)
    return row


def test_build_import_plan_keeps_newest_valid_same_day(tmp_path):
    source_root = tmp_path / LEGACY_SOURCE_ROOT_NAME
    older_valid = _write_bundle(source_root, alias="slushie", mfolder_name="mFolder_1")
    _write_bundle(
        source_root,
        alias="slushie",
        mfolder_name="mFolder_2",
        missing=("morphometrics.json",),
    )

    plan, results = build_import_plan(
        source_root,
        gallery_ids=["slushie"],
        location_hint="10-27-2025_samish_photo_sample",
    )

    assert len(plan) == 1
    assert plan[0].gallery_id == "slushie"
    assert plan[0].source_mfolder == older_valid

    skipped_invalid = [item for item in results if item.status == "skipped_invalid_bundle"]
    assert len(skipped_invalid) == 1
    assert skipped_invalid[0].missing_files == ["morphometrics.json"]


def test_build_import_plan_normalizes_alias_and_direct_mfolder_date(tmp_path):
    source_root = tmp_path / LEGACY_SOURCE_ROOT_NAME
    direct_bundle = _write_bundle(
        source_root,
        alias="cheeto ",
        date_folder=None,
        mfolder_name="mFolder_1",
    )

    plan, results = build_import_plan(source_root, gallery_ids=["cheeto"])

    assert not [item for item in results if item.status == "skipped_missing_gallery"]
    assert len(plan) == 1
    assert plan[0].gallery_id == "cheeto"
    assert plan[0].measurement_date == "10_27_2025"
    assert plan[0].location == "10-27-2025_samish_photo_sample"
    assert plan[0].source_mfolder == direct_bundle


def test_import_patches_bundle_and_appends_gallery_metadata(tmp_path, monkeypatch):
    archive_root = tmp_path / "archive"
    monkeypatch.setenv("STARBOARD_ARCHIVE_DIR", str(archive_root))

    source_root = tmp_path / LEGACY_SOURCE_ROOT_NAME
    source_mfolder = _write_bundle(source_root, alias="apricot")
    measurements_root = tmp_path / "measurements"

    gallery_csv_path, header = ap.metadata_csv_for("Gallery")
    (archive_root / "gallery" / "apricot").mkdir(parents=True, exist_ok=True)
    historical_row = _blank_row(
        header,
        "apricot",
        location="10-27-2025_samish_photo_sample",
        arm_color="orange",
    )
    current_row = _blank_row(
        header,
        "apricot",
        location="latest_location",
        overall_color="purple",
    )
    _seed_gallery_csv(gallery_csv_path, header, [historical_row, current_row])

    report = import_legacy_gallery_measurements(
        source_root=source_root,
        dry_run=False,
        measurements_root=measurements_root,
        gallery_csv_path=gallery_csv_path,
    )

    assert report.errors == []
    imported = [item for item in report.results if item.status in {"imported", "metadata_appended"}]
    assert len(imported) == 1

    destination_mfolder = Path(imported[0].destination_mfolder)
    assert destination_mfolder.exists()

    morph = json.loads((destination_mfolder / "morphometrics.json").read_text(encoding="utf-8"))
    detection = json.loads((destination_mfolder / "corrected_detection.json").read_text(encoding="utf-8"))
    assert morph["identity_type"] == "gallery"
    assert morph["identity_id"] == "apricot"
    assert morph["location"] == "10-27-2025_samish_photo_sample"
    assert morph["user_initials"] == "ABC"
    assert morph[LEGACY_PROVENANCE_KEY] == str(source_mfolder)
    assert detection["identity_type"] == "gallery"
    assert detection["identity_id"] == "apricot"
    assert detection["location"] == "10-27-2025_samish_photo_sample"
    assert detection[LEGACY_PROVENANCE_KEY] == str(source_mfolder)

    rows = read_rows(gallery_csv_path)
    latest = last_row_per_id(rows, "gallery_id")["apricot"]
    assert latest["location"] == "10-27-2025_samish_photo_sample"
    assert latest["overall_color"] == "purple"
    assert latest["arm_color"] == "orange"
    assert latest["morph_source_folder"] == str(destination_mfolder)
    assert latest["morph_num_arms"] == "5"
    assert latest["morph_area_mm2"] == "12.50"
    assert latest["morph_mean_arm_length_mm"] == "11.00"

    history_path = archive_root / "gallery" / "apricot" / "_metadata_history.csv"
    assert history_path.exists()
    assert str(destination_mfolder) in history_path.read_text(encoding="utf-8")

    discovered = {str(path) for path in list_mfolders(measurements_root, "gallery", "apricot")}
    assert str(destination_mfolder) in discovered

    archive_images = list(list_image_files("Gallery", "apricot"))
    assert len(archive_images) == 1
    assert archive_images[0].parent.name == "10_27_25"
    assert archive_images[0].name == "raw_frame.png"


def test_backfill_archive_images_is_idempotent_for_existing_import(tmp_path, monkeypatch):
    archive_root = tmp_path / "archive"
    monkeypatch.setenv("STARBOARD_ARCHIVE_DIR", str(archive_root))

    source_root = tmp_path / LEGACY_SOURCE_ROOT_NAME
    _write_bundle(source_root, alias="apricot")
    measurements_root = tmp_path / "measurements"

    gallery_csv_path, header = ap.metadata_csv_for("Gallery")
    (archive_root / "gallery" / "apricot").mkdir(parents=True, exist_ok=True)
    _seed_gallery_csv(
        gallery_csv_path,
        header,
        [
            _blank_row(header, "apricot", location="10-27-2025_samish_photo_sample"),
        ],
    )

    import_report = import_legacy_gallery_measurements(
        source_root=source_root,
        dry_run=False,
        measurements_root=measurements_root,
        gallery_csv_path=gallery_csv_path,
    )
    imported = [item for item in import_report.results if item.status in {"imported", "metadata_appended"}]
    destination_mfolder = Path(imported[0].destination_mfolder)

    archive_plan = build_archive_backfill_plan(
        source_root=source_root,
        measurements_root=measurements_root,
    )
    assert len(archive_plan) == 1
    assert archive_plan[0].gallery_id == "apricot"
    assert archive_plan[0].encounter_name == "10_27_25"
    assert archive_plan[0].archive_encounter_dir == archive_root / "gallery" / "apricot" / "10_27_25"
    assert archive_plan[0].imported_mfolder == destination_mfolder

    encounter_dir = archive_root / "gallery" / "apricot" / "10_27_25"
    for child in encounter_dir.iterdir():
        if child.is_file():
            child.unlink()
    invalidate_image_cache()
    assert list(list_image_files("Gallery", "apricot")) == []

    backfill_report = backfill_legacy_gallery_archive_images(
        source_root=source_root,
        dry_run=False,
        measurements_root=measurements_root,
    )
    assert backfill_report.errors == []
    backfilled = [item for item in backfill_report.results if item.status == "backfilled"]
    assert len(backfilled) == 1

    images_after_backfill = list(list_image_files("Gallery", "apricot"))
    assert len(images_after_backfill) == 1
    assert images_after_backfill[0].parent.name == "10_27_25"

    second_report = backfill_legacy_gallery_archive_images(
        source_root=source_root,
        dry_run=False,
        measurements_root=measurements_root,
    )
    already_archived = [item for item in second_report.results if item.status == "already_archived"]
    assert len(already_archived) == 1
    assert len(list(list_image_files("Gallery", "apricot"))) == 1
