from __future__ import annotations

from pathlib import Path

from src.data.ingest import place_images
from src.dl.megastar_queue import list_open_jobs
from src.dl.registry import DLRegistry


def test_place_images_enqueues_gallery_identity_with_final_archive_path(tmp_path, monkeypatch):
    monkeypatch.setenv("STARBOARD_ARCHIVE_DIR", str(tmp_path / "archive"))
    src = tmp_path / "source.jpg"
    src.write_bytes(b"jpg")
    target_root = tmp_path / "archive" / "gallery"

    report = place_images(target_root, "g-001", "04_29_26", [src])

    assert not report.errors
    jobs = list_open_jobs()
    assert len(jobs) == 1
    assert jobs[0].target == "Gallery"
    assert jobs[0].id_str == "g-001"
    assert str(report.ops[0].dest) in jobs[0].changed_paths_json
    assert DLRegistry.reload().pending_ids.gallery == ["g-001"]


def test_place_images_enqueues_queries_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("STARBOARD_ARCHIVE_DIR", str(tmp_path / "archive"))
    src = tmp_path / "source.jpg"
    src.write_bytes(b"jpg")
    target_root = tmp_path / "archive" / "queries"

    report = place_images(target_root, "q-001", "04_29_26", [src])

    assert not report.errors
    jobs = list_open_jobs()
    assert len(jobs) == 1
    assert jobs[0].target == "Queries"
    assert jobs[0].id_str == "q-001"
    assert DLRegistry.reload().pending_ids.queries == ["q-001"]


def test_place_images_does_not_enqueue_when_no_file_ops(tmp_path, monkeypatch):
    monkeypatch.setenv("STARBOARD_ARCHIVE_DIR", str(tmp_path / "archive"))

    report = place_images(tmp_path / "archive" / "gallery", "g-001", "04_29_26", [tmp_path / "missing.jpg"])

    assert report.errors
    assert list_open_jobs() == []
    registry = DLRegistry.reload()
    assert registry.pending_ids.gallery == []
    assert registry.pending_ids.queries == []


def test_place_images_skips_duplicate_image_bytes_in_same_encounter(tmp_path, monkeypatch):
    monkeypatch.setenv("STARBOARD_ARCHIVE_DIR", str(tmp_path / "archive"))
    source_a = tmp_path / "source-a.jpg"
    source_b = tmp_path / "source-b.jpg"
    source_a.write_bytes(b"same-image")
    source_b.write_bytes(b"same-image")
    target_root = tmp_path / "archive" / "queries"

    first = place_images(target_root, "q-001", "04_29_26", [source_a])
    second = place_images(target_root, "q-001", "04_29_26", [source_b])

    assert len(first.ops) == 1
    assert second.ops == []
    assert sorted(p.name for p in (target_root / "q-001" / "04_29_26").glob("*.jpg")) == ["source-a.jpg"]


def test_place_images_enqueues_converted_raw_final_jpeg_path(tmp_path, monkeypatch):
    monkeypatch.setenv("STARBOARD_ARCHIVE_DIR", str(tmp_path / "archive"))
    raw_file = tmp_path / "P1010001.ORF"
    raw_file.write_bytes(b"raw")

    def fake_convert(src: Path, dest: Path):
        dest.write_bytes(b"jpg")
        return dest

    monkeypatch.setattr("src.data.raw_conversion.convert_raw_to_jpeg", fake_convert)

    report = place_images(tmp_path / "archive" / "gallery", "g-001", "04_29_26", [raw_file])

    assert report.ops[0].dest.name == "P1010001.jpg"
    jobs = list_open_jobs()
    assert str(report.ops[0].dest) in jobs[0].changed_paths_json
    assert str(raw_file) not in jobs[0].changed_paths_json
