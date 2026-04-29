from pathlib import Path

from src.data.ingest import discover_ids_and_images, place_images
from src.data.image_index import list_image_files


def test_discovery_counts_olympus_orf_as_importable_image(tmp_path):
    source = tmp_path / 'source'
    (source / 'anchovy').mkdir(parents=True)
    (source / 'anchovy' / 'P1010001.ORF').write_bytes(b'raw')

    discovered = discover_ids_and_images(source)

    assert len(discovered) == 1
    assert discovered[0][0] == 'anchovy'
    assert [p.name for p in discovered[0][1]] == ['P1010001.ORF']


def test_place_images_converts_orf_to_archive_jpeg_and_hides_raw_from_image_index(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    raw_file = source / 'P1010001.ORF'
    raw_file.write_bytes(b'raw-orf-bytes')
    archive = tmp_path / 'archive'
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(archive))
    target_root = archive / 'gallery'

    calls = []

    def fake_convert(src: Path, dest: Path):
        calls.append((src, dest))
        dest.write_bytes(b'converted-jpeg')
        return dest

    monkeypatch.setattr('src.data.raw_conversion.convert_raw_to_jpeg', fake_convert)

    report = place_images(target_root, 'anchovy', '04_29_26', [raw_file])

    assert not report.errors
    assert len(report.ops) == 1
    assert report.ops[0].src == raw_file
    assert report.ops[0].dest == target_root / 'anchovy' / '04_29_26' / 'P1010001.jpg'
    assert report.ops[0].converted is True
    assert report.ops[0].action == 'converted'
    assert report.ops[0].dest.read_bytes() == b'converted-jpeg'
    assert calls == [(raw_file, report.ops[0].dest)]
    assert raw_file.exists(), 'copy-mode ORF import must not delete the collaborator raw original'
    assert list_image_files('Gallery', 'anchovy') == (report.ops[0].dest,)
