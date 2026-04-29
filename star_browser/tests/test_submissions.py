import io
import json

from fastapi.testclient import TestClient
from PIL import Image

from star_browser.app.main import create_app


AUTH = {'cf-access-authenticated-user-email': 'field@example.org'}


def _image_bytes(color=(120, 30, 50)):
    buf = io.BytesIO()
    Image.new('RGB', (64, 64), color).save(buf, format='JPEG')
    return buf.getvalue()


def test_submission_rejects_missing_files(tmp_path, monkeypatch):
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(tmp_path / 'archive'))
    client = TestClient(create_app())
    payload = {
        'target_type': 'query',
        'target_mode': 'create',
        'target_id': 'q1',
        'encounter_date': '2026-04-01',
        'metadata': {},
    }

    response = client.post('/api/submissions', headers=AUTH, data={'payload': json.dumps(payload)})

    assert response.status_code in (400, 422)


def test_submission_rejects_invalid_target_id(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(archive))
    client = TestClient(create_app())
    payload = {
        'target_type': 'query',
        'target_mode': 'create',
        'target_id': '../bad',
        'encounter_date': '2026-04-01',
        'metadata': {},
    }
    files = [('files', ('capture.jpg', _image_bytes(), 'image/jpeg'))]

    response = client.post('/api/submissions', headers=AUTH, data={'payload': json.dumps(payload)}, files=files)

    assert response.status_code == 400


def test_submission_rejects_invalid_encounter_suffix(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(archive))
    client = TestClient(create_app())
    payload = {
        'target_type': 'query',
        'target_mode': 'create',
        'target_id': 'q1',
        'encounter_date': '2026-04-01',
        'encounter_suffix': '../escape',
        'metadata': {},
    }
    files = [('files', ('capture.jpg', _image_bytes(), 'image/jpeg'))]

    response = client.post('/api/submissions', headers=AUTH, data={'payload': json.dumps(payload)}, files=files)

    assert response.status_code == 400


def test_submission_accepts_query_create_and_writes_metadata(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(archive))
    client = TestClient(create_app())
    payload = {
        'target_type': 'query',
        'target_mode': 'create',
        'target_id': 'q1',
        'encounter_date': '2026-04-01',
        'metadata': {'location': 'dock', 'num_apparent_arms': '12'},
    }
    files = [('files', ('capture.jpg', _image_bytes(), 'image/jpeg'))]

    response = client.post('/api/submissions', headers=AUTH, data={'payload': json.dumps(payload)}, files=files)

    assert response.status_code == 200
    body = response.json()
    assert body['entity_type'] == 'query'
    assert body['entity_id'] == 'q1'
    assert (archive / 'queries' / 'q1').exists()
    metadata_csv = archive / 'queries' / 'queries_metadata.csv'
    assert metadata_csv.exists()
    metadata_text = metadata_csv.read_text(encoding='utf-8-sig')
    assert 'dock' in metadata_text
    assert '12' in metadata_text


def test_submission_accepts_gallery_append(tmp_path, monkeypatch):
    from src.data.archive_paths import metadata_csv_for
    from src.data.csv_io import append_row

    archive = tmp_path / 'archive'
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(archive))
    (archive / 'gallery' / 'anchovy' / '03_31_26').mkdir(parents=True, exist_ok=True)
    csv_path, header = metadata_csv_for('Gallery')
    append_row(csv_path, header, {'gallery_id': 'anchovy', 'location': 'dock'})
    client = TestClient(create_app())
    payload = {
        'target_type': 'gallery',
        'target_mode': 'append',
        'target_id': 'anchovy',
        'encounter_date': '2026-04-01',
        'metadata': {'location': 'dock', 'health_observation': 'healthy'},
    }
    files = [('files', ('capture.jpg', _image_bytes(), 'image/jpeg'))]

    response = client.post('/api/submissions', headers=AUTH, data={'payload': json.dumps(payload)}, files=files)

    assert response.status_code == 200
    assert (archive / 'gallery' / 'anchovy' / '04_01_26').exists()


def test_submission_converts_olympus_orf_to_jpeg(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(archive))

    def fake_convert(src, dest):
        dest.write_bytes(b'converted-jpeg')
        return dest

    monkeypatch.setattr('src.data.raw_conversion.convert_raw_to_jpeg', fake_convert)
    client = TestClient(create_app())
    payload = {
        'target_type': 'query',
        'target_mode': 'create',
        'target_id': 'raw_query',
        'encounter_date': '2026-04-01',
        'metadata': {'location': 'dock'},
    }
    files = [('files', ('P1010001.ORF', b'raw-orf-bytes', 'application/octet-stream'))]

    response = client.post('/api/submissions', headers=AUTH, data={'payload': json.dumps(payload)}, files=files)

    assert response.status_code == 200
    body = response.json()
    assert body['accepted_images'] == 1
    assert body['archive_paths_written'][0].endswith('/P1010001.jpg')
    assert (archive / 'queries' / 'raw_query' / '04_01_26' / 'P1010001.jpg').read_bytes() == b'converted-jpeg'
