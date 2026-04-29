from pathlib import Path
import io
import zipfile

from fastapi.testclient import TestClient

from star_browser.app.adapters.batch_upload_adapter import discover_batch_source
from star_browser.app.main import create_app
from star_browser.app.models.batch_upload_api import BatchUploadDiscoverRequest


def test_batch_upload_discover_request_model():
    req = BatchUploadDiscoverRequest(
        target_archive='gallery',
        discovery_mode='flat',
        id_prefix='',
        id_suffix='',
        import_source={'type': 'server_path', 'path': '/tmp/example'},
    )
    assert req.target_archive == 'gallery'


def test_discover_batch_source_flat_mode(tmp_path):
    base = tmp_path / 'flat'
    (base / 'anchovy').mkdir(parents=True)
    (base / 'anchovy' / 'a.jpg').write_bytes(b'x')
    rows = discover_batch_source(base, requested_mode='flat')
    assert len(rows) == 1
    assert rows[0].original_detected_id == 'anchovy'
    assert rows[0].image_count == 1


def test_batch_upload_discover_route(tmp_path):
    base = tmp_path / 'flat'
    (base / 'anchovy').mkdir(parents=True)
    (base / 'anchovy' / 'a.jpg').write_bytes(b'x')

    client = TestClient(create_app())
    r = client.post(
        '/api/batch-upload/discover',
        headers={'cf-access-authenticated-user-email': 'field@example.org'},
        json={
            'target_archive': 'gallery',
            'discovery_mode': 'flat',
            'id_prefix': '',
            'id_suffix': '',
            'import_source': {'type': 'server_path', 'path': str(base)},
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body['summary']['detected_ids'] == 1
    assert body['rows'][0]['original_detected_id'] == 'anchovy'


def test_batch_upload_preview_marks_existing_targets(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    (archive / 'gallery' / 'anchovy').mkdir(parents=True)
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(archive))

    base = tmp_path / 'flat'
    (base / 'anchovy').mkdir(parents=True)
    (base / 'anchovy' / 'a.jpg').write_bytes(b'x')

    client = TestClient(create_app())
    r = client.post(
        '/api/batch-upload/discover',
        headers={'cf-access-authenticated-user-email': 'field@example.org'},
        json={
            'target_archive': 'gallery',
            'discovery_mode': 'flat',
            'id_prefix': '',
            'id_suffix': '',
            'import_source': {'type': 'server_path', 'path': str(base)},
        },
    )
    body = r.json()
    assert body['rows'][0]['target_exists'] is True
    assert body['rows'][0]['action'] == 'append_existing'


def test_batch_upload_execute_route_writes_files_and_csv(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    (archive / 'gallery').mkdir(parents=True)
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(archive))

    base = tmp_path / 'flat'
    (base / 'anchovy').mkdir(parents=True)
    (base / 'anchovy' / 'a.jpg').write_bytes(b'x')

    client = TestClient(create_app())
    discover = client.post(
        '/api/batch-upload/discover',
        headers={'cf-access-authenticated-user-email': 'field@example.org'},
        json={
            'target_archive': 'gallery',
            'discovery_mode': 'flat',
            'id_prefix': '',
            'id_suffix': '',
            'flat_encounter_date': '2026-04-21',
            'import_source': {'type': 'server_path', 'path': str(base)},
        },
    )
    assert discover.status_code == 200
    plan = discover.json()
    row_id = plan['rows'][0]['row_id']

    execute = client.post(
        '/api/batch-upload/execute',
        headers={'cf-access-authenticated-user-email': 'field@example.org'},
        json={'plan_id': plan['plan_id'], 'accepted_row_ids': [row_id]},
    )
    assert execute.status_code == 200
    body = execute.json()
    assert body['status'] == 'ok'
    assert body['summary']['executed_rows'] == 1
    assert body['summary']['created_ids'] == 1
    written = archive / 'gallery' / 'anchovy' / '04_21_26' / 'a.jpg'
    assert written.exists()
    csv_path = archive / 'gallery' / 'gallery_metadata.csv'
    assert csv_path.exists()
    assert 'anchovy' in csv_path.read_text(encoding='utf-8-sig')


def test_batch_upload_execute_converts_olympus_orf_to_jpeg(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    (archive / 'gallery').mkdir(parents=True)
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(archive))

    def fake_convert(src, dest):
        dest.write_bytes(b'converted-jpeg')
        return dest

    monkeypatch.setattr('src.data.raw_conversion.convert_raw_to_jpeg', fake_convert)
    base = tmp_path / 'flat'
    (base / 'anchovy').mkdir(parents=True)
    (base / 'anchovy' / 'P1010001.ORF').write_bytes(b'raw')

    client = TestClient(create_app())
    discover = client.post(
        '/api/batch-upload/discover',
        headers={'cf-access-authenticated-user-email': 'field@example.org'},
        json={
            'target_archive': 'gallery',
            'discovery_mode': 'flat',
            'id_prefix': '',
            'id_suffix': '',
            'flat_encounter_date': '2026-04-21',
            'import_source': {'type': 'server_path', 'path': str(base)},
        },
    )
    assert discover.status_code == 200
    plan = discover.json()
    assert plan['summary']['total_images'] == 1

    execute = client.post(
        '/api/batch-upload/execute',
        headers={'cf-access-authenticated-user-email': 'field@example.org'},
        json={'plan_id': plan['plan_id'], 'accepted_row_ids': [plan['rows'][0]['row_id']]},
    )

    assert execute.status_code == 200
    body = execute.json()
    assert body['summary']['accepted_images'] == 1
    assert body['rows'][0]['archive_paths_written'] == ['anchovy/04_21_26/P1010001.jpg']
    assert (archive / 'gallery' / 'anchovy' / '04_21_26' / 'P1010001.jpg').read_bytes() == b'converted-jpeg'


def test_batch_upload_execute_rejects_unknown_plan(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    (archive / 'gallery').mkdir(parents=True)
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(archive))

    client = TestClient(create_app())
    execute = client.post(
        '/api/batch-upload/execute',
        headers={'cf-access-authenticated-user-email': 'field@example.org'},
        json={'plan_id': 'missing_plan', 'accepted_row_ids': ['row_001']},
    )
    assert execute.status_code == 404


def test_batch_upload_uploads_route_returns_token_and_discover_accepts_uploaded_bundle(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    (archive / 'gallery').mkdir(parents=True)
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(archive))

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, 'w') as zf:
        zf.writestr('anchovy/a.jpg', b'x')
    payload.seek(0)

    client = TestClient(create_app())
    upload = client.post(
        '/api/batch-upload/uploads',
        headers={'cf-access-authenticated-user-email': 'field@example.org'},
        files={'file': ('bundle.zip', payload.getvalue(), 'application/zip')},
    )
    assert upload.status_code == 200
    token = upload.json()['upload_token']

    discover = client.post(
        '/api/batch-upload/discover',
        headers={'cf-access-authenticated-user-email': 'field@example.org'},
        json={
            'target_archive': 'gallery',
            'discovery_mode': 'flat',
            'id_prefix': '',
            'id_suffix': '',
            'import_source': {'type': 'uploaded_bundle', 'upload_token': token},
        },
    )
    assert discover.status_code == 200
    body = discover.json()
    assert body['summary']['detected_ids'] == 1
    assert body['rows'][0]['original_detected_id'] == 'anchovy'
