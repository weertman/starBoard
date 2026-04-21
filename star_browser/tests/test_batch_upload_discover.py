from pathlib import Path

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
