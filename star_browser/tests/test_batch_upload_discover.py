from pathlib import Path
import csv
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
        import_source={'type': 'uploaded_bundle', 'upload_token': 'upload_example'},
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


def _upload_folder(client, root_name, files):
    upload_files = [
        ('files', (f'{root_name}/{relative}', data, 'image/jpeg'))
        for relative, data in files
    ]
    response = client.post(
        '/api/batch-upload/folder-uploads',
        headers={'cf-access-authenticated-user-email': 'field@example.org'},
        files=upload_files,
    )
    assert response.status_code == 200
    return response.json()['upload_token']


def test_batch_upload_discover_route():
    client = TestClient(create_app())
    token = _upload_folder(client, 'flat', [('anchovy/a.jpg', b'x')])
    r = client.post(
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
    assert r.status_code == 200
    body = r.json()
    assert body['summary']['detected_ids'] == 1
    assert body['rows'][0]['original_detected_id'] == 'anchovy'


def test_batch_upload_preview_marks_existing_targets(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    (archive / 'gallery' / 'anchovy').mkdir(parents=True)
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(archive))

    client = TestClient(create_app())
    token = _upload_folder(client, 'flat', [('anchovy/a.jpg', b'x')])
    r = client.post(
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
    body = r.json()
    assert body['rows'][0]['target_exists'] is True
    assert body['rows'][0]['action'] == 'append_existing'


def test_batch_upload_execute_route_writes_files_and_csv(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    (archive / 'gallery').mkdir(parents=True)
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(archive))

    client = TestClient(create_app())
    token = _upload_folder(client, 'flat', [('anchovy/a.jpg', b'x')])
    discover = client.post(
        '/api/batch-upload/discover',
        headers={'cf-access-authenticated-user-email': 'field@example.org'},
        json={
            'target_archive': 'gallery',
            'discovery_mode': 'flat',
            'id_prefix': '',
            'id_suffix': '',
            'flat_encounter_date': '2026-04-21',
            'import_source': {'type': 'uploaded_bundle', 'upload_token': token},
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

    client = TestClient(create_app())
    token = _upload_folder(client, 'flat', [('anchovy/P1010001.ORF', b'raw')])
    discover = client.post(
        '/api/batch-upload/discover',
        headers={'cf-access-authenticated-user-email': 'field@example.org'},
        json={
            'target_archive': 'gallery',
            'discovery_mode': 'flat',
            'id_prefix': '',
            'id_suffix': '',
            'flat_encounter_date': '2026-04-21',
            'import_source': {'type': 'uploaded_bundle', 'upload_token': token},
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


def test_batch_upload_folder_upload_preserves_browser_relative_paths(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    (archive / 'gallery').mkdir(parents=True)
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(archive))

    client = TestClient(create_app())
    upload = client.post(
        '/api/batch-upload/folder-uploads',
        headers={'cf-access-authenticated-user-email': 'field@example.org'},
        files=[
            ('files', ('trip_upload/anchovy/a.jpg', b'a', 'image/jpeg')),
            ('files', ('trip_upload/anchovy/b.jpg', b'b', 'image/jpeg')),
        ],
    )

    assert upload.status_code == 200
    body = upload.json()
    assert body['file_count'] == 2
    assert body['root_entries'] == ['trip_upload']

    discover = client.post(
        '/api/batch-upload/discover',
        headers={'cf-access-authenticated-user-email': 'field@example.org'},
        json={
            'target_archive': 'gallery',
            'discovery_mode': 'flat',
            'id_prefix': '',
            'id_suffix': '',
            'import_source': {'type': 'uploaded_bundle', 'upload_token': body['upload_token']},
        },
    )
    assert discover.status_code == 200
    preview = discover.json()
    assert preview['rows'][0]['original_detected_id'] == 'anchovy'


def test_auto_discovery_supports_single_id_root_with_encounters():
    client = TestClient(create_app())
    token = _upload_folder(client, 'ursa_minor', [('04_21_26_dock/a.jpg', b'x')])
    response = client.post(
        '/api/batch-upload/discover',
        headers={'cf-access-authenticated-user-email': 'field@example.org'},
        json={
            'target_archive': 'gallery',
            'discovery_mode': 'auto',
            'id_prefix': '',
            'id_suffix': '',
            'import_source': {'type': 'uploaded_bundle', 'upload_token': token},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body['resolved_discovery_mode'] == 'single_id'
    assert body['summary']['detected_rows'] == 1
    assert body['summary']['detected_ids'] == 1
    row = body['rows'][0]
    assert row['original_detected_id'] == 'ursa_minor'
    assert row['encounter_folder_name'] == '04_21_26_dock'
    assert row['encounter_date'] == '2026-04-21'
    assert row['encounter_suffix'] == 'dock'


def test_uploaded_zip_unwraps_single_container_directory(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    (archive / 'gallery').mkdir(parents=True)
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(archive))

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, 'w') as zf:
        zf.writestr('trip_upload/anchovy/a.jpg', b'x')
    payload.seek(0)

    client = TestClient(create_app())
    upload = client.post(
        '/api/batch-upload/uploads',
        headers={'cf-access-authenticated-user-email': 'field@example.org'},
        files={'file': ('bundle.zip', payload.getvalue(), 'application/zip')},
    )
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
    assert body['rows'][0]['original_detected_id'] == 'anchovy'
    assert body['rows'][0]['source_ref'] == 'anchovy'


def test_encounter_aware_modes_auto_detect_depth_like_desktop(tmp_path):
    base = tmp_path / 'grouped'
    (base / 'adults' / 'ursa_minor' / '04_21_26_dock').mkdir(parents=True)
    (base / 'adults' / 'ursa_minor' / '04_21_26_dock' / 'a.jpg').write_bytes(b'x')

    rows = discover_batch_source(base, requested_mode='encounters')

    assert len(rows) == 1
    assert rows[0].group_name == 'adults'
    assert rows[0].original_detected_id == 'ursa_minor'
    assert rows[0].encounter_date == '2026-04-21'


def test_discover_summary_counts_unique_new_and_existing_ids(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    (archive / 'gallery').mkdir(parents=True)
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(archive))

    client = TestClient(create_app())
    token = _upload_folder(client, 'ids', [('ursa_minor/04_21_26_dock/a.jpg', b'x'), ('ursa_minor/04_22_26_dock/b.jpg', b'x')])
    response = client.post(
        '/api/batch-upload/discover',
        headers={'cf-access-authenticated-user-email': 'field@example.org'},
        json={
            'target_archive': 'gallery',
            'discovery_mode': 'auto',
            'id_prefix': '',
            'id_suffix': '',
            'import_source': {'type': 'uploaded_bundle', 'upload_token': token},
        },
    )

    body = response.json()
    assert body['summary']['detected_rows'] == 2
    assert body['summary']['detected_ids'] == 1
    assert body['summary']['new_ids'] == 1
    assert body['summary']['existing_ids'] == 0


def test_discover_rejects_missing_sources(tmp_path):
    client = TestClient(create_app())

    missing_token = client.post(
        '/api/batch-upload/discover',
        headers={'cf-access-authenticated-user-email': 'field@example.org'},
        json={
            'target_archive': 'gallery',
            'discovery_mode': 'flat',
            'id_prefix': '',
            'id_suffix': '',
            'import_source': {'type': 'uploaded_bundle', 'upload_token': 'upload_missing'},
        },
    )
    assert missing_token.status_code == 404
    assert missing_token.json()['detail'] == 'upload_token_not_found'


def test_server_path_discover_source_is_not_exposed():
    client = TestClient(create_app())
    preview = client.post(
        '/api/batch-upload/server-path/preview',
        headers={'cf-access-authenticated-user-email': 'field@example.org'},
        json={'path': '/tmp', 'discovery_mode': 'auto'},
    )
    assert preview.status_code == 404
    discover = client.post(
        '/api/batch-upload/discover',
        headers={'cf-access-authenticated-user-email': 'field@example.org'},
        json={
            'target_archive': 'gallery',
            'discovery_mode': 'flat',
            'id_prefix': '',
            'id_suffix': '',
            'import_source': {'type': 'server_path', 'path': '/tmp'},
        },
    )
    assert discover.status_code == 422


def test_browser_batch_upload_records_gallery_metadata_history(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    (archive / 'gallery').mkdir(parents=True)
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(archive))

    client = TestClient(create_app())
    token = _upload_folder(client, 'flat', [('anchovy/a.jpg', b'x')])
    discover = client.post(
        '/api/batch-upload/discover',
        headers={'cf-access-authenticated-user-email': 'field@example.org'},
        json={
            'target_archive': 'gallery',
            'discovery_mode': 'flat',
            'id_prefix': '',
            'id_suffix': '',
            'flat_encounter_date': '2026-04-21',
            'batch_location': {'location': 'Dock', 'latitude': '48.5', 'longitude': '-123.1'},
            'import_source': {'type': 'uploaded_bundle', 'upload_token': token},
        },
    )
    plan = discover.json()

    execute = client.post(
        '/api/batch-upload/execute',
        headers={'cf-access-authenticated-user-email': 'field@example.org'},
        json={'plan_id': plan['plan_id'], 'accepted_row_ids': [plan['rows'][0]['row_id']]},
    )

    assert execute.status_code == 200
    history_path = archive / 'gallery' / 'anchovy' / '_metadata_history.csv'
    with history_path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    assert any(row['action'] == 'bulk_update' and row['source'] == 'batch_upload' for row in rows)
    assert any(row['field_name'] == 'location' and row['new_value'] == 'Dock' for row in rows)
