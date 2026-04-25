import json
from fastapi.testclient import TestClient
from PIL import Image

from .conftest import build_test_app

AUTH = {'cf-access-authenticated-user-email': 'field@example.org'}


def _image_bytes(color=(120, 30, 50)):
    import io
    buf = io.BytesIO()
    Image.new('RGB', (64, 64), color).save(buf, format='JPEG')
    return buf.getvalue()


def test_empty_submission_rejected(tmp_path, monkeypatch):
    app, _archive = build_test_app(tmp_path, monkeypatch)
    client = TestClient(app)
    payload = {'target_type': 'query', 'target_mode': 'create', 'target_id': 'q1', 'encounter_date': '2026-04-01', 'metadata': {}}
    r = client.post('/api/submissions', headers=AUTH, data={'payload': json.dumps(payload)})
    assert r.status_code in (400, 422)


def test_valid_gallery_create_submission(tmp_path, monkeypatch):
    app, archive = build_test_app(tmp_path, monkeypatch)
    client = TestClient(app)
    payload = {'target_type': 'gallery', 'target_mode': 'create', 'target_id': 'new_gallery_star', 'encounter_date': '2026-04-01', 'metadata': {'location': 'dock'}}
    files = [('files', ('capture.jpg', _image_bytes(), 'image/jpeg'))]
    r = client.post('/api/submissions', headers=AUTH, data={'payload': json.dumps(payload)}, files=files)
    assert r.status_code == 200
    body = r.json()
    assert body['entity_type'] == 'gallery'
    assert (archive / 'gallery' / 'new_gallery_star' / '04_01_26').exists()


def test_valid_query_create_submission(tmp_path, monkeypatch):
    app, archive = build_test_app(tmp_path, monkeypatch)
    client = TestClient(app)
    payload = {'target_type': 'query', 'target_mode': 'create', 'target_id': 'q1', 'encounter_date': '2026-04-01', 'metadata': {'location': 'dock'}}
    files = [('files', ('capture.jpg', _image_bytes(), 'image/jpeg'))]
    r = client.post('/api/submissions', headers=AUTH, data={'payload': json.dumps(payload)}, files=files)
    assert r.status_code == 200
    body = r.json()
    assert body['entity_type'] == 'query'
    assert (archive / 'queries' / 'q1').exists()


def test_valid_gallery_append_submission(tmp_path, monkeypatch):
    from src.data.csv_io import append_row
    from src.data.archive_paths import metadata_csv_for

    app, archive = build_test_app(tmp_path, monkeypatch)
    (archive / 'gallery' / 'anchovy' / '03_31_26').mkdir(parents=True, exist_ok=True)
    csv_path, header = metadata_csv_for('Gallery')
    append_row(csv_path, header, {'gallery_id': 'anchovy', 'location': 'dock'})
    client = TestClient(app)
    payload = {'target_type': 'gallery', 'target_mode': 'append', 'target_id': 'anchovy', 'encounter_date': '2026-04-01', 'metadata': {'location': 'dock'}}
    files = [('files', ('capture.jpg', _image_bytes(), 'image/jpeg'))]
    r = client.post('/api/submissions', headers=AUTH, data={'payload': json.dumps(payload)}, files=files)
    assert r.status_code == 200
    assert (archive / 'gallery' / 'anchovy' / '04_01_26').exists()
