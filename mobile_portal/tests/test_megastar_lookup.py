from fastapi.testclient import TestClient

from .conftest import build_test_app

AUTH = {'cf-access-authenticated-user-email': 'field@example.org'}


def _upload_file_tuple():
    return {'file': ('selected.jpg', b'fake-image-bytes', 'image/jpeg')}


def test_megastar_lookup_requires_auth(tmp_path, monkeypatch):
    monkeypatch.setenv('STARBOARD_MOBILE_MEGASTAR_ENABLED', '1')
    app, _archive = build_test_app(tmp_path, monkeypatch)
    client = TestClient(app)

    r = client.post('/api/megastar/lookup', files=_upload_file_tuple())

    assert r.status_code == 401


def test_megastar_lookup_returns_controlled_unavailable_response_when_assets_missing(tmp_path, monkeypatch):
    monkeypatch.setenv('STARBOARD_MOBILE_MEGASTAR_ENABLED', '1')
    app, _archive = build_test_app(tmp_path, monkeypatch)
    client = TestClient(app)

    r = client.post('/api/megastar/lookup', headers=AUTH, files=_upload_file_tuple())

    assert r.status_code == 503
    body = r.json()
    assert body['status'] == 'unavailable'
    assert body['query_image_name'] == 'selected.jpg'
    assert body['capability_state'] == 'unavailable'
    assert body['availability_reason'] == 'registry_missing'
    assert body['candidates'] == []
    assert body['processing_ms'] >= 0
