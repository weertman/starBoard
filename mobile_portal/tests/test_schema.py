from fastapi.testclient import TestClient

from .conftest import build_test_app

AUTH = {'cf-access-authenticated-user-email': 'field@example.org'}


def test_metadata_schema_requires_auth(tmp_path, monkeypatch):
    app, _archive = build_test_app(tmp_path, monkeypatch)
    client = TestClient(app)
    r = client.get('/api/schema/metadata')
    assert r.status_code == 401


def test_metadata_schema_returns_fields(tmp_path, monkeypatch):
    app, _archive = build_test_app(tmp_path, monkeypatch)
    client = TestClient(app)
    r = client.get('/api/schema/metadata', headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert 'fields' in body
    assert len(body['fields']) > 0
    names = {f['name'] for f in body['fields']}
    assert 'location' in names
