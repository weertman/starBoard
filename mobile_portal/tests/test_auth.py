from fastapi.testclient import TestClient

from .conftest import build_test_app


def test_session_requires_auth_header(tmp_path, monkeypatch):
    app, _archive = build_test_app(tmp_path, monkeypatch)
    client = TestClient(app)
    r = client.get('/api/session')
    assert r.status_code == 401


def test_session_returns_authenticated_email(tmp_path, monkeypatch):
    app, _archive = build_test_app(tmp_path, monkeypatch)
    client = TestClient(app)
    r = client.get('/api/session', headers={'cf-access-authenticated-user-email': 'field@example.org'})
    assert r.status_code == 200
    assert r.json()['authenticated_email'] == 'field@example.org'
