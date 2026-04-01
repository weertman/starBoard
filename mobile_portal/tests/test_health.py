from fastapi.testclient import TestClient

from .conftest import build_test_app


def test_health_returns_ok(tmp_path, monkeypatch):
    app, _archive = build_test_app(tmp_path, monkeypatch)
    client = TestClient(app)
    r = client.get('/api/health')
    assert r.status_code == 200
    body = r.json()
    assert body['status'] == 'ok'
    assert body['service'] == 'starboard-mobile-portal'
