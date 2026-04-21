from fastapi.testclient import TestClient

from star_browser.app.main import create_app


def test_health_route_exists():
    client = TestClient(create_app())
    r = client.get('/api/health')
    assert r.status_code == 200
    body = r.json()
    assert body['status'] == 'ok'
    assert body['service'] == 'star-browser'
