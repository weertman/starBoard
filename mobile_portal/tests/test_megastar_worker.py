from fastapi.testclient import TestClient

from mobile_portal.megastar_worker.main import create_app


def test_megastar_worker_health():
    client = TestClient(create_app())
    r = client.get('/health')
    assert r.status_code == 200
    body = r.json()
    assert body['status'] == 'ok'
    assert body['service'] == 'starboard-megastar-worker'


def test_megastar_worker_status_disabled_by_default(monkeypatch):
    monkeypatch.delenv('STARBOARD_MEGASTAR_WORKER_ENABLED', raising=False)
    client = TestClient(create_app())
    r = client.get('/status')
    assert r.status_code == 200
    body = r.json()
    assert body['enabled'] is False
    assert body['state'] == 'disabled'
    assert body['reason'] == 'feature_flag_disabled'
