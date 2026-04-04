import json
from urllib.error import URLError

from fastapi.testclient import TestClient

from .conftest import build_test_app

AUTH = {'cf-access-authenticated-user-email': 'field@example.org'}


class _FakeUrlopenResponse:
    def __init__(self, payload, *, status=200):
        self.payload = json.dumps(payload).encode('utf-8')
        self.status = status

    def read(self):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _seed_megastar_assets(archive_dir, *, pending_gallery=None, pending_queries=None):
    model_key = 'test_megastar_model'
    root = archive_dir / '_dl_precompute' / model_key
    checkpoint_path = archive_dir.parent / 'megastar-test-checkpoint.pth'
    checkpoint_path.write_bytes(b'checkpoint')

    asset_contents = {
        'embeddings/gallery_image_embeddings.npz': 'placeholder',
        'embeddings/gallery_image_paths.json': '{}',
        'similarity/gallery_image_index.json': '[]',
        'similarity/metadata.json': json.dumps({'use_tta': True, 'image_size': 384, 'embedding_dim': 512}),
    }
    for rel_path, content in asset_contents.items():
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    registry_path = archive_dir / '_dl_precompute' / '_dl_registry.json'
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            {
                'active_model': model_key,
                'models': {model_key: {'precomputed': True, 'checkpoint_path': str(checkpoint_path)}},
                'pending_ids': {'gallery': pending_gallery or [], 'queries': pending_queries or []},
            }
        )
    )


def test_session_requires_auth_header(tmp_path, monkeypatch):
    app, _archive = build_test_app(tmp_path, monkeypatch)
    client = TestClient(app)
    r = client.get('/api/session')
    assert r.status_code == 401


def test_session_returns_authenticated_email(tmp_path, monkeypatch):
    app, _archive = build_test_app(tmp_path, monkeypatch)
    client = TestClient(app)
    r = client.get('/api/session', headers=AUTH)
    assert r.status_code == 200
    assert r.json()['authenticated_email'] == 'field@example.org'


def test_session_reports_megastar_capability_disabled_when_unconfigured(tmp_path, monkeypatch):
    monkeypatch.setenv('STARBOARD_MOBILE_MEGASTAR_ENABLED', '1')
    app, _archive = build_test_app(tmp_path, monkeypatch)
    client = TestClient(app)

    r = client.get('/api/session', headers=AUTH)

    assert r.status_code == 200
    assert r.json()['capabilities']['megastar_lookup'] is False
    assert r.json()['capabilities']['lookup'] is True


def test_session_reports_megastar_capability_enabled_when_assets_are_ready(tmp_path, monkeypatch):
    monkeypatch.setenv('STARBOARD_MOBILE_MEGASTAR_ENABLED', '1')
    monkeypatch.setattr('mobile_portal.app.adapters.megastar_artifact_loader.DL_AVAILABLE', True)
    app, archive = build_test_app(tmp_path, monkeypatch)
    _seed_megastar_assets(archive)
    client = TestClient(app)

    r = client.get('/api/session', headers=AUTH)

    assert r.status_code == 200
    assert r.json()['capabilities']['megastar_lookup'] is True
    assert r.json()['megastar_lookup']['backend'] == 'local'


def test_session_reports_megastar_capability_disabled_when_registry_is_stale(tmp_path, monkeypatch):
    monkeypatch.setenv('STARBOARD_MOBILE_MEGASTAR_ENABLED', '1')
    monkeypatch.setattr('mobile_portal.app.adapters.megastar_artifact_loader.DL_AVAILABLE', True)
    app, archive = build_test_app(tmp_path, monkeypatch)
    _seed_megastar_assets(archive, pending_gallery=['stale-id'])
    client = TestClient(app)

    r = client.get('/api/session', headers=AUTH)

    assert r.status_code == 200
    assert r.json()['capabilities']['megastar_lookup'] is False



def test_session_reports_worker_backend_capability_when_worker_selected(tmp_path, monkeypatch):
    monkeypatch.setenv('STARBOARD_MOBILE_MEGASTAR_ENABLED', '1')
    monkeypatch.setenv('STARBOARD_MOBILE_MEGASTAR_BACKEND', 'worker')
    monkeypatch.setenv('STARBOARD_MOBILE_MEGASTAR_WORKER_URL', 'http://megastar-worker.test')
    app, _archive = build_test_app(tmp_path, monkeypatch)
    client = TestClient(app)

    def _fake_urlopen(req, timeout=0):
        assert req.full_url == 'http://megastar-worker.test/status'
        return _FakeUrlopenResponse(
            {
                'enabled': True,
                'state': 'enabled',
                'reason': None,
                'model_key': 'worker-model-v1',
            }
        )

    monkeypatch.setattr('mobile_portal.app.services.megastar_worker_client.request.urlopen', _fake_urlopen)

    r = client.get('/api/session', headers=AUTH)

    assert r.status_code == 200
    body = r.json()
    assert body['capabilities']['megastar_lookup'] is True
    assert body['megastar_lookup']['backend'] == 'worker'
    assert body['megastar_lookup']['state'] == 'enabled'
    assert body['megastar_lookup']['model_key'] == 'worker-model-v1'



def test_session_reports_worker_backend_unavailable_when_worker_absent(tmp_path, monkeypatch):
    monkeypatch.setenv('STARBOARD_MOBILE_MEGASTAR_ENABLED', '1')
    monkeypatch.setenv('STARBOARD_MOBILE_MEGASTAR_BACKEND', 'worker')
    app, _archive = build_test_app(tmp_path, monkeypatch)
    client = TestClient(app)

    def _fail_urlopen(req, timeout=0):
        raise URLError('connection refused')

    monkeypatch.setattr('mobile_portal.app.services.megastar_worker_client.request.urlopen', _fail_urlopen)

    r = client.get('/api/session', headers=AUTH)

    assert r.status_code == 200
    body = r.json()
    assert body['capabilities']['megastar_lookup'] is False
    assert body['megastar_lookup']['backend'] == 'worker'
    assert body['megastar_lookup']['state'] == 'unavailable'
    assert body['megastar_lookup']['reason'] == 'worker_unreachable'
