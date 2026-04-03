import json

from fastapi.testclient import TestClient

from .conftest import build_test_app

AUTH = {'cf-access-authenticated-user-email': 'field@example.org'}


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


def test_session_reports_megastar_capability_disabled_when_registry_is_stale(tmp_path, monkeypatch):
    monkeypatch.setenv('STARBOARD_MOBILE_MEGASTAR_ENABLED', '1')
    monkeypatch.setattr('mobile_portal.app.adapters.megastar_artifact_loader.DL_AVAILABLE', True)
    app, archive = build_test_app(tmp_path, monkeypatch)
    _seed_megastar_assets(archive, pending_gallery=['stale-id'])
    client = TestClient(app)

    r = client.get('/api/session', headers=AUTH)

    assert r.status_code == 200
    assert r.json()['capabilities']['megastar_lookup'] is False
