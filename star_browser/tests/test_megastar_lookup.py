from fastapi.testclient import TestClient

from mobile_portal.app.config import MegaStarCapabilityStatus
from mobile_portal.app.models.megastar_api import MegaStarLookupResponse
from star_browser.app.main import create_app

AUTH = {'cf-access-authenticated-user-email': 'field@example.org'}


def _upload_file_tuple(content=b'fake-image'):
    return {'file': ('query.jpg', content, 'image/jpeg')}


def test_star_browser_megastar_status_reports_capability(monkeypatch):
    import star_browser.app.routes.megastar_lookup as route_mod

    monkeypatch.setattr(
        route_mod,
        'get_megastar_capability_status',
        lambda: MegaStarCapabilityStatus(
            enabled=True,
            state='enabled',
            backend='local',
            reason=None,
            model_key='default_megastarid_v1',
            artifact_dir=None,
        ),
    )

    client = TestClient(create_app())
    r = client.get('/api/megastar/status', headers=AUTH)

    assert r.status_code == 200
    body = r.json()
    assert body['enabled'] is True
    assert body['state'] == 'enabled'
    assert body['backend'] == 'local'
    assert body['model_key'] == 'default_megastarid_v1'


def test_star_browser_megastar_lookup_dispatches_to_backend(monkeypatch):
    import star_browser.app.routes.megastar_lookup as route_mod

    monkeypatch.setattr(
        route_mod,
        'get_megastar_capability_status',
        lambda: MegaStarCapabilityStatus(
            enabled=True,
            state='enabled',
            backend='local',
            reason=None,
            model_key='default_megastarid_v1',
            artifact_dir=None,
        ),
    )

    seen = {}

    class StubBackend:
        def lookup_upload(self, *, filename, content, content_type=None, max_candidates=5):
            seen['filename'] = filename
            seen['content'] = content
            seen['content_type'] = content_type
            seen['max_candidates'] = max_candidates
            return MegaStarLookupResponse(
                query_image_name=filename,
                status='ok',
                processing_ms=12,
                capability_state='enabled',
                candidates=[],
            )

    monkeypatch.setattr(route_mod, 'get_megastar_lookup_backend', lambda: StubBackend())

    client = TestClient(create_app())
    r = client.post('/api/megastar/lookup?max_candidates=7', headers=AUTH, files=_upload_file_tuple(b'jpeg-bytes'))

    assert r.status_code == 200
    body = r.json()
    assert body['query_image_name'] == 'query.jpg'
    assert body['status'] == 'ok'
    assert seen == {
        'filename': 'query.jpg',
        'content': b'jpeg-bytes',
        'content_type': 'image/jpeg',
        'max_candidates': 7,
    }


def test_star_browser_session_advertises_megastar_lookup_capability(monkeypatch):
    import star_browser.app.routes.session as session_mod

    monkeypatch.setattr(
        session_mod,
        'get_megastar_capability_status',
        lambda: MegaStarCapabilityStatus(
            enabled=False,
            state='disabled',
            backend='local',
            reason='feature_flag_disabled',
            model_key=None,
            artifact_dir=None,
        ),
    )

    client = TestClient(create_app())
    r = client.get('/api/session', headers=AUTH)

    assert r.status_code == 200
    body = r.json()
    assert body['capabilities']['megastar_lookup'] is True
    assert body['megastar_lookup']['state'] == 'disabled'
    assert body['megastar_lookup']['reason'] == 'feature_flag_disabled'
