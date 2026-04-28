from fastapi.testclient import TestClient

from mobile_portal.app.models.api import ImageDescriptor
from mobile_portal.app.models.megastar_api import MegaStarLookupCandidate, MegaStarLookupResponse
from mobile_portal.megastar_worker.main import create_app


def _upload_file_tuple():
    return {'file': ('selected.jpg', b'fake-image-bytes', 'image/jpeg')}


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


def test_megastar_worker_lookup_disabled_by_default():
    client = TestClient(create_app())
    r = client.post('/lookup', files=_upload_file_tuple())
    assert r.status_code == 503
    body = r.json()
    assert body['status'] == 'unavailable'
    assert body['availability_reason'] == 'feature_flag_disabled'


def test_megastar_worker_lookup_uses_real_service_contract(monkeypatch):
    from mobile_portal.megastar_worker import routes_lookup

    class EnabledCapability:
        enabled = True
        state = 'enabled'
        reason = None
        model_key = 'default_megastarid_v1'

    seen_max_candidates = []

    class StubService:
        def lookup_upload(self, *, filename, content, content_type=None, max_candidates=5):
            seen_max_candidates.append(max_candidates)
            return MegaStarLookupResponse(
                query_image_name=filename,
                status='ok',
                processing_ms=12,
                capability_state='enabled',
                availability_reason=None,
                candidates=[
                    MegaStarLookupCandidate(
                        rank=1,
                        entity_id='anchovy',
                        retrieval_score=0.91,
                        best_match_image=ImageDescriptor(
                            image_id='gallery:anchovy:0',
                            label='best.jpg',
                            encounter='enc-2026-04-03',
                            preview_url='/api/archive/media/gallery:anchovy:0/preview',
                            fullres_url='/api/archive/media/gallery:anchovy:0/full',
                        ),
                        best_match_label='best.jpg',
                        encounter='enc-2026-04-03',
                        encounter_date='2026-04-03',
                    )
                ],
            )

    monkeypatch.setattr(routes_lookup, 'capability_status', lambda: EnabledCapability())
    monkeypatch.setattr(routes_lookup, 'get_megastar_worker_lookup_service', lambda: StubService())
    client = TestClient(create_app())
    r = client.post('/lookup?max_candidates=7', files=_upload_file_tuple())
    assert r.status_code == 200
    body = r.json()
    assert body['status'] == 'ok'
    assert body['candidates'][0]['entity_id'] == 'anchovy'
    assert seen_max_candidates == [7]
