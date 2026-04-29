import io
import json
from pathlib import Path
from urllib.error import URLError

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency in some test envs
    torch = None

from mobile_portal.app.adapters.megastar_artifact_loader import MegaStarArtifactAvailability, load_megastar_artifact_availability
from mobile_portal.app.adapters.megastar_model_adapter import MegaStarModelAdapter
from mobile_portal.app.adapters.megastar_query_preprocess import MegaStarQueryPreprocessor
from mobile_portal.app.adapters.megastar_result_resolver import MegaStarArtifactMatch, MegaStarResultResolver
from mobile_portal.app.config import get_settings
from mobile_portal.app.models.megastar_api import MegaStarLookupCandidate, MegaStarLookupResponse
from mobile_portal.app.services.megastar_lookup_service import MegaStarLookupService, prepare_upload_content_for_lookup

from .conftest import build_test_app, make_image

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


def _upload_file_tuple(content: bytes | None = None, *, filename: str = 'selected.jpg', content_type: str = 'image/jpeg'):
    return {'file': (filename, content or b'fake-image-bytes', content_type)}



def _make_image_bytes(color=(200, 80, 40)) -> bytes:
    image = Image.new('RGB', (120, 90), color)
    buf = io.BytesIO()
    image.save(buf, format='JPEG')
    return buf.getvalue()


def _seed_available_megastar_assets(tmp_path, archive_dir, *, pending_gallery=None, pending_queries=None):
    model_key = 'test_megastar_model'
    checkpoint_path = tmp_path / 'star_identification' / 'checkpoints' / 'default' / 'best.pth'
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b'checkpoint')

    root = archive_dir / '_dl_precompute' / model_key
    (root / 'embeddings').mkdir(parents=True, exist_ok=True)
    (root / 'similarity').mkdir(parents=True, exist_ok=True)
    (root / 'embeddings' / 'gallery_image_embeddings.npz').write_bytes(b'npz-placeholder')
    (root / 'embeddings' / 'gallery_image_paths.json').write_text(
        json.dumps({'specimen-1': ['C:\\cache\\gallery\\specimen-1\\frame001.png']})
    )
    (root / 'similarity' / 'gallery_image_index.json').write_text(
        json.dumps([{'id': 'specimen-1', 'local_idx': 0, 'path': 'C:\\cache\\gallery\\specimen-1\\frame001.png'}])
    )
    (root / 'similarity' / 'metadata.json').write_text(
        json.dumps({'use_tta': True, 'image_size': 384, 'embedding_dim': 4})
    )

    registry_path = archive_dir / '_dl_precompute' / '_dl_registry.json'
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            {
                'active_model': model_key,
                'models': {
                    model_key: {
                        'precomputed': True,
                        'checkpoint_path': 'star_identification\\checkpoints\\default\\best.pth',
                    }
                },
                'pending_ids': {'gallery': pending_gallery or [], 'queries': pending_queries or []},
            }
        )
    )
    return model_key


def _seed_searchable_megastar_assets(tmp_path, archive_dir):
    model_key = 'test_megastar_model'
    checkpoint_path = tmp_path / 'star_identification' / 'checkpoints' / 'default' / 'best.pth'
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b'checkpoint')

    make_image(archive_dir / 'gallery' / 'alpha' / 'enc-2026-04-03' / 'alpha1.jpg', color=(220, 60, 60))
    make_image(archive_dir / 'gallery' / 'alpha' / 'enc-2026-04-03' / 'alpha2.jpg', color=(210, 50, 50))
    make_image(archive_dir / 'gallery' / 'beta' / 'enc-2026-04-04' / 'beta1.jpg', color=(60, 220, 60))
    make_image(archive_dir / 'gallery' / 'delta' / 'enc-2026-04-06' / 'delta1.jpg', color=(200, 220, 60))
    make_image(archive_dir / 'gallery' / 'gamma' / 'enc-2026-04-05' / 'gamma1.jpg', color=(60, 60, 220))

    root = archive_dir / '_dl_precompute' / model_key
    embeddings_dir = root / 'embeddings'
    similarity_dir = root / 'similarity'
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    similarity_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        embeddings_dir / 'gallery_image_embeddings.npz',
        alpha=np.array([[1.0, 0.0, 0.0, 0.0], [0.97, 0.03, 0.0, 0.0]], dtype=np.float32),
        beta=np.array([[0.6, 0.8, 0.0, 0.0]], dtype=np.float32),
        delta=np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
        gamma=np.array([[-1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    (embeddings_dir / 'gallery_image_paths.json').write_text(
        json.dumps(
            {
                'alpha': [
                    'C:\\cache\\gallery\\alpha\\alpha1.png',
                    'C:\\cache\\gallery\\alpha\\alpha2.png',
                ],
                'beta': ['C:\\cache\\gallery\\beta\\beta1.png'],
                'delta': ['C:\\cache\\gallery\\delta\\delta1.png'],
                'gamma': ['C:\\cache\\gallery\\gamma\\gamma1.png'],
            }
        )
    )
    (similarity_dir / 'gallery_image_index.json').write_text(
        json.dumps(
            [
                {'id': 'alpha', 'local_idx': 0, 'path': 'C:\\cache\\gallery\\alpha\\alpha1.png'},
                {'id': 'alpha', 'local_idx': 1, 'path': 'C:\\cache\\gallery\\alpha\\alpha2.png'},
                {'id': 'beta', 'local_idx': 0, 'path': 'C:\\cache\\gallery\\beta\\beta1.png'},
                {'id': 'delta', 'local_idx': 0, 'path': 'C:\\cache\\gallery\\delta\\delta1.png'},
                {'id': 'gamma', 'local_idx': 0, 'path': 'C:\\cache\\gallery\\gamma\\gamma1.png'},
            ]
        )
    )
    (similarity_dir / 'metadata.json').write_text(
        json.dumps({'use_tta': False, 'image_size': 384, 'embedding_dim': 4})
    )

    registry_path = archive_dir / '_dl_precompute' / '_dl_registry.json'
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            {
                'active_model': model_key,
                'models': {
                    model_key: {
                        'precomputed': True,
                        'checkpoint_path': 'star_identification/checkpoints/default/best.pth',
                    }
                },
                'pending_ids': {'gallery': [], 'queries': []},
            }
        )
    )
    return model_key


class _FakeYOLOPreprocessor:
    def process_image(self, image):
        return image.crop((10, 5, 110, 55))


class _FakeModel:
    def eval(self):
        return self

    def __call__(self, batch, return_normalized=True):
        assert torch is not None
        flat = batch.reshape(batch.shape[0], -1)
        features = torch.stack(
            (
                flat.mean(dim=1),
                flat.max(dim=1).values,
                flat.min(dim=1).values,
                flat[:, 0],
            ),
            dim=1,
        )
        if return_normalized:
            return torch.nn.functional.normalize(features, p=2, dim=1)
        return features


class _FakeBackend:
    load_calls = 0

    def __init__(self):
        assert torch is not None
        self._model = None
        self._device = torch.device('cpu')
        self._loaded_model_path = None

    def is_model_loaded(self):
        return self._model is not None

    def get_loaded_model_path(self):
        return self._loaded_model_path

    def load_model(self, checkpoint_path):
        type(self).load_calls += 1
        self._model = _FakeModel()
        self._loaded_model_path = checkpoint_path
        return True

    def get_image_size(self):
        return 384


class _ConstantEmbeddingModel:
    def eval(self):
        return self

    def __call__(self, batch, return_normalized=True):
        assert torch is not None
        embedding = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=batch.device)
        return embedding


class _ConstantEmbeddingBackend:
    def __init__(self):
        assert torch is not None
        self._model = None
        self._device = torch.device('cpu')
        self._loaded_model_path = None

    def is_model_loaded(self):
        return self._model is not None

    def get_loaded_model_path(self):
        return self._loaded_model_path

    def load_model(self, checkpoint_path):
        self._model = _ConstantEmbeddingModel()
        self._loaded_model_path = checkpoint_path
        return True

    def get_image_size(self):
        return 384


def _archive_file_set(root: Path) -> set[str]:
    return {str(path.relative_to(root)) for path in root.rglob('*') if path.is_file()}


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


def test_megastar_lookup_rejects_non_image_upload(tmp_path, monkeypatch):
    monkeypatch.setenv('STARBOARD_MOBILE_MEGASTAR_ENABLED', '1')
    monkeypatch.setattr('mobile_portal.app.adapters.megastar_artifact_loader.DL_AVAILABLE', True)
    app, archive = build_test_app(tmp_path, monkeypatch)
    _seed_searchable_megastar_assets(tmp_path, archive)
    client = TestClient(app)

    r = client.post(
        '/api/megastar/lookup',
        headers=AUTH,
        files=_upload_file_tuple(b'plain-text', filename='selected.txt', content_type='text/plain'),
    )

    assert r.status_code == 400
    assert r.json()['detail'] == 'unsupported_media_type'


def test_megastar_lookup_prepares_olympus_orf_bytes_as_jpeg(monkeypatch):
    calls = []

    def fake_convert(payload, suffix='.orf'):
        calls.append((payload, suffix))
        return b'jpeg-bytes'

    monkeypatch.setattr('mobile_portal.app.services.megastar_lookup_service.convert_raw_bytes_to_jpeg_bytes', fake_convert)

    converted = prepare_upload_content_for_lookup(
        filename='P1010001.ORF',
        content=b'raw-orf-bytes',
        content_type='application/octet-stream',
    )

    assert converted == b'jpeg-bytes'
    assert calls == [(b'raw-orf-bytes', '.orf')]


def test_megastar_artifact_loader_reports_enabled_with_portable_checkpoint_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv('STARBOARD_MOBILE_MEGASTAR_ENABLED', '1')
    monkeypatch.setattr('mobile_portal.app.adapters.megastar_artifact_loader.DL_AVAILABLE', True)
    app, archive = build_test_app(tmp_path, monkeypatch)
    _seed_available_megastar_assets(tmp_path, archive)

    settings = get_settings()
    availability = load_megastar_artifact_availability(settings)

    assert app is not None
    assert availability.enabled is True
    assert availability.reason is None
    assert availability.model_key == 'test_megastar_model'
    assert availability.checkpoint_path is not None
    assert availability.checkpoint_path.exists()
    assert availability.checkpoint_path.name == 'best.pth'
    assert availability.use_tta is True
    assert availability.image_size == 384
    assert availability.embedding_dim == 4


def test_megastar_artifact_loader_fails_closed_when_registry_stale(tmp_path, monkeypatch):
    monkeypatch.setenv('STARBOARD_MOBILE_MEGASTAR_ENABLED', '1')
    monkeypatch.setattr('mobile_portal.app.adapters.megastar_artifact_loader.DL_AVAILABLE', True)
    build_test_app(tmp_path, monkeypatch)
    archive = tmp_path / 'archive'
    _seed_available_megastar_assets(tmp_path, archive, pending_gallery=['stale-id'])

    availability = load_megastar_artifact_availability(get_settings())

    assert availability.enabled is False
    assert availability.state == 'unavailable'
    assert availability.reason == 'stale_artifacts'


def test_query_preprocessor_matches_cache_then_tensor_contract():
    if torch is None:
        return
    image = Image.new('RGB', (1200, 300), color=(120, 60, 30))
    preprocessor = MegaStarQueryPreprocessor(image_size=384, yolo_preprocessor=_FakeYOLOPreprocessor())

    result = preprocessor.preprocess_pil_image(image)

    assert result.original_image_size == (1200, 300)
    assert result.cache_image_size == (100, 50)
    assert tuple(result.image_tensor.shape) == (3, 384, 384)
    assert result.image_tensor.dtype == torch.float32
    assert torch.isfinite(result.image_tensor).all()


def test_query_preprocessor_rejects_invalid_upload_bytes():
    if torch is None:
        return
    preprocessor = MegaStarQueryPreprocessor(image_size=384)

    try:
        preprocessor.preprocess_upload_bytes(b'not-a-real-image')
    except Exception as exc:
        assert str(exc) == 'image_decode_failed'
    else:
        raise AssertionError('expected image decode failure')


def test_megastar_model_adapter_loads_once_and_returns_normalized_embedding(tmp_path):
    if torch is None:
        return
    checkpoint_path = tmp_path / 'best.pth'
    checkpoint_path.write_bytes(b'checkpoint')
    availability = MegaStarArtifactAvailability(
        enabled=True,
        state='enabled',
        model_key='test_megastar_model',
        checkpoint_path=checkpoint_path,
        image_size=384,
        use_tta=True,
    )
    _FakeBackend.load_calls = 0
    adapter = MegaStarModelAdapter(availability=availability, backend_factory=_FakeBackend)
    tensor = torch.ones((3, 384, 384), dtype=torch.float32)

    embedding1 = adapter.extract_embedding(tensor)
    embedding2 = adapter.extract_embedding(tensor)

    assert _FakeBackend.load_calls == 1
    assert embedding1.shape == (4,)
    assert np.isclose(np.linalg.norm(embedding1), 1.0)
    assert np.allclose(embedding1, embedding2)
    assert str(adapter._backend._device) == 'cpu'


def test_megastar_result_resolver_maps_windows_artifact_path_to_portal_descriptor(tmp_path, monkeypatch):
    build_test_app(tmp_path, monkeypatch)
    archive = tmp_path / 'archive'
    make_image(archive / 'gallery' / 'specimen-1' / 'enc-2026-04-03' / 'frame001.jpg')
    resolver = MegaStarResultResolver()

    descriptor = resolver.resolve_best_match(
        MegaStarArtifactMatch(
            entity_id='specimen-1',
            local_idx=0,
            artifact_path='C:\\Users\\name\\star_identification\\precompute_cache\\gallery\\specimen-1\\frame001.png',
        )
    )

    assert descriptor['image_id'] == 'gallery:specimen-1:0'
    assert descriptor['label'] == 'frame001.jpg'
    assert descriptor['encounter'] == 'enc-2026-04-03'
    assert descriptor['preview_url'].endswith('/api/archive/media/gallery:specimen-1:0/preview')
    assert descriptor['fullres_url'].endswith('/api/archive/media/gallery:specimen-1:0/full')


def test_megastar_lookup_service_returns_ranked_gallery_ids_from_image_hits(tmp_path, monkeypatch):
    monkeypatch.setenv('STARBOARD_MOBILE_MEGASTAR_ENABLED', '1')
    monkeypatch.setattr('mobile_portal.app.adapters.megastar_artifact_loader.DL_AVAILABLE', True)
    build_test_app(tmp_path, monkeypatch)
    archive = tmp_path / 'archive'
    _seed_searchable_megastar_assets(tmp_path, archive)

    service = MegaStarLookupService(settings=get_settings())
    availability = load_megastar_artifact_availability(get_settings())
    hits = service.search_image_hits(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), availability=availability)
    candidates = service.aggregate_id_candidates(hits)

    assert [hit.entity_id for hit in hits[:3]] == ['alpha', 'alpha', 'beta']
    assert [candidate.entity_id for candidate in candidates] == ['alpha', 'beta', 'delta', 'gamma']
    assert candidates[0].best_hit.local_idx == 0
    assert candidates[0].encounter == 'enc-2026-04-03'
    assert candidates[0].encounter_date == '2026-04-03'
    assert candidates[0].retrieval_score > candidates[1].retrieval_score


def test_megastar_lookup_service_reports_empty_when_no_candidates_clear_threshold(tmp_path, monkeypatch):
    monkeypatch.setenv('STARBOARD_MOBILE_MEGASTAR_ENABLED', '1')
    monkeypatch.setattr('mobile_portal.app.adapters.megastar_artifact_loader.DL_AVAILABLE', True)
    build_test_app(tmp_path, monkeypatch)
    archive = tmp_path / 'archive'
    _seed_searchable_megastar_assets(tmp_path, archive)

    service = MegaStarLookupService(settings=get_settings())
    availability = load_megastar_artifact_availability(get_settings())
    hits = service.search_image_hits(np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32), availability=availability)
    candidates = service.aggregate_id_candidates(hits)

    assert [candidate.entity_id for candidate in candidates] == ['alpha', 'beta', 'delta', 'gamma']
    assert all(candidate.retrieval_score == 0.0 for candidate in candidates)
    assert service._result_status(candidates) == 'weak'


def test_megastar_lookup_service_reports_weak_when_top_result_is_low_confidence(tmp_path, monkeypatch):
    monkeypatch.setenv('STARBOARD_MOBILE_MEGASTAR_ENABLED', '1')
    monkeypatch.setattr('mobile_portal.app.adapters.megastar_artifact_loader.DL_AVAILABLE', True)
    build_test_app(tmp_path, monkeypatch)
    archive = tmp_path / 'archive'
    _seed_searchable_megastar_assets(tmp_path, archive)

    service = MegaStarLookupService(settings=get_settings())
    availability = load_megastar_artifact_availability(get_settings())
    hits = service.search_image_hits(np.array([0.3, 0.9539392, 0.0, 0.0], dtype=np.float32), availability=availability)
    candidates = service.aggregate_id_candidates(hits)

    assert candidates[0].entity_id == 'delta'
    assert candidates[1].entity_id == 'beta'
    assert candidates[0].retrieval_score - candidates[1].retrieval_score < 0.03
    assert service._result_status(candidates) == 'weak'


def test_megastar_lookup_route_returns_real_results_without_archive_writes(tmp_path, monkeypatch):
    if torch is None:
        return
    monkeypatch.setenv('STARBOARD_MOBILE_MEGASTAR_ENABLED', '1')
    monkeypatch.setattr('mobile_portal.app.adapters.megastar_artifact_loader.DL_AVAILABLE', True)
    app, archive = build_test_app(tmp_path, monkeypatch)
    _seed_searchable_megastar_assets(tmp_path, archive)

    import mobile_portal.app.routes.megastar_lookup as route_mod

    monkeypatch.setattr(
        route_mod,
        'get_megastar_lookup_backend',
        lambda: type(
            'LocalBackend',
            (),
            {
                'lookup_upload': lambda self, *, filename, content, content_type=None, max_candidates=5: MegaStarLookupService(
                    settings=get_settings(),
                    backend_factory=_ConstantEmbeddingBackend,
                ).lookup_upload(
                    filename=filename,
                    content=content,
                    content_type=content_type,
                    max_candidates=max_candidates,
                )
            },
        )(),
    )

    before_files = _archive_file_set(archive)
    client = TestClient(app)
    r = client.post('/api/megastar/lookup', headers=AUTH, files=_upload_file_tuple(_make_image_bytes()))
    after_files = _archive_file_set(archive)

    assert r.status_code == 200
    body = r.json()
    assert body['status'] == 'ok'
    assert [item['entity_id'] for item in body['candidates']] == ['alpha', 'beta', 'delta', 'gamma']
    assert body['candidates'][0]['best_match_image']['image_id'] == 'gallery:alpha:0'
    assert body['candidates'][0]['encounter_date'] == '2026-04-03'
    assert body['processing_ms'] >= 0
    assert before_files == after_files


def test_megastar_lookup_route_can_return_empty_state(tmp_path, monkeypatch):
    monkeypatch.setenv('STARBOARD_MOBILE_MEGASTAR_ENABLED', '1')
    monkeypatch.setattr('mobile_portal.app.adapters.megastar_artifact_loader.DL_AVAILABLE', True)
    app, archive = build_test_app(tmp_path, monkeypatch)
    _seed_searchable_megastar_assets(tmp_path, archive)

    import mobile_portal.app.routes.megastar_lookup as route_mod

    class _StubService:
        def lookup_upload(self, *, filename, content, content_type=None, max_candidates=5):
            return MegaStarLookupResponse(
                query_image_name=filename,
                status='empty',
                candidates=[],
                processing_ms=7,
                capability_state='enabled',
            )

    monkeypatch.setattr(route_mod, 'get_megastar_lookup_backend', lambda: _StubService())
    client = TestClient(app)
    r = client.post('/api/megastar/lookup', headers=AUTH, files=_upload_file_tuple(_make_image_bytes()))

    assert r.status_code == 200
    body = r.json()
    assert body['status'] == 'empty'
    assert body['candidates'] == []


def test_megastar_lookup_route_can_return_weak_state(tmp_path, monkeypatch):
    monkeypatch.setenv('STARBOARD_MOBILE_MEGASTAR_ENABLED', '1')
    monkeypatch.setattr('mobile_portal.app.adapters.megastar_artifact_loader.DL_AVAILABLE', True)
    app, archive = build_test_app(tmp_path, monkeypatch)
    _seed_searchable_megastar_assets(tmp_path, archive)

    import mobile_portal.app.routes.megastar_lookup as route_mod

    class _StubService:
        def lookup_upload(self, *, filename, content, content_type=None, max_candidates=5):
            return MegaStarLookupResponse(
                query_image_name=filename,
                status='weak',
                candidates=[
                    MegaStarLookupCandidate(
                        rank=1,
                        entity_id='beta',
                        retrieval_score=0.31,
                        best_match_image={
                            'image_id': 'gallery:beta:0',
                            'label': 'beta1.jpg',
                            'encounter': 'enc-2026-04-04',
                            'fullres_url': '/api/archive/media/gallery:beta:0/full',
                            'preview_url': '/api/archive/media/gallery:beta:0/preview',
                            'width': 120,
                            'height': 90,
                        },
                        best_match_label='beta1.jpg',
                        encounter='enc-2026-04-04',
                        encounter_date='2026-04-04',
                    )
                ],
                processing_ms=9,
                capability_state='enabled',
            )

    monkeypatch.setattr(route_mod, 'get_megastar_lookup_backend', lambda: _StubService())
    client = TestClient(app)
    r = client.post('/api/megastar/lookup', headers=AUTH, files=_upload_file_tuple(_make_image_bytes()))

    assert r.status_code == 200
    body = r.json()
    assert body['status'] == 'weak'
    assert body['candidates'][0]['entity_id'] == 'beta'



def test_megastar_lookup_route_dispatches_to_worker_backend_when_selected(tmp_path, monkeypatch):
    monkeypatch.setenv('STARBOARD_MOBILE_MEGASTAR_ENABLED', '1')
    monkeypatch.setenv('STARBOARD_MOBILE_MEGASTAR_BACKEND', 'worker')
    monkeypatch.setenv('STARBOARD_MOBILE_MEGASTAR_WORKER_URL', 'http://megastar-worker.test')
    app, _archive = build_test_app(tmp_path, monkeypatch)
    client = TestClient(app)
    seen_lookup_urls = []

    def _fake_urlopen(req, timeout=0):
        if req.full_url.endswith('/status'):
            return _FakeUrlopenResponse(
                {
                    'enabled': True,
                    'state': 'enabled',
                    'reason': None,
                    'model_key': 'worker-model-v1',
                }
            )
        if '/lookup?' in req.full_url:
            seen_lookup_urls.append(req.full_url)
            return _FakeUrlopenResponse(
                {
                    'query_image_name': 'selected.jpg',
                    'status': 'ok',
                    'candidates': [
                        {
                            'rank': 1,
                            'entity_type': 'gallery',
                            'entity_id': 'anchovy',
                            'retrieval_score': 0.91,
                            'best_match_image': {
                                'image_id': 'gallery:anchovy:0',
                                'label': 'best.jpg',
                                'encounter': 'enc-2026-04-03',
                                'preview_url': '/api/archive/media/gallery:anchovy:0/preview',
                                'fullres_url': '/api/archive/media/gallery:anchovy:0/full',
                                'width': 120,
                                'height': 90,
                            },
                            'best_match_label': 'best.jpg',
                            'encounter': 'enc-2026-04-03',
                            'encounter_date': '2026-04-03',
                        }
                    ],
                    'processing_ms': 12,
                    'capability_state': 'enabled',
                    'availability_reason': None,
                }
            )
        raise AssertionError(f'unexpected worker URL: {req.full_url}')

    monkeypatch.setattr('mobile_portal.app.services.megastar_worker_client.request.urlopen', _fake_urlopen)

    r = client.post('/api/megastar/lookup?max_candidates=7', headers=AUTH, files=_upload_file_tuple(_make_image_bytes()))

    assert r.status_code == 200
    body = r.json()
    assert body['status'] == 'ok'
    assert body['candidates'][0]['entity_id'] == 'anchovy'
    assert body['capability_state'] == 'enabled'
    assert seen_lookup_urls == ['http://megastar-worker.test/lookup?max_candidates=7']



def test_megastar_lookup_route_returns_unavailable_when_worker_backend_unreachable(tmp_path, monkeypatch):
    monkeypatch.setenv('STARBOARD_MOBILE_MEGASTAR_ENABLED', '1')
    monkeypatch.setenv('STARBOARD_MOBILE_MEGASTAR_BACKEND', 'worker')
    app, _archive = build_test_app(tmp_path, monkeypatch)
    client = TestClient(app)

    def _fail_urlopen(req, timeout=0):
        raise URLError('connection refused')

    monkeypatch.setattr('mobile_portal.app.services.megastar_worker_client.request.urlopen', _fail_urlopen)

    r = client.post('/api/megastar/lookup', headers=AUTH, files=_upload_file_tuple(_make_image_bytes()))

    assert r.status_code == 503
    body = r.json()
    assert body['status'] == 'unavailable'
    assert body['capability_state'] == 'unavailable'
    assert body['availability_reason'] == 'worker_unreachable'
