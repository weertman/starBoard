import json

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

from .conftest import build_test_app, make_image

AUTH={'cf-a...il': 'field@example.org'}


def _upload_file_tuple():
    return {'file': ('selected.jpg', b'fake-image-bytes', 'image/jpeg')}


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
