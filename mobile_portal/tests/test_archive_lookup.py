from fastapi.testclient import TestClient

from src.data.csv_io import append_row
from src.data.archive_paths import metadata_csv_for

from .conftest import build_test_app, make_image

AUTH = {'cf-access-authenticated-user-email': 'field@example.org'}


def _seed_gallery(archive):
    make_image(archive / 'gallery' / 'feta' / '11_16_25' / 'DSC02125.JPG')
    make_image(archive / 'gallery' / 'feta' / '11_16_25' / 'DSC02126.JPG', color=(10, 20, 30))
    csv_path, header = metadata_csv_for('Gallery')
    append_row(csv_path, header, {'gallery_id': 'feta', 'location': 'dock'})


def test_gallery_lookup_not_found_returns_404(tmp_path, monkeypatch):
    app, _archive = build_test_app(tmp_path, monkeypatch)
    client = TestClient(app)
    r = client.get('/api/archive/entities/does-not-exist?entity_type=gallery', headers=AUTH)
    assert r.status_code == 404


def test_gallery_lookup_returns_image_window_for_real_id(tmp_path, monkeypatch):
    app, archive = build_test_app(tmp_path, monkeypatch)
    _seed_gallery(archive)
    client = TestClient(app)
    r = client.get('/api/archive/entities/feta?entity_type=gallery', headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body['entity_id'] == 'feta'
    assert body['metadata_summary']['location'] == 'dock'
    assert body['image_window']['count'] >= 1


def test_gallery_lookup_images_endpoint_supports_offset_limit(tmp_path, monkeypatch):
    app, archive = build_test_app(tmp_path, monkeypatch)
    _seed_gallery(archive)
    client = TestClient(app)
    r = client.get('/api/archive/entities/feta/images?entity_type=gallery&offset=0&limit=1', headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body['offset'] == 0
    assert body['count'] <= 1


def test_gallery_suggest_returns_matching_ids(tmp_path, monkeypatch):
    app, archive = build_test_app(tmp_path, monkeypatch)
    _seed_gallery(archive)
    client = TestClient(app)
    r = client.get('/api/archive/suggest?entity_type=gallery&query=fe', headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body['entity_type'] == 'gallery'
    assert 'feta' in body['items']
