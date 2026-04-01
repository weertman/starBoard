from fastapi.testclient import TestClient

from src.data.csv_io import append_row
from src.data.archive_paths import metadata_csv_for

from .conftest import build_test_app, make_image

AUTH = {'cf-access-authenticated-user-email': 'field@example.org'}


def _seed_gallery(archive):
    make_image(archive / 'gallery' / 'feta' / '11_16_25' / 'DSC02125.JPG')
    csv_path, header = metadata_csv_for('Gallery')
    append_row(csv_path, header, {'gallery_id': 'feta', 'location': 'dock'})


def test_full_media_route_returns_image_bytes_for_real_id(tmp_path, monkeypatch):
    app, archive = build_test_app(tmp_path, monkeypatch)
    _seed_gallery(archive)
    client = TestClient(app)
    lookup = client.get('/api/archive/entities/feta?entity_type=gallery', headers=AUTH).json()
    image_id = lookup['image_window']['items'][0]['image_id']
    r = client.get(f'/api/archive/media/{image_id}/full', headers=AUTH)
    assert r.status_code == 200
    assert r.headers['content-type'].startswith('image/')


def test_preview_media_route_returns_image_bytes_for_real_id(tmp_path, monkeypatch):
    app, archive = build_test_app(tmp_path, monkeypatch)
    _seed_gallery(archive)
    client = TestClient(app)
    lookup = client.get('/api/archive/entities/feta?entity_type=gallery', headers=AUTH).json()
    image_id = lookup['image_window']['items'][0]['image_id']
    r = client.get(f'/api/archive/media/{image_id}/preview', headers=AUTH)
    assert r.status_code == 200
    assert r.headers['content-type'].startswith('image/')
