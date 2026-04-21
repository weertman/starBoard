from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from star_browser.app.main import create_app
from star_browser.app.models.gallery_api import GalleryEntityResponse


def test_gallery_entity_response_model_fields():
    obj = GalleryEntityResponse(
        entity_id='anchovy',
        metadata_summary={},
        encounters=[],
        images=[],
    )
    assert obj.entity_id == 'anchovy'


def test_gallery_entity_route_returns_seeded_gallery(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    g = archive / 'gallery' / 'anchovy' / '03_15_24'
    g.mkdir(parents=True)
    img = g / 'IMG_001.jpg'
    Image.new('RGB', (20, 20), color=(255, 0, 0)).save(img)
    (archive / 'gallery' / 'gallery_metadata.csv').write_text('gallery_id,location\nanchovy,Friday Harbor\n')
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(archive))

    client = TestClient(create_app())
    r = client.get('/api/gallery/entities/anchovy', headers={'cf-access-authenticated-user-email': 'field@example.org'})
    assert r.status_code == 200
    body = r.json()
    assert body['entity_id'] == 'anchovy'
    assert len(body['images']) == 1


def test_gallery_media_routes_return_image_bytes(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    g = archive / 'gallery' / 'anchovy' / '03_15_24'
    g.mkdir(parents=True)
    img = g / 'IMG_001.jpg'
    Image.new('RGB', (20, 20), color=(255, 0, 0)).save(img)
    (archive / 'gallery' / 'gallery_metadata.csv').write_text('gallery_id,location\nanchovy,Friday Harbor\n')
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(archive))

    client = TestClient(create_app())
    entity = client.get('/api/gallery/entities/anchovy', headers={'cf-access-authenticated-user-email': 'field@example.org'})
    image_id = entity.json()['images'][0]['image_id']

    full = client.get(f'/api/gallery/media/{image_id}/full', headers={'cf-access-authenticated-user-email': 'field@example.org'})
    preview = client.get(f'/api/gallery/media/{image_id}/preview', headers={'cf-access-authenticated-user-email': 'field@example.org'})
    assert full.status_code == 200
    assert preview.status_code == 200
    assert len(full.content) > 0
    assert len(preview.content) > 0
