from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from star_browser.app.main import create_app
from star_browser.app.models.gallery_api import GalleryEntityResponse


def test_gallery_entity_response_model_fields():
    obj = GalleryEntityResponse(
        entity_id='anchovy',
        metadata_summary={},
        metadata_rows=[],
        timeline=[],
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


def test_id_review_options_route_returns_filterable_ids(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    friday = archive / 'queries' / 'query_friday_001' / '04_01_26'
    cattle = archive / 'queries' / 'query_cattle_002' / '04_02_26'
    friday.mkdir(parents=True)
    cattle.mkdir(parents=True)
    Image.new('RGB', (20, 20), color=(0, 0, 255)).save(friday / 'IMG_001.jpg')
    Image.new('RGB', (20, 20), color=(0, 255, 0)).save(cattle / 'IMG_001.jpg')
    (archive / 'queries' / 'queries_metadata.csv').write_text(
        'query_id,location,sex\n'
        'query_friday_001,Friday Harbor,female\n'
        'query_cattle_002,Cattle Point,male\n'
    )
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(archive))

    client = TestClient(create_app())
    r = client.get('/api/id-review/options/query', headers={'cf-access-authenticated-user-email': 'field@example.org'})
    assert r.status_code == 200
    body = r.json()
    assert body['archive_type'] == 'query'
    options = {item['entity_id']: item for item in body['options']}
    assert options['query_friday_001']['location'] == 'Friday Harbor'
    assert options['query_friday_001']['last_observation_date'] == '2026-04-01'
    assert options['query_friday_001']['metadata']['sex'] == 'female'
    assert options['query_friday_001']['label'] == 'query_friday_001 — Friday Harbor — 2026-04-01'
    assert options['query_cattle_002']['location'] == 'Cattle Point'


def test_id_review_query_entity_route_returns_seeded_query(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    q = archive / 'queries' / 'query_review_001' / '03_15_24'
    q.mkdir(parents=True)
    img = q / 'IMG_001.jpg'
    Image.new('RGB', (20, 20), color=(0, 0, 255)).save(img)
    (archive / 'queries' / 'queries_metadata.csv').write_text('query_id,location\nquery_review_001,Friday Harbor\n')
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(archive))

    client = TestClient(create_app())
    r = client.get('/api/id-review/entities/query/query_review_001', headers={'cf-access-authenticated-user-email': 'field@example.org'})
    assert r.status_code == 200
    body = r.json()
    assert body['entity_id'] == 'query_review_001'
    assert body['archive_type'] == 'query'
    assert body['metadata_summary']['location'] == 'Friday Harbor'
    assert body['metadata_rows'][0]['source'] == 'queries_metadata.csv'
    assert body['metadata_rows'][0]['values']['location'] == 'Friday Harbor'
    assert body['timeline'][0]['encounter'] == '03_15_24'
    assert body['timeline'][0]['image_count'] == 1
    assert body['timeline'][0]['image_labels'] == ['IMG_001.jpg']
    assert len(body['images']) == 1
    assert body['images'][0]['image_id'].startswith('query:query_review_001:')
    assert body['images'][0]['preview_url'].startswith('/api/id-review/media/query:query_review_001:')


def test_id_review_entity_orders_timeline_and_images_newest_first(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    older = archive / 'queries' / 'query_review_001' / '04_01_26'
    newer = archive / 'queries' / 'query_review_001' / '04_03_26'
    undated = archive / 'queries' / 'query_review_001' / 'unknown_encounter'
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    undated.mkdir(parents=True)
    Image.new('RGB', (20, 20), color=(0, 0, 255)).save(older / 'older.jpg')
    Image.new('RGB', (20, 20), color=(0, 255, 0)).save(newer / 'newer.jpg')
    Image.new('RGB', (20, 20), color=(255, 0, 0)).save(undated / 'undated.jpg')
    (archive / 'queries' / 'queries_metadata.csv').write_text('query_id,location\nquery_review_001,Friday Harbor\n')
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(archive))

    client = TestClient(create_app())
    r = client.get('/api/id-review/entities/query/query_review_001', headers={'cf-access-authenticated-user-email': 'field@example.org'})

    assert r.status_code == 200
    body = r.json()
    assert [event['date'] for event in body['timeline']] == ['2026-04-03', '2026-04-01', '']
    assert [image['label'] for image in body['images']] == ['newer.jpg', 'older.jpg', 'undated.jpg']


def test_id_review_media_routes_return_query_image_bytes(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    q = archive / 'queries' / 'query_review_001' / '03_15_24'
    q.mkdir(parents=True)
    img = q / 'IMG_001.jpg'
    Image.new('RGB', (20, 20), color=(0, 0, 255)).save(img)
    (archive / 'queries' / 'queries_metadata.csv').write_text('query_id,location\nquery_review_001,Friday Harbor\n')
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(archive))

    client = TestClient(create_app())
    entity = client.get('/api/id-review/entities/query/query_review_001', headers={'cf-access-authenticated-user-email': 'field@example.org'})
    image_id = entity.json()['images'][0]['image_id']

    full = client.get(f'/api/id-review/media/{image_id}/full', headers={'cf-access-authenticated-user-email': 'field@example.org'})
    preview = client.get(f'/api/id-review/media/{image_id}/preview', headers={'cf-access-authenticated-user-email': 'field@example.org'})
    assert full.status_code == 200
    assert preview.status_code == 200
    assert len(full.content) > 0
    assert len(preview.content) > 0
