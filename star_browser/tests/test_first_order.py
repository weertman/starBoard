from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from star_browser.app.main import create_app


def test_first_order_search_route_returns_ranked_candidate(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    gallery = archive / 'gallery'
    queries = archive / 'queries'
    gallery.mkdir(parents=True)
    queries.mkdir(parents=True)
    (gallery / 'gallery_metadata.csv').write_text('gallery_id,location\nmedia_anchovy,Friday Harbor\n', encoding='utf-8-sig')
    (queries / 'queries_metadata.csv').write_text('query_id,location\nquery_001,Friday Harbor\n', encoding='utf-8-sig')
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(archive))

    client = TestClient(create_app())
    r = client.post(
        '/api/first-order/search',
        headers={'cf-access-authenticated-user-email': 'field@example.org'},
        json={'query_id': 'query_001', 'top_k': 5, 'preset': 'text'},
    )
    assert r.status_code == 200
    body = r.json()
    assert body['query_id'] == 'query_001'
    assert body['preset'] == 'text'
    assert len(body['candidates']) >= 1
    assert body['candidates'][0]['entity_id'] == 'media_anchovy'


def test_first_order_gallery_filter_options_and_search_filters_candidates(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    gallery = archive / 'gallery'
    queries = archive / 'queries'
    gallery.mkdir(parents=True)
    queries.mkdir(parents=True)
    (gallery / 'gallery_metadata.csv').write_text(
        'gallery_id,location,arm_color,arm_thickness,short_arm_code,sex,tip_to_tip_size_cm,last_modified_utc\n'
        'media_anchovy,Friday Harbor,orange,thin,small(7),female,21.4,2026-01-01T00:00:00Z\n'
        'media_cattle,Cattle Point,purple,thick,,male,39.8,2026-01-02T00:00:00Z\n',
        encoding='utf-8-sig',
    )
    (queries / 'queries_metadata.csv').write_text('query_id,location,arm_color\nquery_001,Friday Harbor,orange\n', encoding='utf-8-sig')
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(archive))

    client = TestClient(create_app())
    headers = {'cf-access-authenticated-user-email': 'field@example.org'}

    options = client.get('/api/first-order/gallery-filters', headers=headers)
    assert options.status_code == 200
    body = options.json()
    by_field = {item['field']: item for item in body['fields']}
    assert sorted(by_field) == ['arm_color', 'arm_thickness', 'location', 'short_arm_code']
    assert by_field['location']['values'] == ['Cattle Point', 'Friday Harbor']
    assert by_field['arm_color']['values'] == ['orange', 'purple']
    assert by_field['arm_thickness']['values'] == ['thick', 'thin']
    assert by_field['short_arm_code']['values'] == ['small(7)']
    assert 'sex' not in by_field
    assert 'tip_to_tip_size_cm' not in by_field
    assert 'last_modified_utc' not in by_field

    r = client.post(
        '/api/first-order/search',
        headers=headers,
        json={'query_id': 'query_001', 'top_k': 10, 'preset': 'all', 'gallery_filters': {'location': 'Cattle Point'}},
    )
    assert r.status_code == 200
    result = r.json()
    assert [candidate['entity_id'] for candidate in result['candidates']] == ['media_cattle']


def test_first_order_query_options_match_desktop_selector_order_and_state(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    gallery = archive / 'gallery'
    queries = archive / 'queries'
    gallery.mkdir(parents=True)
    queries.mkdir(parents=True)
    (gallery / 'gallery_metadata.csv').write_text('gallery_id,location\nmedia_anchovy,Friday Harbor\n', encoding='utf-8-sig')
    (queries / 'queries_metadata.csv').write_text(
        'query_id,madreporite_visibility,anus_visibility,postural_visibility,location,notes\n'
        'matched,3,1,4,Friday Harbor,already matched note\n'
        'pinned,2,,,Eagle Point,large selected-query metadata note\n'
        'attempted,,,,Cattle Point,attempted note\n'
        'silent,,,,Hidden Cove,hidden note\n',
        encoding='utf-8-sig',
    )
    for query_id, encounter in {
        'matched': '01_04_26_a',
        'pinned': '01_02_26_a',
        'attempted': '01_03_26_a',
        'silent': '01_01_26_a',
    }.items():
        (queries / query_id / encounter).mkdir(parents=True)
    (queries / 'pinned' / '_pins_first_order.json').write_text('{"pinned": ["media_anchovy"]}', encoding='utf-8')
    (queries / 'attempted' / '_second_order_labels.csv').write_text(
        'query_id,gallery_id,verdict,notes,updated_utc\nattempted,media_anchovy,no,,2026-01-01T00:00:00Z\n',
        encoding='utf-8-sig',
    )
    (queries / 'matched' / '_second_order_labels.csv').write_text(
        'query_id,gallery_id,verdict,notes,updated_utc\nmatched,media_anchovy,yes,,2026-01-01T00:00:00Z\n',
        encoding='utf-8-sig',
    )
    (queries / 'silent' / '_SILENT.flag').write_text('{"reason": "test"}', encoding='utf-8')
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(archive))

    client = TestClient(create_app())
    r = client.get('/api/first-order/queries', headers={'cf-access-authenticated-user-email': 'field@example.org'})

    assert r.status_code == 200
    body = r.json()
    assert [item['query_id'] for item in body['queries']] == ['attempted', 'pinned', 'matched']
    assert [item['state'] for item in body['queries']] == ['attempted', 'pinned', 'matched']
    assert body['queries'][1]['last_observation_date'] == '2026-01-02'
    assert body['queries'][1]['last_location'] == 'Eagle Point'
    assert body['queries'][1]['easy_match_score'] > body['queries'][0]['easy_match_score']
    assert body['queries'][1]['quality']['madreporite_visibility'] == 2 / 3
    assert body['queries'][1]['quality']['anus_visibility'] is None
    assert body['queries'][1]['metadata']['notes'] == 'large selected-query metadata note'
    assert body['queries'][1]['metadata']['location'] == 'Eagle Point'


def test_first_order_media_routes_return_best_first_descriptors_and_resized_previews(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    query_dir = archive / 'queries' / 'query_001' / '01_02_26_a'
    gallery_dir = archive / 'gallery' / 'media_anchovy' / '03_15_24'
    query_dir.mkdir(parents=True)
    gallery_dir.mkdir(parents=True)
    query_first = query_dir / 'query_first.jpg'
    query_best = query_dir / 'query_best.jpg'
    gallery_first = gallery_dir / 'gallery_first.jpg'
    gallery_best = gallery_dir / 'gallery_best.jpg'
    Image.new('RGB', (40, 20), color=(255, 0, 0)).save(query_first)
    Image.new('RGB', (80, 40), color=(0, 255, 0)).save(query_best)
    Image.new('RGB', (30, 90), color=(0, 0, 255)).save(gallery_first)
    Image.new('RGB', (120, 60), color=(255, 255, 0)).save(gallery_best)
    (archive / 'queries' / 'query_001' / '.best_photo.json').write_text(
        '{"best_rel": "01_02_26_a/query_best.jpg"}', encoding='utf-8'
    )
    (archive / 'gallery' / 'media_anchovy' / '.best_photo.json').write_text(
        '{"best_rel": "03_15_24/gallery_best.jpg"}', encoding='utf-8'
    )
    (archive / 'queries' / 'queries_metadata.csv').write_text('query_id,location\nquery_001,Friday Harbor\n', encoding='utf-8-sig')
    (archive / 'gallery' / 'gallery_metadata.csv').write_text('gallery_id,location\nmedia_anchovy,Friday Harbor\n', encoding='utf-8-sig')
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(archive))

    client = TestClient(create_app())
    headers = {'cf-access-authenticated-user-email': 'field@example.org'}

    query_media = client.get('/api/first-order/queries/query_001/media', headers=headers)
    assert query_media.status_code == 200
    query_body = query_media.json()
    assert query_body['target_type'] == 'query'
    assert query_body['entity_id'] == 'query_001'
    assert [image['label'] for image in query_body['images']] == ['query_best.jpg', 'query_first.jpg']
    assert query_body['images'][0]['is_best'] is True
    assert query_body['images'][0]['preview_url'].endswith('/preview')
    assert query_body['images'][0]['fullres_url'].endswith('/full')

    gallery_media = client.get('/api/first-order/candidates/media_anchovy/media', headers=headers)
    assert gallery_media.status_code == 200
    gallery_body = gallery_media.json()
    assert gallery_body['target_type'] == 'gallery'
    assert [image['label'] for image in gallery_body['images']] == ['gallery_best.jpg', 'gallery_first.jpg']
    assert gallery_body['images'][0]['is_best'] is True

    preview = client.get(query_body['images'][0]['preview_url'], headers=headers)
    full = client.get(query_body['images'][0]['fullres_url'], headers=headers)
    assert preview.status_code == 200
    assert full.status_code == 200
    assert preview.headers['content-type'].startswith('image/')
    assert full.headers['content-type'].startswith('image/')
    assert len(preview.content) < len(full.content)


def test_first_order_preview_is_large_enough_for_query_matcher_comparison(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    query_dir = archive / 'queries' / 'query_001' / '01_02_26_a'
    query_dir.mkdir(parents=True)
    query_image = query_dir / 'query_large.jpg'
    Image.linear_gradient('L').resize((1600, 900)).convert('RGB').save(query_image, quality=95)
    (archive / 'queries' / 'queries_metadata.csv').write_text('query_id,location\nquery_001,Friday Harbor\n', encoding='utf-8-sig')
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(archive))

    client = TestClient(create_app())
    headers = {'cf-access-authenticated-user-email': 'field@example.org'}
    query_media = client.get('/api/first-order/queries/query_001/media', headers=headers)
    preview = client.get(query_media.json()['images'][0]['preview_url'], headers=headers)

    assert preview.status_code == 200
    preview_image = Image.open(BytesIO(preview.content))
    assert max(preview_image.size) >= 1200
