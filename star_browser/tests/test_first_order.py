from fastapi.testclient import TestClient

from star_browser.app.main import create_app


def test_first_order_search_route_returns_ranked_candidate(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    gallery = archive / 'gallery'
    queries = archive / 'queries'
    gallery.mkdir(parents=True)
    queries.mkdir(parents=True)
    (gallery / 'gallery_metadata.csv').write_text('gallery_id,location\nanchovy,Friday Harbor\n', encoding='utf-8-sig')
    (queries / 'queries_metadata.csv').write_text('query_id,location\nquery_001,Friday Harbor\n', encoding='utf-8-sig')
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(archive))

    client = TestClient(create_app())
    r = client.post(
        '/api/first-order/search',
        headers={'cf-access-authenticated-user-email': 'field@example.org'},
        json={'query_id': 'query_001', 'top_k': 5},
    )
    assert r.status_code == 200
    body = r.json()
    assert body['query_id'] == 'query_001'
    assert len(body['candidates']) >= 1
    assert body['candidates'][0]['entity_id'] == 'anchovy'
