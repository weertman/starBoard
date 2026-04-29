from fastapi.testclient import TestClient

from public_site.app.main import create_app


def test_root_serves_public_landing_page_without_auth_header():
    client = TestClient(create_app())

    response = client.get('/')

    assert response.status_code == 200
    assert 'text/html' in response.headers['content-type']
    assert 'starBoard' in response.text
    assert 'mobile.fhl-star-board.com' in response.text
    assert 'browser.fhl-star-board.com' in response.text
    assert 'Cloudflare Access' not in response.text


def test_root_supports_head_for_public_edge_probes():
    client = TestClient(create_app())

    response = client.head('/')

    assert response.status_code == 200
    assert 'text/html' in response.headers['content-type']


def test_health_reports_public_site_service():
    client = TestClient(create_app())

    response = client.get('/api/health')

    assert response.status_code == 200
    assert response.json() == {'status': 'ok', 'service': 'starboard-public-site'}
