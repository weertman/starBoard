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
    assert 'github.com/weertman/starBoard' in response.text
    assert 'currently in active development' in response.text
    assert 'href="https://together.uw.edu/campaign/seastars"' in response.text
    assert 'href="/contact"' in response.text
    assert 'href="/lab"' in response.text
    assert 'href="https://mobile.fhl-star-board.com/" target="_blank" rel="noopener noreferrer"' in response.text
    assert 'href="https://browser.fhl-star-board.com/" target="_blank" rel="noopener noreferrer"' in response.text
    assert 'Cloudflare Access' not in response.text


def test_root_supports_head_for_public_edge_probes():
    client = TestClient(create_app())

    response = client.head('/')

    assert response.status_code == 200
    assert 'text/html' in response.headers['content-type']


def test_contact_page_lists_project_contact():
    client = TestClient(create_app())

    response = client.get('/contact')

    assert response.status_code == 200
    assert 'text/html' in response.headers['content-type']
    assert 'Willem Weertman' in response.text
    assert 'wlweert@gmail.com' in response.text
    assert 'mailto:wlweert@gmail.com' in response.text


def test_lab_page_summarizes_uwfhl_sea_star_lab():
    client = TestClient(create_app())

    response = client.get('/lab')

    assert response.status_code == 200
    assert 'UW Friday Harbor Laboratories' in response.text
    assert 'Dr. Jason Hodin' in response.text
    assert 'raising sunflower sea stars in captivity since 2019' in response.text
    assert 'Pycnopodia helianthoides' in response.text
    assert 'purple sea urchins' in response.text
    assert 'kelp forests' in response.text


def test_health_reports_public_site_service():
    client = TestClient(create_app())

    response = client.get('/api/health')

    assert response.status_code == 200
    assert response.json() == {'status': 'ok', 'service': 'starboard-public-site'}
