from pathlib import Path

from fastapi.testclient import TestClient

from star_browser.app.main import create_app


def test_root_serves_built_frontend_index():
    client = TestClient(create_app())

    response = client.get('/')

    assert response.status_code == 200
    assert 'text/html' in response.headers['content-type']
    assert '<title>star_browser</title>' in response.text


def test_assets_route_serves_built_frontend_bundle():
    dist_assets = Path(__file__).resolve().parents[1] / 'frontend' / 'dist' / 'assets'
    asset_path = next(p for p in dist_assets.iterdir() if p.is_file() and p.suffix == '.js')
    client = TestClient(create_app())

    response = client.get(f'/assets/{asset_path.name}')

    assert response.status_code == 200
    assert response.content
