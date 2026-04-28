from fastapi.testclient import TestClient

from src.data.csv_io import append_row
from src.data.archive_paths import metadata_csv_for
from src.data.field_visits import append_field_visit
from star_browser.app.main import create_app


AUTH = {'cf-access-authenticated-user-email': 'field@example.org'}


def test_location_sites_requires_auth(tmp_path, monkeypatch):
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(tmp_path / 'archive'))
    client = TestClient(create_app())

    response = client.get('/api/locations/sites')

    assert response.status_code == 401


def test_location_sites_returns_known_sites_with_coordinates(tmp_path, monkeypatch):
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(tmp_path / 'archive'))
    gallery_csv, gallery_header = metadata_csv_for('Gallery')
    append_row(gallery_csv, gallery_header, {
        'gallery_id': 'g1',
        'location': 'Dock',
        'latitude': '48.546000',
        'longitude': '-123.013000',
    })
    append_field_visit(
        visit_date=__import__('datetime').date(2026, 4, 21),
        location='Pier',
        latitude=48.500000,
        longitude=-123.200000,
    )
    client = TestClient(create_app())

    response = client.get('/api/locations/sites', headers=AUTH)

    assert response.status_code == 200
    body = response.json()
    names = {site['name'] for site in body['sites']}
    assert 'Dock' in names
    assert 'Pier' in names
    dock = next(site for site in body['sites'] if site['name'] == 'Dock')
    assert dock['latitude'] == 48.546
    assert dock['longitude'] == -123.013
