from fastapi.testclient import TestClient

from src.data.archive_paths import metadata_csv_for
from src.data.csv_io import append_row
from star_browser.app.main import create_app


AUTH = {'cf-access-authenticated-user-email': 'field@example.org'}


def test_metadata_schema_requires_auth():
    client = TestClient(create_app())

    response = client.get('/api/schema/metadata')

    assert response.status_code == 401


def test_metadata_schema_returns_projected_fields():
    client = TestClient(create_app())

    response = client.get('/api/schema/metadata', headers=AUTH)

    assert response.status_code == 200
    body = response.json()
    assert 'fields' in body
    assert len(body['fields']) > 0
    names = {field['name'] for field in body['fields']}
    assert 'location' in names
    assert 'num_apparent_arms' in names
    widgets = {field['name']: field['mobile_widget'] for field in body['fields']}
    assert widgets['location'] == 'location'
    assert widgets['short_arm_code'] == 'short_arm_code'
    assert widgets['health_codes'] == 'health_code'
    health = next(field for field in body['fields'] if field['name'] == 'health_codes')
    assert [option['value'] for option in health['options'][:3]] == ['X', 'NA', 'UNK']
    assert [option['category'] for option in health['options'][:3]] == ['normal', 'feeding', 'status']
    lesion = next(option for option in health['options'] if option['value'] == 'L')
    assert lesion['requires_count'] is True
    assert lesion['allows_plus'] is True
    assert lesion['definition']


def test_metadata_schema_location_vocabulary_uses_saved_sites_not_free_text_history(tmp_path, monkeypatch):
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(tmp_path / 'archive'))
    vocab_path = tmp_path / 'archive' / 'vocabularies' / 'locations.json'
    vocab_path.parent.mkdir(parents=True)
    vocab_path.write_text('["Polluted import folder"]', encoding='utf-8')
    gallery_csv, gallery_header = metadata_csv_for('Gallery')
    append_row(gallery_csv, gallery_header, {
        'gallery_id': 'g1',
        'location': 'Dock',
        'latitude': '48.546',
        'longitude': '-123.013',
    })
    client = TestClient(create_app())

    response = client.get('/api/schema/metadata', headers=AUTH)

    assert response.status_code == 200
    location = next(field for field in response.json()['fields'] if field['name'] == 'location')
    assert location['vocabulary'] == ['Dock']
