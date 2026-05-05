from fastapi.testclient import TestClient

from src.data.archive_paths import metadata_csv_for
from src.data.csv_io import append_row

from .conftest import build_test_app

AUTH = {'cf-access-authenticated-user-email': 'field@example.org'}


def test_metadata_schema_requires_auth(tmp_path, monkeypatch):
    app, _archive = build_test_app(tmp_path, monkeypatch)
    client = TestClient(app)
    r = client.get('/api/schema/metadata')
    assert r.status_code == 401


def test_metadata_schema_returns_fields(tmp_path, monkeypatch):
    app, _archive = build_test_app(tmp_path, monkeypatch)
    client = TestClient(app)
    r = client.get('/api/schema/metadata', headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert 'fields' in body
    assert len(body['fields']) > 0
    names = {f['name'] for f in body['fields']}
    assert 'location' in names
    widgets = {f['name']: f['mobile_widget'] for f in body['fields']}
    assert widgets['health_codes'] == 'health_code'
    health = next(f for f in body['fields'] if f['name'] == 'health_codes')
    assert [option['value'] for option in health['options'][:3]] == ['X', 'NA', 'UNK']
    assert [option['category'] for option in health['options'][:3]] == ['normal', 'feeding', 'status']
    lesion = next(option for option in health['options'] if option['value'] == 'L')
    assert lesion['requires_count'] is True
    assert lesion['allows_plus'] is True
    assert lesion['definition']


def test_metadata_schema_location_vocabulary_uses_saved_sites_not_free_text_history(tmp_path, monkeypatch):
    app, archive = build_test_app(tmp_path, monkeypatch)
    vocab_path = archive / 'vocabularies' / 'locations.json'
    vocab_path.parent.mkdir(parents=True)
    vocab_path.write_text('["Polluted import folder"]', encoding='utf-8')
    gallery_csv, gallery_header = metadata_csv_for('Gallery')
    append_row(gallery_csv, gallery_header, {
        'gallery_id': 'g1',
        'location': 'Dock',
        'latitude': '48.546',
        'longitude': '-123.013',
    })
    client = TestClient(app)

    r = client.get('/api/schema/metadata', headers=AUTH)

    assert r.status_code == 200
    location = next(f for f in r.json()['fields'] if f['name'] == 'location')
    assert location['vocabulary'] == ['Dock']
