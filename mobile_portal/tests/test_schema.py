from fastapi.testclient import TestClient

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
