from fastapi.testclient import TestClient

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
