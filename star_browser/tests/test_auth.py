from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from star_browser.app.auth import require_authenticated_email
from star_browser.app.main import create_app


def test_auth_dependency_requires_header():
    app = FastAPI()

    @app.get('/protected')
    def protected(user_email: str = Depends(require_authenticated_email)):
        return {'authenticated_email': user_email}

    client = TestClient(app)
    r = client.get('/protected')
    assert r.status_code == 401


def test_session_returns_authenticated_email():
    client = TestClient(create_app())
    r = client.get('/api/session', headers={'cf-access-authenticated-user-email': 'field@example.org'})
    assert r.status_code == 200
    body = r.json()
    assert body['authenticated_email'] == 'field@example.org'
    assert 'batch_upload' in body['capabilities']
