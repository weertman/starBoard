import json

from fastapi.testclient import TestClient

from .conftest import build_test_app

AUTH = {'cf-access-authenticated-user-email': 'field@example.org'}


def _read_activity_events(archive):
    paths = sorted((archive / 'logs' / 'activity').glob('activity_*.jsonl'))
    assert paths, 'expected an activity JSONL file'
    events = []
    for path in paths:
        for line in path.read_text(encoding='utf-8').splitlines():
            events.append(json.loads(line))
    return events


def test_activity_events_require_authentication(tmp_path, monkeypatch):
    app, _archive = build_test_app(tmp_path, monkeypatch)
    client = TestClient(app)

    response = client.post('/api/activity/events', json={'events': [{'event_type': 'mobile.observation.open'}]})

    assert response.status_code == 401


def test_activity_events_are_written_with_authenticated_user_and_session(tmp_path, monkeypatch):
    app, archive = build_test_app(tmp_path, monkeypatch)
    client = TestClient(app)

    response = client.post(
        '/api/activity/events',
        headers={**AUTH, 'X-Starboard-Session-Id': 'mobile-session-1'},
        json={
            'events': [
                {
                    'event_type': 'mobile.observation.open',
                    'workflow': 'observation',
                    'details': {'entry': 'home'},
                }
            ]
        },
    )

    assert response.status_code == 200
    assert response.json()['accepted_events'] == 1
    events = _read_activity_events(archive)
    assert len(events) == 1
    event = events[0]
    assert event['surface'] == 'mobile_portal'
    assert event['user_email'] == 'field@example.org'
    assert event['session_id'] == 'mobile-session-1'
    assert event['event_type'] == 'mobile.observation.open'
    assert event['workflow'] == 'observation'
    assert event['details'] == {'entry': 'home'}


def test_activity_body_session_id_wins_over_header_session_id(tmp_path, monkeypatch):
    app, archive = build_test_app(tmp_path, monkeypatch)
    client = TestClient(app)

    response = client.post(
        '/api/activity/events',
        headers={**AUTH, 'X-Starboard-Session-Id': 'mobile-header-session'},
        json={
            'session_id': 'mobile-body-session',
            'events': [{'event_type': 'mobile.screen.open', 'workflow': 'home'}],
        },
    )

    assert response.status_code == 200
    events = _read_activity_events(archive)
    assert len(events) == 1
    assert events[0]['session_id'] == 'mobile-body-session'


def test_session_request_is_logged_server_side(tmp_path, monkeypatch):
    app, archive = build_test_app(tmp_path, monkeypatch)
    client = TestClient(app)

    response = client.get('/api/session', headers={**AUTH, 'X-Starboard-Session-Id': 'mobile-session-2'})

    assert response.status_code == 200
    events = _read_activity_events(archive)
    assert any(
        event['event_type'] == 'session.loaded'
        and event['surface'] == 'mobile_portal'
        and event['user_email'] == 'field@example.org'
        and event['session_id'] == 'mobile-session-2'
        for event in events
    )
