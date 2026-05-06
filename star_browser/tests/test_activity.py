import json

from fastapi.testclient import TestClient

from star_browser.app.main import create_app

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
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(tmp_path / 'archive'))
    client = TestClient(create_app())

    response = client.post('/api/activity/events', json={'events': [{'event_type': 'ui.tab.open'}]})

    assert response.status_code == 401


def test_activity_events_are_written_with_authenticated_user_and_session(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(archive))
    client = TestClient(create_app())

    response = client.post(
        '/api/activity/events',
        headers={**AUTH, 'X-Starboard-Session-Id': 'browser-session-1'},
        json={
            'events': [
                {
                    'event_type': 'query_matcher.query.selected',
                    'workflow': 'query_matcher',
                    'entity_type': 'query',
                    'entity_id': 'q1',
                    'query_id': 'q1',
                    'details': {'source': 'selector'},
                }
            ]
        },
    )

    assert response.status_code == 200
    assert response.json()['accepted_events'] == 1
    events = _read_activity_events(archive)
    assert len(events) == 1
    event = events[0]
    assert event['surface'] == 'star_browser'
    assert event['user_email'] == 'field@example.org'
    assert event['session_id'] == 'browser-session-1'
    assert event['event_type'] == 'query_matcher.query.selected'
    assert event['workflow'] == 'query_matcher'
    assert event['query_id'] == 'q1'
    assert event['details'] == {'source': 'selector'}
    assert event['schema_version'] == 1
    assert event['event_id']
    assert event['timestamp_utc']


def test_activity_body_session_id_wins_over_header_session_id(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(archive))
    client = TestClient(create_app())

    response = client.post(
        '/api/activity/events',
        headers={**AUTH, 'X-Starboard-Session-Id': 'browser-header-session'},
        json={
            'session_id': 'browser-body-session',
            'events': [{'event_type': 'ui.tab.open', 'workflow': 'single-entry'}],
        },
    )

    assert response.status_code == 200
    events = _read_activity_events(archive)
    assert len(events) == 1
    assert events[0]['session_id'] == 'browser-body-session'


def test_session_request_is_logged_server_side(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(archive))
    client = TestClient(create_app())

    response = client.get('/api/session', headers={**AUTH, 'X-Starboard-Session-Id': 'browser-session-2'})

    assert response.status_code == 200
    events = _read_activity_events(archive)
    assert any(
        event['event_type'] == 'session.loaded'
        and event['surface'] == 'star_browser'
        and event['user_email'] == 'field@example.org'
        and event['session_id'] == 'browser-session-2'
        for event in events
    )
