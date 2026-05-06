from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data.archive_paths import archive_root

_SCHEMA_VERSION = 1
_LOCK = threading.Lock()
_LOG = logging.getLogger('starboard.activity')


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, sort_keys=True)
    except (TypeError, ValueError):
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(v) for v in value]
        return str(value)
    return value


def activity_log_dir(root: Path | None = None) -> Path:
    return (root or archive_root()) / 'logs' / 'activity'


def record_activity_event(
    *,
    surface: str,
    user_email: str,
    event_type: str,
    session_id: str | None = None,
    workflow: str | None = None,
    entity_type: str | None = None,
    entity_id: str | None = None,
    query_id: str | None = None,
    gallery_id: str | None = None,
    success: bool | None = None,
    duration_ms: int | None = None,
    details: dict[str, Any] | None = None,
    request_path: str | None = None,
    request_method: str | None = None,
    user_agent: str | None = None,
    client_ip: str | None = None,
    client_timestamp_utc: str | None = None,
    auth_source: str = 'cloudflare_access',
    event_id: str | None = None,
) -> dict[str, Any]:
    """Append one structured browser/mobile activity event to archive JSONL."""
    now = _utc_now()
    event = {
        'schema_version': _SCHEMA_VERSION,
        'event_id': event_id or str(uuid.uuid4()),
        'timestamp_utc': now.isoformat().replace('+00:00', 'Z'),
        'client_timestamp_utc': client_timestamp_utc,
        'surface': surface,
        'session_id': session_id or '',
        'user_email': user_email,
        'auth_source': auth_source,
        'request_path': request_path,
        'request_method': request_method,
        'event_type': event_type,
        'entity_type': entity_type,
        'entity_id': entity_id,
        'query_id': query_id,
        'gallery_id': gallery_id,
        'workflow': workflow,
        'success': success,
        'duration_ms': duration_ms,
        'details': _json_safe(details or {}),
        'user_agent': user_agent,
        'client_ip': client_ip,
    }
    log_dir = activity_log_dir()
    log_path = log_dir / f"activity_{now.date().isoformat()}.jsonl"
    line = json.dumps(event, ensure_ascii=False, sort_keys=True)
    with _LOCK:
        log_dir.mkdir(parents=True, exist_ok=True)
        with log_path.open('a', encoding='utf-8') as fh:
            fh.write(line + '\n')
    return event


def try_record_activity_event(**kwargs: Any) -> dict[str, Any] | None:
    """Best-effort activity logging for primary archive workflows."""
    try:
        return record_activity_event(**kwargs)
    except Exception:
        _LOG.exception('activity event logging failed')
        return None


def request_context(request: Any) -> dict[str, str | None]:
    client_ip = request.headers.get('cf-connecting-ip') or request.headers.get('x-forwarded-for') or (request.client.host if getattr(request, 'client', None) else None)
    return {
        'request_path': str(request.url.path),
        'request_method': request.method,
        'user_agent': request.headers.get('user-agent'),
        'client_ip': client_ip,
        'session_id': request.headers.get('X-Starboard-Session-Id') or request.headers.get('x-starboard-session-id') or '',
    }
