from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any
from urllib import error, request

from fastapi import HTTPException, status

from ..config import MegaStarCapabilityStatus, Settings
from ..models.megastar_api import MegaStarLookupResponse
from .megastar_lookup_service import MegaStarLookupUnavailable


@dataclass(frozen=True)
class MegaStarWorkerClient:
    settings: Settings

    def capability_status(self) -> MegaStarCapabilityStatus:
        if not self.settings.megastar_enabled:
            return MegaStarCapabilityStatus(
                enabled=False,
                state='disabled',
                backend='worker',
                reason='feature_flag_disabled',
                model_key=self.settings.megastar_model_key_override,
                artifact_dir=None,
            )

        try:
            payload = self._request_json('/status', method='GET')
        except MegaStarLookupUnavailable as exc:
            return MegaStarCapabilityStatus(
                enabled=False,
                state='unavailable',
                backend='worker',
                reason=str(exc),
                model_key=self.settings.megastar_model_key_override,
                artifact_dir=None,
            )

        state = payload.get('state') if isinstance(payload, dict) else None
        reason = payload.get('reason') if isinstance(payload, dict) else None
        model_key = payload.get('model_key') if isinstance(payload, dict) else None
        enabled = bool(payload.get('enabled')) if isinstance(payload, dict) else False
        if state not in {'enabled', 'disabled', 'unavailable'}:
            return MegaStarCapabilityStatus(
                enabled=False,
                state='unavailable',
                backend='worker',
                reason='worker_status_invalid',
                model_key=model_key if isinstance(model_key, str) else self.settings.megastar_model_key_override,
                artifact_dir=None,
            )

        return MegaStarCapabilityStatus(
            enabled=enabled,
            state=state,
            backend='worker',
            reason=reason if isinstance(reason, str) else None,
            model_key=model_key if isinstance(model_key, str) else None,
            artifact_dir=None,
        )

    def lookup_upload(self, *, filename: str, content: bytes, content_type: str | None = None, max_candidates: int = 5) -> MegaStarLookupResponse:
        boundary = f'----starboard-megastar-{uuid.uuid4().hex}'
        file_content_type = content_type or 'application/octet-stream'
        body = b''.join(
            (
                f'--{boundary}\r\n'.encode('utf-8'),
                (
                    f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                    f'Content-Type: {file_content_type}\r\n\r\n'
                ).encode('utf-8'),
                content,
                b'\r\n',
                f'--{boundary}--\r\n'.encode('utf-8'),
            )
        )
        payload = self._request_json(
            f'/lookup?max_candidates={max_candidates}',
            method='POST',
            data=body,
            headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
        )
        return MegaStarLookupResponse.model_validate(payload)

    def _request_json(
        self,
        path: str,
        *,
        method: str,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        req = request.Request(
            url=f'{self.settings.megastar_worker_url}{path}',
            data=data,
            method=method,
            headers=headers or {},
        )
        try:
            with request.urlopen(req, timeout=self.settings.megastar_worker_timeout_seconds) as response:
                payload = response.read().decode('utf-8')
        except error.HTTPError as exc:
            response_body = exc.read().decode('utf-8', errors='replace')
            parsed = self._parse_json(response_body)
            if exc.code == status.HTTP_400_BAD_REQUEST:
                detail = parsed.get('detail') if isinstance(parsed, dict) else None
                raise HTTPException(status_code=exc.code, detail=detail or 'worker_request_rejected') from exc
            if exc.code == status.HTTP_503_SERVICE_UNAVAILABLE:
                reason = parsed.get('availability_reason') if isinstance(parsed, dict) else None
                raise MegaStarLookupUnavailable(reason or 'worker_unavailable') from exc
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail='megastar_lookup_failed') from exc
        except error.URLError as exc:
            raise MegaStarLookupUnavailable('worker_unreachable') from exc

        parsed = self._parse_json(payload)
        if not isinstance(parsed, dict):
            raise MegaStarLookupUnavailable('worker_response_invalid')
        return parsed

    def _parse_json(self, payload: str) -> dict[str, Any] | None:
        if not payload:
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None
