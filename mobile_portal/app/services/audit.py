from __future__ import annotations

import logging

log = logging.getLogger('starboard.mobile_portal.audit')


def audit(action: str, user_email: str, **fields) -> None:
    payload = {'action': action, 'user_email': user_email, **fields}
    log.info('audit %s', payload)
