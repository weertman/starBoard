import logging

from mobile_portal.app.services.audit import audit


def test_audit_logs_structured_action(caplog):
    caplog.set_level(logging.INFO, logger='starboard.mobile_portal.audit')
    audit('lookup_entity', 'field@example.org', entity_id='feta')
    assert any('lookup_entity' in rec.message for rec in caplog.records)
