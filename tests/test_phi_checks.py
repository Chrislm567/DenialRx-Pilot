from app.utils.phi import ALLOWED_PATIENT_FIELDS, hash_payload, redact_phi


def test_phi_redaction_masks_sensitive_fields():
    payload = {
        "patient_first_name": "Casey",
        "patient_last_name": "Rivers",
        "member_id": "ABC123",
        "ssn": "123-45-6789",
        "full_address": "123 Main St",
        "notes": "DOB: 1990-01-01",
    }

    redacted = redact_phi(payload)

    assert redacted["ssn"] == "[REDACTED]"
    assert redacted["full_address"] == "[REDACTED]"
    assert redacted["notes"] == "[REDACTED]"
    for field in ALLOWED_PATIENT_FIELDS:
        assert redacted[field] == payload[field]


def test_hash_payload_stable_order():
    payload_a = {"a": 1, "b": {"c": 2}}
    payload_b = {"b": {"c": 2}, "a": 1}

    assert hash_payload(payload_a) == hash_payload(payload_b)
