from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable

SENSITIVE_FIELDS = {
    "ssn",
    "social_security_number",
    "diagnosis_details",
    "credit_card",
    "full_address",
    "date_of_birth",
}

ALLOWED_PATIENT_FIELDS = {
    "patient_first_name",
    "patient_last_name",
    "member_id",
}


def redact_phi(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Redact known PHI fields by hashing their values."""
    redacted: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            redacted[key] = redact_phi(value)
            continue
        if key in SENSITIVE_FIELDS or _looks_like_phi_value(value):
            redacted[key] = "[REDACTED]"
        elif key in ALLOWED_PATIENT_FIELDS:
            redacted[key] = value
        elif _looks_like_phi_field(key):
            redacted[key] = "[REDACTED]"
        else:
            redacted[key] = value
    return redacted


def _looks_like_phi_field(key: str) -> bool:
    lower = key.lower()
    return any(fragment in lower for fragment in ["dob", "birth", "ssn", "address"])


def _looks_like_phi_value(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    lowered = value.lower()
    return any(fragment in lowered for fragment in ["dob", "date of birth", "ssn", "social security"])


def hash_payload(payload: Dict[str, Any]) -> str:
    serialized = repr(sorted(_flatten_items(payload)))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _flatten_items(payload: Dict[str, Any], prefix: str = "") -> Iterable[str]:
    for key, value in payload.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from _flatten_items(value, new_key)
        else:
            yield f"{new_key}={value}" if value is not None else f"{new_key}="
