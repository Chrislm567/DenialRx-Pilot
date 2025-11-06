from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from packages.security.logging.redaction import OpenTelemetryRedactor, PHIRedactingFilter, RedactionPolicy
from packages.security.phi.tokenization import DeterministicTokenizer, TokenizationConfig


def test_deterministic_tokenizer_returns_same_token():
    config = TokenizationConfig(secret=b"secret")
    tokenizer = DeterministicTokenizer(config)
    first = tokenizer.tokenize("John Doe", context="patient")
    second = tokenizer.tokenize("John Doe", context="patient")
    assert first == second


def test_deterministic_tokenizer_is_context_sensitive():
    config = TokenizationConfig(secret=b"secret")
    tokenizer = DeterministicTokenizer(config)
    token_a = tokenizer.tokenize("John Doe", context="patient")
    token_b = tokenizer.tokenize("John Doe", context="subscriber")
    assert token_a != token_b


def test_tokenizer_uses_kms_hook():
    kms = MagicMock()
    kms.fetch_key.return_value = b"kms-secret"
    config = TokenizationConfig(secret=b"unused", key_id="kms-key")
    tokenizer = DeterministicTokenizer(config, kms=kms)
    token = tokenizer.tokenize("123-45-6789")  # PHI_OK
    kms.fetch_key.assert_called_once_with("kms-key")
    assert token


def test_phi_redacting_filter_masks_patterns():
    policy = RedactionPolicy(placeholder="[MASK]")
    filter_ = PHIRedactingFilter(policy)
    message = "Patient SSN 123-45-6789 called from 555-123-4567"  # PHI_OK
    redacted = filter_._redact(message)
    assert "123-45-6789" not in redacted  # PHI_OK
    assert "555-123-4567" not in redacted  # PHI_OK
    assert redacted.count("[MASK]") >= 2


def test_opentelemetry_redactor_sanitizes_attributes():
    redactor = OpenTelemetryRedactor()
    attributes = {"ssn": "123-45-6789", "note": "Call 555.987.6543"}  # PHI_OK
    sanitized = redactor.sanitize(attributes)
    assert "123-45-6789" not in sanitized["ssn"]  # PHI_OK
    assert "555.987.6543" not in sanitized["note"]


@pytest.mark.parametrize(
    "value",
    ["", None],
)
def test_tokenizer_handles_empty_values_gracefully(value):
    config = TokenizationConfig(secret=b"secret")
    tokenizer = DeterministicTokenizer(config)
    assert tokenizer.tokenize(value or "") == (value or "")
