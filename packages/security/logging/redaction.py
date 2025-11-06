"""Logging and telemetry redaction middleware."""
from __future__ import annotations

import re
from dataclasses import dataclass
from logging import LogRecord
from typing import Iterable, Mapping


PHI_PATTERNS = {
    "ssn": re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b"),
    "mrn": re.compile(r"\b[0-9]{6,10}\b"),
    "phone": re.compile(r"\b\d{3}[\s.-]?\d{3}[\s.-]?\d{4}\b"),
}


@dataclass
class RedactionPolicy:
    placeholder: str = "[REDACTED]"
    extra_patterns: Mapping[str, re.Pattern[str]] | None = None

    def compile(self) -> Mapping[str, re.Pattern[str]]:
        patterns = dict(PHI_PATTERNS)
        if self.extra_patterns:
            patterns.update(self.extra_patterns)
        return patterns


class PHIRedactingFilter:
    """Standard logging filter that masks PHI patterns in log messages."""

    def __init__(self, policy: RedactionPolicy | None = None):
        self.policy = policy or RedactionPolicy()
        self.patterns = self.policy.compile()

    def filter(self, record: LogRecord) -> bool:  # pragma: no cover - logging integration
        record.msg = self._redact(record.getMessage())
        return True

    def _redact(self, message: str) -> str:
        redacted = message
        for pattern in self.patterns.values():
            redacted = pattern.sub(self.policy.placeholder, redacted)
        return redacted


class OpenTelemetryRedactor:
    """Utility to redact attribute dictionaries before exporting to OTLP."""

    def __init__(self, policy: RedactionPolicy | None = None):
        self.policy = policy or RedactionPolicy()
        self.patterns = self.policy.compile()

    def sanitize(self, attributes: Mapping[str, str]) -> Mapping[str, str]:
        sanitized = {}
        for key, value in attributes.items():
            sanitized[key] = self._sanitize_value(value)
        return sanitized

    def _sanitize_value(self, value: str) -> str:
        redacted = value
        for pattern in self.patterns.values():
            redacted = pattern.sub(self.policy.placeholder, redacted)
        return redacted

    def sanitize_events(self, events: Iterable[Mapping[str, str]]) -> Iterable[Mapping[str, str]]:
        for event in events:
            yield {k: self._sanitize_value(v) for k, v in event.items()}


__all__ = ["OpenTelemetryRedactor", "PHIRedactingFilter", "RedactionPolicy"]
