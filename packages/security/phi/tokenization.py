"""Deterministic PHI tokenization utilities."""
from __future__ import annotations

import base64
import hashlib
import hmac
from dataclasses import dataclass
from typing import Protocol


class KeyManagementHook(Protocol):
    """Lightweight abstraction for fetching encryption keys from a KMS."""

    def fetch_key(self, key_id: str) -> bytes:
        """Return the raw key material for the provided key identifier."""


@dataclass
class TokenizationConfig:
    secret: bytes
    key_id: str = "default"
    salt: bytes = b"denialrx-phase"

    def materialize(self, kms: KeyManagementHook | None = None) -> bytes:
        if kms is None:
            return self.secret
        return kms.fetch_key(self.key_id)


class DeterministicTokenizer:
    """Tokenizes PHI fields using keyed hashing with domain separation."""

    def __init__(self, config: TokenizationConfig, kms: KeyManagementHook | None = None):
        self._config = config
        self._kms = kms
        self._materialized_secret = self._config.materialize(kms)

    def tokenize(self, value: str, context: str = "") -> str:
        if not value:
            return value
        digest = hmac.digest(
            self._materialized_secret,
            self._config.salt + context.encode("utf-8") + value.encode("utf-8"),
            hashlib.sha256,
        )
        return base64.urlsafe_b64encode(digest)[:22].decode("ascii")

    def detokenize(self, token: str) -> str:
        raise NotImplementedError("Detokenization is intentionally unsupported")


__all__ = ["DeterministicTokenizer", "KeyManagementHook", "TokenizationConfig"]
