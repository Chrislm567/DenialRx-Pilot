from __future__ import annotations

from typing import Any, Dict, Optional

import httpx


class AppealsClient:
    """Thin SDK wrapper for the DenialRx Appeals API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 10.0,
        api_key: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def draft(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self._client.post("/appeals/draft", json=payload, headers=self._headers())
        response.raise_for_status()
        return response.json()

    def submit(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self._client.post("/appeals/submit", json=payload, headers=self._headers())
        response.raise_for_status()
        return response.json()

    def status(self, appeal_id: str) -> Dict[str, Any]:
        response = self._client.get(f"/appeals/{appeal_id}", headers=self._headers())
        response.raise_for_status()
        return response.json()

    def audit(self, appeal_id: str) -> Dict[str, Any]:
        response = self._client.get(f"/appeals/{appeal_id}/audit", headers=self._headers())
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "AppealsClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
