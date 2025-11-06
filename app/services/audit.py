from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

from app.config import settings
from app.models import AuditRecord
from app.utils.phi import hash_payload, redact_phi


@dataclass
class AuditService:
    path: Path = Path(settings.audit_log_path)

    def append(self, action: str, actor: str, appeal_id: str, payload: Dict) -> AuditRecord:
        timestamp = datetime.now(timezone.utc)
        redacted = redact_phi(payload)
        payload_hash = hash_payload(redacted)
        record = AuditRecord(
            appeal_id=appeal_id,
            action=action,
            actor=actor,
            occurred_at=timestamp,
            payload_hash=payload_hash,
            payload_schema=f"v1:{action}",
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "appeal_id": record.appeal_id,
                "action": record.action,
                "actor": record.actor,
                "occurred_at": record.occurred_at.isoformat(),
                "payload_hash": record.payload_hash,
                "payload_schema": record.payload_schema,
                "payload": redacted,
            }) + "\n")
        return record

    def load(self) -> List[AuditRecord]:
        if not self.path.exists():
            return []
        records: List[AuditRecord] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                data = json.loads(line)
                records.append(
                    AuditRecord(
                        appeal_id=data["appeal_id"],
                        action=data["action"],
                        actor=data["actor"],
                        occurred_at=datetime.fromisoformat(data["occurred_at"]),
                        payload_hash=data["payload_hash"],
                        payload_schema=data["payload_schema"],
                    )
                )
        return records

    def history_for(self, appeal_id: str) -> Iterable[AuditRecord]:
        return [record for record in self.load() if record.appeal_id == appeal_id]


def get_audit_service() -> AuditService:
    return AuditService()
