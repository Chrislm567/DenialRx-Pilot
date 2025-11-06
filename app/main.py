from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from app.logging import configure_logging
from app.models import AppealDraft, AppealStatus, AppealSubmission, PHICheckResult
from app.services.audit import get_audit_service
from app.services.templates import render_denial_letter, render_submission_email
from app.utils.phi import ALLOWED_PATIENT_FIELDS, hash_payload, redact_phi

app = FastAPI(title="DenialRx Appeals API", version="0.3.0")
logger = configure_logging()
audit_service = get_audit_service()

APPEAL_STORE: Dict[str, Dict] = {}


@app.get("/healthz")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


def perform_phi_check(draft: AppealDraft) -> PHICheckResult:
    payload = draft.model_dump()
    redacted = redact_phi(payload)
    reasons = []
    for key in payload:
        if key.startswith("patient_") and key not in ALLOWED_PATIENT_FIELDS:
            reasons.append(f"Field '{key}' exceeds minimum necessary PHI policy")
    compliant = not reasons
    return PHICheckResult(compliant=compliant, reasons=reasons, redacted_payload=redacted)


@app.post("/appeals/draft")
def create_draft(draft: AppealDraft) -> Dict:
    phi = perform_phi_check(draft)
    letter = render_denial_letter(draft)
    submission_stub = AppealSubmission(**draft.model_dump(), submitted_by="preview@denialrx.ai")
    email_preview = render_submission_email("PREVIEW", submission_stub)
    audit_service.append(
        action="draft",
        actor="system",
        appeal_id=draft.draft_id,
        payload=redacted_payload_for_audit(draft.model_dump()),
    )
    logger.info("Generated draft for appeal %s", draft.draft_id)
    return {
        "appeal_draft": draft.model_dump(),
        "letter": letter,
        "phi_check": phi.model_dump(),
        "email_preview": email_preview,
    }


@app.post("/appeals/submit", response_model=AppealStatus)
def submit_appeal(submission: AppealSubmission) -> AppealStatus:
    phi = perform_phi_check(submission)
    if not phi.compliant:
        logger.warning("Submission failed PHI check", {"draft_id": submission.draft_id, "reasons": phi.reasons})
        raise HTTPException(status_code=422, detail={"phi": phi.reasons})
    appeal_id = str(uuid4())
    timestamp = datetime.now(timezone.utc)
    APPEAL_STORE[appeal_id] = {
        "status": "submitted",
        "draft": submission.model_dump(),
        "updated_at": timestamp,
    }
    audit_service.append(
        action="submit",
        actor=submission.submitted_by,
        appeal_id=appeal_id,
        payload=redacted_payload_for_audit(submission.model_dump()),
    )
    logger.info("Appeal %s submitted", appeal_id)
    return AppealStatus(
        appeal_id=appeal_id,
        status="submitted",
        updated_at=timestamp,
        payload_hash=hash_payload(phi.redacted_payload),
    )


@app.get("/appeals/{appeal_id}", response_model=AppealStatus)
def get_status(appeal_id: str) -> AppealStatus:
    record = APPEAL_STORE.get(appeal_id)
    if not record:
        raise HTTPException(status_code=404, detail="Appeal not found")
    return AppealStatus(
        appeal_id=appeal_id,
        status=record["status"],
        updated_at=record["updated_at"],
        payload_hash=hash_payload(redacted_payload_for_audit(record["draft"])),
    )


@app.get("/appeals/{appeal_id}/audit")
def get_audit_trail(appeal_id: str) -> Dict:
    record = APPEAL_STORE.get(appeal_id)
    history: List[Dict] = [event.model_dump() for event in audit_service.history_for(appeal_id)]
    if record:
        draft_id = record["draft"].get("draft_id")
        if draft_id:
            history.extend(
                event.model_dump() for event in audit_service.history_for(draft_id)
            )
    if not history:
        raise HTTPException(status_code=404, detail="Appeal not found in audit")
    unique = {entry["payload_hash"]: entry for entry in history}
    ordered = sorted(unique.values(), key=lambda entry: entry["occurred_at"])
    return {"appeal_id": appeal_id, "events": ordered}


def redacted_payload_for_audit(payload: Dict) -> Dict:
    return redact_phi(payload)
