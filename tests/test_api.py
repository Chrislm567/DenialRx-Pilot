from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import APPEAL_STORE, app
from app.models import DenialScenario


@pytest.fixture(autouse=True)
def reset_store(tmp_path):
    APPEAL_STORE.clear()
    audit_path = tmp_path / "audit.log"
    from app import main

    main.audit_service.path = Path(audit_path)
    yield
    APPEAL_STORE.clear()


@pytest.fixture
def client():
    return TestClient(app)


def base_payload():
    return {
        "draft_id": "API-1",
        "patient_first_name": "Jordan",
        "patient_last_name": "Morgan",
        "member_id": "MEM123",
        "payer_name": "Evergreen",
        "provider_npi": "1417111111",
        "denial_code": "D-404",
        "scenario": DenialScenario.lack_of_medical_necessity.value,
        "clinical_summary": "Therapy is clinically necessary and supported by chart notes.",
    }


def test_healthcheck(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    datetime.fromisoformat(data["timestamp"])


def test_draft_and_submit_flow(client):
    payload = base_payload()

    draft_response = client.post("/appeals/draft", json=payload)
    assert draft_response.status_code == 200
    draft_data = draft_response.json()
    assert draft_data["phi_check"]["compliant"] is True

    submission_payload = {**payload, "submitted_by": "nurse@denialrx.ai"}
    submit_response = client.post("/appeals/submit", json=submission_payload)
    assert submit_response.status_code == 200
    status_body = submit_response.json()
    appeal_id = status_body["appeal_id"]
    assert status_body["status"] == "submitted"
    assert len(status_body["payload_hash"]) == 64

    status_response = client.get(f"/appeals/{appeal_id}")
    assert status_response.status_code == 200
    assert status_response.json()["appeal_id"] == appeal_id

    audit_response = client.get(f"/appeals/{appeal_id}/audit")
    assert audit_response.status_code == 200
    events = audit_response.json()["events"]
    actions = {event["action"] for event in events}
    assert {"draft", "submit"}.issubset(actions)
