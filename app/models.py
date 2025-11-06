from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DenialScenario(str, Enum):
    lack_of_medical_necessity = "lack_of_medical_necessity"
    prior_auth_missing = "prior_auth_missing"
    out_of_network = "out_of_network"
    experimental_treatment = "experimental_treatment"
    coding_error = "coding_error"


class AppealDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    draft_id: str = Field(..., description="Client supplied draft identifier")
    patient_first_name: str
    patient_last_name: str
    member_id: str
    payer_name: str
    provider_npi: str
    denial_code: str
    scenario: DenialScenario
    clinical_summary: str
    attachments: List[str] = Field(default_factory=list)

    @field_validator("member_id", "provider_npi", "denial_code")
    @classmethod
    def enforce_alphanumeric(cls, value: str) -> str:
        allowed = value.replace("-", "")
        if not allowed.isalnum():
            raise ValueError("Identifiers must be alphanumeric or dash delimited")
        return value


class AppealSubmission(AppealDraft):
    submitted_by: str
    callback_url: Optional[str] = Field(
        default=None, description="Optional webhook for status updates"
    )


class AppealStatus(BaseModel):
    appeal_id: str
    status: str
    updated_at: datetime
    payload_hash: str


class AuditRecord(BaseModel):
    appeal_id: str
    action: str
    actor: str
    occurred_at: datetime
    payload_hash: str
    payload_schema: str


class PHICheckResult(BaseModel):
    compliant: bool
    reasons: List[str] = Field(default_factory=list)
    redacted_payload: Dict[str, Any] = Field(default_factory=dict)
