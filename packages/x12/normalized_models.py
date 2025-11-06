"""Normalized 837/835 models.

These models provide a canonical representation across trading partners.
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Dict, Iterable, List, Optional

from pydantic import BaseModel, Field, model_validator


class Gender(str, Enum):
    """Simplified gender codes."""

    FEMALE = "F"
    MALE = "M"
    UNKNOWN = "U"


class Currency(str, Enum):
    USD = "USD"
    CAD = "CAD"


class ServiceUnit(str, Enum):
    DAY = "DAY"
    UNIT = "UNIT"
    VISIT = "VISIT"


class AdjustmentGroup(str, Enum):
    CONTRACTUAL = "CO"
    PATIENT_RESPONSIBILITY = "PR"
    PAYOR_INITIATED = "PI"
    OTHER = "OA"


class NetworkStatus(str, Enum):
    IN_NETWORK = "IN"
    OUT_OF_NETWORK = "OUT"


class Contact(BaseModel):
    name: str
    phone: Optional[str] = None
    email: Optional[str] = None


class Address(BaseModel):
    line1: str
    line2: Optional[str] = None
    city: str
    state: str = Field(min_length=2, max_length=2)
    postal_code: str = Field(pattern=r"^\d{5}(?:-\d{4})?$")


class Payer(BaseModel):
    id: str = Field(description="Unique payer identifier")
    name: str
    contact: Optional[Contact] = None


class Provider(BaseModel):
    id: str
    npi: Optional[str] = Field(default=None, min_length=10, max_length=10)
    tax_id: Optional[str] = Field(default=None, min_length=9, max_length=9)
    name: str
    address: Optional[Address] = None
    network_status: NetworkStatus = NetworkStatus.IN_NETWORK


class Member(BaseModel):
    id: str
    first_name: str
    last_name: str
    date_of_birth: date
    gender: Gender = Gender.UNKNOWN
    address: Optional[Address] = None


class Subscriber(Member):
    relationship: str = Field(description="Relationship of patient to subscriber")


class DiagnosisCode(BaseModel):
    code: str = Field(pattern=r"^[A-TV-Z][0-9][0-9A-Z][A-Z0-9\.]{0,4}$")
    description: Optional[str] = None


class ProcedureCode(BaseModel):
    code: str = Field(pattern=r"^[A-Z0-9]{5}$")
    description: Optional[str] = None


class ServiceLine(BaseModel):
    line_number: int
    procedure_code: ProcedureCode
    modifiers: List[str] = Field(default_factory=list)
    diagnosis_pointers: List[int] = Field(default_factory=list, description="Pointers to diagnosis indices")
    service_start: date
    service_end: date
    charge_amount: float
    units: float = 1.0
    unit_type: ServiceUnit = ServiceUnit.UNIT
    rendering_provider: Optional[Provider] = None

    @model_validator(mode="after")
    def validate_dates(cls, values: "ServiceLine") -> "ServiceLine":
        if values.service_end < values.service_start:
            raise ValueError("service_end cannot precede service_start")
        return values


class Claim(BaseModel):
    claim_id: str
    patient: Member
    subscriber: Subscriber
    billing_provider: Provider
    payer: Payer
    diagnoses: List[DiagnosisCode] = Field(default_factory=list)
    service_lines: List[ServiceLine] = Field(default_factory=list)
    total_charge_amount: float
    received_at: datetime
    currency: Currency = Currency.USD
    control_number: Optional[str] = None

    @model_validator(mode="after")
    def validate_totals(cls, values: "Claim") -> "Claim":
        line_total = sum(line.charge_amount for line in values.service_lines)
        if values.service_lines and round(line_total, 2) != round(values.total_charge_amount, 2):
            raise ValueError(
                "total_charge_amount must equal sum of service line charge_amounts"
            )
        return values


class Adjustment(BaseModel):
    group: AdjustmentGroup
    reason_code: str
    amount: float


class ServiceLinePayment(BaseModel):
    line_number: int
    paid_amount: float
    allowed_amount: Optional[float] = None
    adjustments: List[Adjustment] = Field(default_factory=list)

    @property
    def adjustment_total(self) -> float:
        return round(sum(adj.amount for adj in self.adjustments), 2)


class ClaimPayment(BaseModel):
    claim_id: str
    payment_amount: float
    patient_responsibility: float
    adjudication_date: date
    service_lines: List[ServiceLinePayment] = Field(default_factory=list)
    remark_codes: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_payment(cls, values: "ClaimPayment") -> "ClaimPayment":
        total = sum(line.paid_amount for line in values.service_lines)
        if values.service_lines and round(total, 2) != round(values.payment_amount, 2):
            raise ValueError("payment_amount must equal sum of service line paid_amount")
        return values


class Remittance(BaseModel):
    remittance_id: str
    payer: Payer
    payee: Provider
    payments: List[ClaimPayment] = Field(default_factory=list)
    payment_issue_date: date
    check_number: Optional[str] = None
    currency: Currency = Currency.USD

    @property
    def total_paid(self) -> float:
        return round(sum(payment.payment_amount for payment in self.payments), 2)


class NormalizationMetadata(BaseModel):
    trading_partner: str
    version: str
    extracted_at: datetime
    source_system: str
    warnings: List[str] = Field(default_factory=list)


class NormalizedDocument(BaseModel):
    """Wrapper linking normalized payloads and metadata."""

    document_type: str = Field(pattern="^(837|835)$")
    metadata: NormalizationMetadata
    payload: Claim | Remittance
    checksum: str

    @model_validator(mode="after")
    def validate_type_matches_payload(cls, values: "NormalizedDocument") -> "NormalizedDocument":
        payload = values.payload
        if values.document_type == "837" and not isinstance(payload, Claim):
            raise ValueError("837 document requires Claim payload")
        if values.document_type == "835" and not isinstance(payload, Remittance):
            raise ValueError("835 document requires Remittance payload")
        return values


class BatchEnvelope(BaseModel):
    batch_id: str
    documents: List[NormalizedDocument]
    generated_at: datetime

    def iter_claims(self) -> Iterable[Claim]:
        for doc in self.documents:
            if isinstance(doc.payload, Claim):
                yield doc.payload

    def iter_remittances(self) -> Iterable[Remittance]:
        for doc in self.documents:
            if isinstance(doc.payload, Remittance):
                yield doc.payload

    def payer_summary(self) -> Dict[str, int]:
        summary: Dict[str, int] = {}
        for doc in self.documents:
            payer_id = doc.payload.payer.id if isinstance(doc.payload, Claim) else doc.payload.payer.id
            summary[payer_id] = summary.get(payer_id, 0) + 1
        return summary


__all__ = [
    "Adjustment",
    "AdjustmentGroup",
    "Address",
    "BatchEnvelope",
    "Claim",
    "ClaimPayment",
    "Contact",
    "Currency",
    "DiagnosisCode",
    "Gender",
    "Member",
    "NetworkStatus",
    "NormalizationMetadata",
    "NormalizedDocument",
    "Payer",
    "Provider",
    "ProcedureCode",
    "Remittance",
    "ServiceLine",
    "ServiceLinePayment",
    "ServiceUnit",
    "Subscriber",
]
