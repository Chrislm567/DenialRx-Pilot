"""Synthetic fixtures for normalized claim and remittance documents."""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Dict, List

from .normalized_models import (
    Adjustment,
    AdjustmentGroup,
    Address,
    BatchEnvelope,
    Claim,
    ClaimPayment,
    Contact,
    DiagnosisCode,
    Gender,
    Member,
    NetworkStatus,
    NormalizationMetadata,
    NormalizedDocument,
    Payer,
    ProcedureCode,
    Provider,
    Remittance,
    ServiceLine,
    ServiceLinePayment,
    ServiceUnit,
    Subscriber,
)


def example_claim(claim_id: str = "CLM12345", payer_id: str = "PAYER1") -> Claim:
    patient_address = Address(
        line1="123 Main St",
        city="Metropolis",
        state="NY",
        postal_code="10001",
    )
    subscriber = Subscriber(
        id="SUB123",
        first_name="Alex",
        last_name="Rivera",
        date_of_birth=date(1990, 4, 3),
        gender=Gender.UNKNOWN,
        address=patient_address,
        relationship="SELF",
    )
    patient = Member(
        id="MEM123",
        first_name="Alex",
        last_name="Rivera",
        date_of_birth=date(1990, 4, 3),
        gender=Gender.UNKNOWN,
        address=patient_address,
    )
    payer = Payer(id=payer_id, name="Waystar Health", contact=Contact(name="EDI Ops"))
    billing_provider = Provider(
        id="PRV123",
        npi="1234567890",
        tax_id="987654321",
        name="Rivera Primary Care",
        network_status=NetworkStatus.IN_NETWORK,
    )
    service_line = ServiceLine(
        line_number=1,
        procedure_code=ProcedureCode(code="99213", description="Office visit"),
        diagnosis_pointers=[1],
        modifiers=["25"],
        service_start=date(2024, 1, 15),
        service_end=date(2024, 1, 15),
        charge_amount=120.0,
        units=1,
        unit_type=ServiceUnit.UNIT,
        rendering_provider=billing_provider,
    )
    claim = Claim(
        claim_id=claim_id,
        patient=patient,
        subscriber=subscriber,
        billing_provider=billing_provider,
        payer=payer,
        diagnoses=[DiagnosisCode(code="J10.1", description="Flu")],
        service_lines=[service_line],
        total_charge_amount=120.0,
        received_at=datetime(2024, 1, 20, 15, 30, 0),
        currency="USD",
        control_number="000123456",
    )
    return claim


def example_remittance(remittance_id: str = "RMT12345", payer_id: str = "PAYER1") -> Remittance:
    payer = Payer(id=payer_id, name="Waystar Health")
    payee = Provider(id="PRV123", name="Rivera Primary Care", npi="1234567890")
    service_payment = ServiceLinePayment(
        line_number=1,
        paid_amount=96.0,
        allowed_amount=100.0,
        adjustments=[
            Adjustment(group=AdjustmentGroup.CONTRACTUAL, reason_code="45", amount=-20.0),
            Adjustment(group=AdjustmentGroup.PATIENT_RESPONSIBILITY, reason_code="1", amount=-4.0),
        ],
    )
    claim_payment = ClaimPayment(
        claim_id="CLM12345",
        payment_amount=96.0,
        patient_responsibility=24.0,
        adjudication_date=date(2024, 1, 25),
        service_lines=[service_payment],
        remark_codes=["N620"],
    )
    remittance = Remittance(
        remittance_id=remittance_id,
        payer=payer,
        payee=payee,
        payments=[claim_payment],
        payment_issue_date=date(2024, 1, 26),
        check_number="1234567",
        currency="USD",
    )
    return remittance


def example_envelope(batch_id: str = "BATCH1") -> BatchEnvelope:
    claim = example_claim()
    remittance = example_remittance()
    documents = [
        NormalizedDocument(
            document_type="837",
            metadata=NormalizationMetadata(
                trading_partner="Acme Clearinghouse",
                version="5010",
                extracted_at=datetime(2024, 1, 20, 15, 30, 0),
                source_system="phase-runner",
                warnings=["Control number padded"],
            ),
            payload=claim,
            checksum="abc123",
        ),
        NormalizedDocument(
            document_type="835",
            metadata=NormalizationMetadata(
                trading_partner="Acme Clearinghouse",
                version="5010",
                extracted_at=datetime(2024, 1, 26, 12, 0, 0),
                source_system="phase-runner",
            ),
            payload=remittance,
            checksum="def456",
        ),
    ]
    return BatchEnvelope(batch_id=batch_id, documents=documents, generated_at=datetime.now(timezone.utc))


def fixture_matrix() -> Dict[str, List[str]]:
    """High-level overview of fixture availability for smoke testing."""

    return {
        "claim": ["example_claim"],
        "remittance": ["example_remittance"],
        "envelope": ["example_envelope"],
    }


__all__ = [
    "example_claim",
    "example_envelope",
    "example_remittance",
    "fixture_matrix",
]
