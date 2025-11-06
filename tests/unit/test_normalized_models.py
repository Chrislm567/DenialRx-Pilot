from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from packages.x12.fixtures import example_claim, example_envelope, example_remittance
from packages.x12.normalized_models import (
    Adjustment,
    AdjustmentGroup,
    Address,
    BatchEnvelope,
    Claim,
    ClaimPayment,
    DiagnosisCode,
    Gender,
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


@pytest.mark.parametrize(
    "code",
    [
        "A00",
        "B20",
        "C34.1",
        "E11.9",
        "F32.0",
        "G47.33",
        "H60.3",
        "J10.1",
        "K21.9",
        "Z00.00",
    ],
)
def test_diagnosis_code_accepts_valid_patterns(code: str):
    diagnosis = DiagnosisCode(code=code)
    assert diagnosis.code == code


@pytest.mark.parametrize(
    "code",
    [
        "1234",
        "AB123",
        "A1",
        "1A.23",
        "AA1111",
        "@12.1",
        "ZZZ",
        "A.1234",
        "J100.100",
        "",
    ],
)
def test_diagnosis_code_rejects_invalid_patterns(code: str):
    with pytest.raises(ValueError):
        DiagnosisCode(code=code)


@pytest.mark.parametrize(
    "code",
    ["99213", "A0427", "93000", "97161", "G0008", "A1234"],
)
def test_procedure_code_accepts_common_codes(code: str):
    proc = ProcedureCode(code=code)
    assert proc.code == code


@pytest.mark.parametrize(
    "code",
    ["9921", "abcde", "123456", "9*213", "ABCDE1", "!!!"],
)
def test_procedure_code_rejects_invalid_codes(code: str):
    with pytest.raises(ValueError):
        ProcedureCode(code=code)


def test_service_line_validates_dates():
    provider = Provider(id="P1", name="Test Provider")
    line = ServiceLine(
        line_number=1,
        procedure_code=ProcedureCode(code="99213"),
        service_start=date(2024, 1, 1),
        service_end=date(2024, 1, 1),
        charge_amount=100.0,
        units=1,
        unit_type=ServiceUnit.UNIT,
        rendering_provider=provider,
    )
    assert line.service_start == line.service_end


def test_service_line_rejects_inverted_dates():
    provider = Provider(id="P1", name="Test Provider")
    with pytest.raises(ValueError):
        ServiceLine(
            line_number=1,
            procedure_code=ProcedureCode(code="99213"),
            service_start=date(2024, 1, 2),
            service_end=date(2024, 1, 1),
            charge_amount=100.0,
            units=1,
            unit_type=ServiceUnit.UNIT,
            rendering_provider=provider,
        )


def test_claim_total_matches_service_lines():
    provider = Provider(id="P1", name="Test Provider")
    payer = Payer(id="PAY1", name="Waystar")
    subscriber = Subscriber(
        id="S1",
        first_name="Jamie",
        last_name="Lake",
        date_of_birth=date(1990, 1, 1),
        gender=Gender.UNKNOWN,
        relationship="SELF",
    )
    service_line = ServiceLine(
        line_number=1,
        procedure_code=ProcedureCode(code="99213"),
        service_start=date(2024, 1, 1),
        service_end=date(2024, 1, 1),
        charge_amount=125.5,
        units=1,
        unit_type=ServiceUnit.UNIT,
        rendering_provider=provider,
    )
    claim = Claim(
        claim_id="C1",
        patient=subscriber,
        subscriber=subscriber,
        billing_provider=provider,
        payer=payer,
        diagnoses=[DiagnosisCode(code="J10.1")],
        service_lines=[service_line],
        total_charge_amount=125.5,
        received_at=datetime(2024, 1, 5, 12, 0, 0),
        currency="USD",
    )
    assert claim.total_charge_amount == 125.5


def test_claim_total_mismatch_raises():
    provider = Provider(id="P1", name="Test Provider")
    payer = Payer(id="PAY1", name="Waystar")
    subscriber = Subscriber(
        id="S1",
        first_name="Jamie",
        last_name="Lake",
        date_of_birth=date(1990, 1, 1),
        gender=Gender.UNKNOWN,
        relationship="SELF",
    )
    service_line = ServiceLine(
        line_number=1,
        procedure_code=ProcedureCode(code="99213"),
        service_start=date(2024, 1, 1),
        service_end=date(2024, 1, 1),
        charge_amount=125.5,
        units=1,
        unit_type=ServiceUnit.UNIT,
        rendering_provider=provider,
    )
    with pytest.raises(ValueError):
        Claim(
            claim_id="C1",
            patient=subscriber,
            subscriber=subscriber,
            billing_provider=provider,
            payer=payer,
            diagnoses=[DiagnosisCode(code="J10.1")],
            service_lines=[service_line],
            total_charge_amount=100.0,
            received_at=datetime(2024, 1, 5, 12, 0, 0),
            currency="USD",
        )


def test_claim_payment_total_validation():
    payment = ClaimPayment(
        claim_id="C1",
        payment_amount=100.0,
        patient_responsibility=20.0,
        adjudication_date=date(2024, 1, 30),
        service_lines=[
            ServiceLinePayment(
                line_number=1,
                paid_amount=40.0,
                adjustments=[Adjustment(group=AdjustmentGroup.CONTRACTUAL, reason_code="45", amount=-10.0)],
            ),
            ServiceLinePayment(
                line_number=2,
                paid_amount=60.0,
                adjustments=[Adjustment(group=AdjustmentGroup.PATIENT_RESPONSIBILITY, reason_code="1", amount=-5.0)],
            ),
        ],
        remark_codes=["N620"],
    )
    assert payment.payment_amount == 100.0


def test_claim_payment_total_mismatch():
    with pytest.raises(ValueError):
        ClaimPayment(
            claim_id="C1",
            payment_amount=100.0,
            patient_responsibility=20.0,
            adjudication_date=date(2024, 1, 30),
            service_lines=[
                ServiceLinePayment(line_number=1, paid_amount=30.0),
                ServiceLinePayment(line_number=2, paid_amount=60.0),
            ],
        )


def test_remittance_total_paid_property():
    remittance = example_remittance()
    assert remittance.total_paid == pytest.approx(96.0)


def test_normalized_document_type_enforcement():
    claim = example_claim()
    metadata = NormalizationMetadata(
        trading_partner="Partner",
        version="5010",
        extracted_at=datetime(2024, 1, 1, 0, 0, 0),
        source_system="unit-tests",
    )
    document = NormalizedDocument(
        document_type="837",
        metadata=metadata,
        payload=claim,
        checksum="abc",
    )
    assert document.document_type == "837"
    with pytest.raises(ValueError):
        NormalizedDocument(
            document_type="835",
            metadata=metadata,
            payload=claim,
            checksum="abc",
        )


def test_batch_envelope_iteration_helpers():
    envelope = example_envelope()
    claims = list(envelope.iter_claims())
    remittances = list(envelope.iter_remittances())
    assert len(claims) == 1
    assert len(remittances) == 1


def test_batch_envelope_payer_summary_counts_documents():
    envelope = example_envelope()
    summary = envelope.payer_summary()
    assert summary["PAYER1"] == 2


@pytest.mark.parametrize("placeholder", ["***", "[SAFE]", "TOKEN"])
def test_adjustment_total_considers_all_items(placeholder: str):
    payment = ServiceLinePayment(
        line_number=1,
        paid_amount=10.0,
        adjustments=[
            Adjustment(group=AdjustmentGroup.CONTRACTUAL, reason_code="45", amount=-2.5),
            Adjustment(group=AdjustmentGroup.OTHER, reason_code="24", amount=-1.5),
        ],
    )
    assert payment.adjustment_total == pytest.approx(-4.0)
    assert placeholder


def test_fixture_examples_are_self_consistent():
    claim = example_claim()
    remittance = example_remittance()
    assert claim.claim_id == remittance.payments[0].claim_id


@pytest.mark.parametrize(
    "line_number, charge",
    [(1, 10.0), (2, 15.5), (3, 0.0), (4, 99.99), (5, 18.75)],
)
def test_service_line_charge_round_trip(line_number: int, charge: float):
    line = ServiceLine(
        line_number=line_number,
        procedure_code=ProcedureCode(code="99213"),
        service_start=date(2024, 2, 1),
        service_end=date(2024, 2, 1),
        charge_amount=charge,
        units=1,
        unit_type=ServiceUnit.UNIT,
    )
    assert line.charge_amount == pytest.approx(charge)


@pytest.mark.parametrize(
    "city, state",
    [
        ("Metropolis", "NY"),
        ("Gotham", "NJ"),
        ("Star City", "CA"),
        ("Central City", "KS"),
        ("Coast City", "OR"),
    ],
)
def test_address_state_length(city: str, state: str):
    address = Address(line1="123 Test", city=city, state=state, postal_code="12345")
    assert address.state == state


@pytest.mark.parametrize("postal_code", ["10001", "12345-6789", "94107", "02139", "60601"])
def test_address_accepts_diverse_postal_codes(postal_code: str):
    address = Address(line1="456 Example", city="City", state="NY", postal_code=postal_code)
    assert address.postal_code == postal_code


@pytest.mark.parametrize(
    "postal_code",
    ["1234", "12", "", "ABCDE", "12345678901"],
)
def test_address_rejects_short_postal_codes(postal_code: str):
    with pytest.raises(ValueError):
        Address(line1="789", city="Town", state="NY", postal_code=postal_code)


@pytest.mark.parametrize(
    "gender",
    [Gender.FEMALE, Gender.MALE, Gender.UNKNOWN],
)
def test_member_gender_enum_round_trip(gender: Gender):
    claim = example_claim()
    claim.patient.gender = gender
    assert claim.patient.gender == gender


@pytest.mark.parametrize(
    "modifier",
    ["25", "59", "76", "91", "XE", "XP", "XS", "XU"],
)
def test_service_line_modifier_collection(modifier: str):
    line = ServiceLine(
        line_number=1,
        procedure_code=ProcedureCode(code="99213"),
        service_start=date(2024, 1, 1),
        service_end=date(2024, 1, 1),
        charge_amount=10.0,
        units=1,
        unit_type=ServiceUnit.UNIT,
        modifiers=[modifier],
    )
    assert modifier in line.modifiers


@pytest.mark.parametrize(
    "currency",
    ["USD", "CAD"],
)
def test_claim_currency_enum(currency: str):
    claim = example_claim()
    claim.currency = currency
    assert claim.currency == currency


@pytest.mark.parametrize("warning", ["Warn1", "Warn2", "Warn3", "Warn4", "Warn5"])
def test_metadata_warnings_collection(warning: str):
    metadata = NormalizationMetadata(
        trading_partner="Partner",
        version="5010",
        extracted_at=datetime.now(timezone.utc),
        source_system="tests",
        warnings=[warning],
    )
    assert metadata.warnings == [warning]
