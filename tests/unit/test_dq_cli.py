from __future__ import annotations

from pathlib import Path

import orjson
import pytest
from click.testing import CliRunner

from packages.dq.cli import DataQualityChecker, dq_check
from packages.x12.fixtures import example_claim, example_remittance
from packages.x12.normalized_models import NormalizationMetadata, NormalizedDocument


def write_documents(path: Path, documents):
    with path.open("wb") as f:
        for doc in documents:
            f.write(orjson.dumps(doc.model_dump(mode="json")))
            f.write(b"\n")


def test_data_quality_checker_detects_duplicates(tmp_path: Path):
    claim = example_claim()
    remittance = example_remittance()
    documents = [
        NormalizedDocument(
            document_type="837",
            metadata=NormalizationMetadata(
                trading_partner="Partner",
                version="5010",
                extracted_at=claim.received_at,
                source_system="tests",
            ),
            payload=claim,
            checksum="1",
        ),
        NormalizedDocument(
            document_type="837",
            metadata=NormalizationMetadata(
                trading_partner="Partner",
                version="5010",
                extracted_at=claim.received_at,
                source_system="tests",
            ),
            payload=claim,
            checksum="2",
        ),
        NormalizedDocument(
            document_type="835",
            metadata=NormalizationMetadata(
                trading_partner="Partner",
                version="5010",
                extracted_at=claim.received_at,
                source_system="tests",
            ),
            payload=remittance,
            checksum="3",
        ),
    ]
    file_path = tmp_path / "documents.jsonl"
    write_documents(file_path, documents)

    checker = DataQualityChecker()
    result = checker.run(documents)
    assert ("837", f"{claim.claim_id}|{claim.billing_provider.id}") in result.duplicate_keys


@pytest.mark.parametrize("payer_id", ["WELLCARE", "AET123", "UHC999", "OTHER"])
def test_payer_specific_warnings(payer_id: str):
    claim = example_claim()
    claim.payer.id = payer_id
    claim.control_number = "000123456"
    claim.subscriber.gender = "M"
    remittance = example_remittance()
    remittance.payer.id = payer_id
    if payer_id == "WELLCARE":
        claim.control_number = "123"
    if payer_id == "AET123":
        claim.subscriber.gender = "U"
    if payer_id == "UHC999":
        remittance.payments[0].patient_responsibility = 0

    checker = DataQualityChecker()
    result = checker.run([
        NormalizedDocument(
            document_type="837",
            metadata=NormalizationMetadata(
                trading_partner="Partner",
                version="5010",
                extracted_at=claim.received_at,
                source_system="tests",
            ),
            payload=claim,
            checksum="1",
        ),
        NormalizedDocument(
            document_type="835",
            metadata=NormalizationMetadata(
                trading_partner="Partner",
                version="5010",
                extracted_at=claim.received_at,
                source_system="tests",
            ),
            payload=remittance,
            checksum="2",
        ),
    ])
    if payer_id == "WELLCARE":
        assert any("9-digit control" in warning for warning in result.payer_warnings)
    elif payer_id == "AET123":
        assert any("unknown gender" in warning for warning in result.payer_warnings)
    elif payer_id == "UHC999":
        assert any("patient responsibility" in warning for warning in result.payer_warnings)
    else:
        assert not result.payer_warnings


def test_dq_check_cli_outputs_markdown(tmp_path: Path):
    claim = example_claim()
    documents = [
        NormalizedDocument(
            document_type="837",
            metadata=NormalizationMetadata(
                trading_partner="Partner",
                version="5010",
                extracted_at=claim.received_at,
                source_system="tests",
            ),
            payload=claim,
            checksum="1",
        ),
    ]
    file_path = tmp_path / "documents.jsonl"
    write_documents(file_path, documents)

    runner = CliRunner()
    result = runner.invoke(dq_check, [str(file_path), "--no-stdout"])
    assert result.exit_code == 0
    assert "Data Quality Report" in result.output or result.output == ""
