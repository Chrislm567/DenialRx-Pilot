from pathlib import Path

import pytest
from PyPDF2 import PdfWriter

from app.services.evidence import EvidencePackError, bundle_evidence_pack


@pytest.fixture
def pdf_file(tmp_path: Path):
    def _create(name: str) -> Path:
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        path = tmp_path / name
        with path.open("wb") as handle:
            writer.write(handle)
        return path

    return _create


def test_bundle_evidence_pack(tmp_path: Path, pdf_file):
    first = pdf_file("lab1.pdf")
    second = pdf_file("lab2.pdf")

    output = tmp_path / "bundle.pdf"
    hashes = bundle_evidence_pack(output, [first, second])

    assert output.exists()
    assert len(hashes) == 3
    assert "bundle.pdf" in hashes
    assert all(len(digest) == 64 for digest in hashes.values())


def test_bundle_requires_pdfs(tmp_path: Path, pdf_file):
    pdf_path = pdf_file("clinical.pdf")
    txt_path = tmp_path / "notes.txt"
    txt_path.write_text("not a pdf")

    with pytest.raises(EvidencePackError):
        bundle_evidence_pack(tmp_path / "bundle.pdf", [pdf_path, txt_path])
