"""Error taxonomy representing normalization and data quality failures."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Category(str, Enum):
    STRUCTURE = "structure"
    CODING = "coding"
    ENRICHMENT = "enrichment"
    PAYER_QUIRK = "payer_quirk"
    PRIVACY = "privacy"


@dataclass(frozen=True)
class TaxonomyEntry:
    code: str
    category: Category
    severity: Severity
    message: str
    remediation: Optional[str] = None
    document_types: Optional[List[str]] = None


class ErrorTaxonomy:
    """Registry of known normalization errors."""

    def __init__(self, entries: Iterable[TaxonomyEntry]):
        self._by_code: Dict[str, TaxonomyEntry] = {}
        for entry in entries:
            self.add_entry(entry)

    def add_entry(self, entry: TaxonomyEntry) -> None:
        if entry.code in self._by_code:
            raise ValueError(f"Duplicate taxonomy code {entry.code}")
        if entry.document_types is not None:
            for doc_type in entry.document_types:
                if doc_type not in {"837", "835"}:
                    raise ValueError(f"Invalid document type {doc_type}")
        self._by_code[entry.code] = entry

    def get(self, code: str) -> TaxonomyEntry:
        try:
            return self._by_code[code]
        except KeyError as exc:  # pragma: no cover - helpful message
            raise KeyError(f"Unknown taxonomy code: {code}") from exc

    def group_by_category(self) -> Dict[Category, List[TaxonomyEntry]]:
        grouped: Dict[Category, List[TaxonomyEntry]] = {}
        for entry in self._by_code.values():
            grouped.setdefault(entry.category, []).append(entry)
        return grouped

    def for_document(self, document_type: str) -> List[TaxonomyEntry]:
        if document_type not in {"835", "837"}:
            raise ValueError("document_type must be '835' or '837'")
        results = []
        for entry in self._by_code.values():
            if entry.document_types is None or document_type in entry.document_types:
                results.append(entry)
        return sorted(results, key=lambda entry: entry.code)

    def codes(self) -> List[str]:
        return sorted(self._by_code)


DEFAULT_TAXONOMY = ErrorTaxonomy(
    entries=[
        TaxonomyEntry(
            code="X12-001",
            category=Category.STRUCTURE,
            severity=Severity.ERROR,
            message="Segment parsing failure",
            remediation="Inspect raw segment sequence and confirm delimiters",
        ),
        TaxonomyEntry(
            code="X12-005",
            category=Category.CODING,
            severity=Severity.WARNING,
            message="Diagnosis code failed ICD validation",
            remediation="Consult ICD-10 tables for the payer's effective date",
            document_types=["837"],
        ),
        TaxonomyEntry(
            code="X12-020",
            category=Category.ENRICHMENT,
            severity=Severity.WARNING,
            message="Missing subscriber address, enriched from eligibility feed",
        ),
        TaxonomyEntry(
            code="X12-041",
            category=Category.PAYER_QUIRK,
            severity=Severity.INFO,
            message="Payer requires zero-padded control numbers",
            document_types=["837"],
        ),
        TaxonomyEntry(
            code="X12-075",
            category=Category.PRIVACY,
            severity=Severity.CRITICAL,
            message="Unredacted PHI detected in free-text note",
            remediation="Run deterministic tokenization before persistence",
        ),
        TaxonomyEntry(
            code="X12-101",
            category=Category.STRUCTURE,
            severity=Severity.ERROR,
            message="835 claim payment missing service line detail",
            document_types=["835"],
        ),
        TaxonomyEntry(
            code="X12-150",
            category=Category.CODING,
            severity=Severity.ERROR,
            message="Procedure modifier mismatch with payer contract",
            document_types=["835", "837"],
        ),
        TaxonomyEntry(
            code="X12-212",
            category=Category.ENRICHMENT,
            severity=Severity.WARNING,
            message="Provider taxonomy code inferred from NPI registry",
        ),
        TaxonomyEntry(
            code="X12-310",
            category=Category.PAYER_QUIRK,
            severity=Severity.WARNING,
            message="Payer requires subscriber gender for COB",
            document_types=["837"],
        ),
        TaxonomyEntry(
            code="X12-411",
            category=Category.STRUCTURE,
            severity=Severity.CRITICAL,
            message="Envelope checksum mismatch",
        ),
    ]
)


__all__ = [
    "Category",
    "DEFAULT_TAXONOMY",
    "ErrorTaxonomy",
    "Severity",
    "TaxonomyEntry",
]
