from __future__ import annotations

import pytest

from packages.x12.error_taxonomy import Category, DEFAULT_TAXONOMY, ErrorTaxonomy, Severity, TaxonomyEntry


@pytest.mark.parametrize("code", DEFAULT_TAXONOMY.codes())
def test_default_taxonomy_contains_unique_codes(code: str):
    entry = DEFAULT_TAXONOMY.get(code)
    assert entry.code == code


@pytest.mark.parametrize(
    "category, expected",
    [
        (Category.STRUCTURE, {"X12-001", "X12-101", "X12-411"}),
        (Category.CODING, {"X12-005", "X12-150"}),
        (Category.ENRICHMENT, {"X12-020", "X12-212"}),
        (Category.PAYER_QUIRK, {"X12-041", "X12-310"}),
        (Category.PRIVACY, {"X12-075"}),
    ],
)
def test_group_by_category_matches_expected(category: Category, expected: set[str]):
    grouped = DEFAULT_TAXONOMY.group_by_category()
    assert expected.issubset({entry.code for entry in grouped[category]})


@pytest.mark.parametrize("doc_type", ["835", "837"])
def test_for_document_filters_by_type(doc_type: str):
    entries = DEFAULT_TAXONOMY.for_document(doc_type)
    assert all(entry.document_types is None or doc_type in entry.document_types for entry in entries)


def test_error_taxonomy_rejects_duplicate_codes():
    taxonomy = ErrorTaxonomy(entries=[
        TaxonomyEntry(
            code="TEST-1",
            category=Category.STRUCTURE,
            severity=Severity.INFO,
            message="test",
        )
    ])
    with pytest.raises(ValueError):
        taxonomy.add_entry(
            TaxonomyEntry(
                code="TEST-1",
                category=Category.STRUCTURE,
                severity=Severity.INFO,
                message="duplicate",
            )
        )


@pytest.mark.parametrize("invalid_type", ["999", "abc", "", "A37"])
def test_for_document_rejects_invalid_type(invalid_type: str):
    with pytest.raises(ValueError):
        DEFAULT_TAXONOMY.for_document(invalid_type)


@pytest.mark.parametrize(
    "severity",
    [Severity.INFO, Severity.WARNING, Severity.ERROR, Severity.CRITICAL],
)
def test_severity_enum_round_trip(severity: Severity):
    entry = TaxonomyEntry(
        code=f"TEST-{severity.value}",
        category=Category.STRUCTURE,
        severity=severity,
        message="test",
    )
    assert entry.severity == severity


@pytest.mark.parametrize(
    "category",
    [Category.STRUCTURE, Category.CODING, Category.ENRICHMENT, Category.PAYER_QUIRK, Category.PRIVACY],
)
def test_category_enum_round_trip(category: Category):
    entry = TaxonomyEntry(
        code=f"TEST-{category.value}",
        category=category,
        severity=Severity.INFO,
        message="test",
    )
    assert entry.category == category


@pytest.mark.parametrize(
    "doc_types",
    [["835"], ["837"], ["835", "837"], None],
)
def test_document_type_validation(doc_types: list[str] | None):
    entry = TaxonomyEntry(
        code="TEST-VALID",
        category=Category.STRUCTURE,
        severity=Severity.INFO,
        message="test",
        document_types=doc_types,
    )
    taxonomy = ErrorTaxonomy(entries=[entry])
    assert taxonomy.get("TEST-VALID").document_types == doc_types


def test_invalid_document_type_rejected():
    with pytest.raises(ValueError):
        ErrorTaxonomy(
            entries=[
                TaxonomyEntry(
                    code="TEST",
                    category=Category.STRUCTURE,
                    severity=Severity.INFO,
                    message="test",
                    document_types=["999"],
                )
            ]
        )
