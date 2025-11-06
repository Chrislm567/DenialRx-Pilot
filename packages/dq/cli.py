"""CLI for running schema and duplicate checks on normalized payloads."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import click
import orjson

from packages.x12.error_taxonomy import DEFAULT_TAXONOMY
from packages.x12.normalized_models import Claim, NormalizedDocument, Remittance


@dataclass
class CheckResult:
    schema_errors: List[str]
    duplicate_keys: List[Tuple[str, str]]
    payer_warnings: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_errors": self.schema_errors,
            "duplicate_keys": [list(item) for item in self.duplicate_keys],
            "payer_warnings": self.payer_warnings,
        }


class DataQualityChecker:
    def __init__(self):
        self.taxonomy = DEFAULT_TAXONOMY

    def load_documents(self, path: Path) -> Iterable[NormalizedDocument]:
        for line in path.read_text().splitlines():
            payload = orjson.loads(line)
            yield NormalizedDocument.model_validate(payload)

    def run(self, documents: Iterable[NormalizedDocument]) -> CheckResult:
        schema_errors: List[str] = []
        duplicate_keys: List[Tuple[str, str]] = []
        payer_warnings: List[str] = []
        seen_claims: Dict[Tuple[str, str], int] = defaultdict(int)
        seen_remittances: Dict[str, int] = defaultdict(int)
        payer_counter: Counter[str] = Counter()

        for doc in documents:
            payer_id = self._resolve_payer_id(doc)
            payer_counter[payer_id] += 1
            try:
                self._validate_payload(doc)
            except ValueError as exc:
                schema_errors.append(f"{doc.document_type}:{exc}")

            if isinstance(doc.payload, Claim):
                composite = (doc.payload.claim_id, doc.payload.billing_provider.id)
                seen_claims[composite] += 1
                if seen_claims[composite] > 1:
                    duplicate_keys.append(("837", f"{composite[0]}|{composite[1]}"))
                payer_warnings.extend(self._payer_quirks_claim(doc.payload))
            elif isinstance(doc.payload, Remittance):
                seen_remittances[doc.payload.remittance_id] += 1
                if seen_remittances[doc.payload.remittance_id] > 1:
                    duplicate_keys.append(("835", doc.payload.remittance_id))
                payer_warnings.extend(self._payer_quirks_remit(doc.payload))

        payer_warnings.extend(self._summarize_payer_volume(payer_counter))
        return CheckResult(schema_errors, duplicate_keys, payer_warnings)

    def _resolve_payer_id(self, doc: NormalizedDocument) -> str:
        if isinstance(doc.payload, Claim):
            return doc.payload.payer.id
        if isinstance(doc.payload, Remittance):
            return doc.payload.payer.id
        return "unknown"

    def _validate_payload(self, doc: NormalizedDocument) -> None:
        # Additional validation beyond pydantic
        if doc.document_type == "837":
            if not doc.payload.diagnoses:
                raise ValueError("837 claims require at least one diagnosis code")
        if doc.document_type == "835":
            if not doc.payload.payments:
                raise ValueError("835 remittances require claim payments")

    def _payer_quirks_claim(self, claim: Claim) -> List[str]:
        warnings: List[str] = []
        if claim.payer.id.startswith("WELL") and (not claim.control_number or len(claim.control_number) != 9):
            warnings.append("WELL* payers require 9-digit control numbers")
        if claim.payer.id.startswith("AET") and claim.subscriber.gender == "U":
            warnings.append("AET* payers reject unknown gender for COB")
        return warnings

    def _payer_quirks_remit(self, remittance: Remittance) -> List[str]:
        warnings: List[str] = []
        if remittance.payer.id.startswith("UHC"):
            for payment in remittance.payments:
                if payment.patient_responsibility <= 0:
                    warnings.append("UHC remittances expect patient responsibility > 0")
                    break
        return warnings

    def _summarize_payer_volume(self, counts: Counter[str]) -> List[str]:
        total = sum(counts.values())
        if total < 5:
            return []
        dominant = counts.most_common(1)[0]
        percent = (dominant[1] / total) * 100
        if percent > 75:
            return [
                f"Volume skew: payer {dominant[0]} accounts for {percent:.1f}% of batch",
            ]
        return []

    def render_markdown(self, result: CheckResult) -> str:
        lines = ["# Data Quality Report", ""]
        lines.append(f"*Schema errors:* {len(result.schema_errors)}")
        for error in result.schema_errors:
            lines.append(f"  - {error}")
        lines.append("")
        lines.append(f"*Duplicate keys:* {len(result.duplicate_keys)}")
        for doc_type, key in result.duplicate_keys:
            lines.append(f"  - {doc_type} -> {key}")
        lines.append("")
        lines.append(f"*Payer warnings:* {len(result.payer_warnings)}")
        for warning in result.payer_warnings:
            lines.append(f"  - {warning}")
        lines.append("")
        lines.append("Known taxonomy codes:")
        for code in self.taxonomy.codes():
            lines.append(f"  - {code}")
        return "\n".join(lines)


@click.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--markdown", "markdown_path", type=click.Path(path_type=Path))
@click.option("--json", "json_path", type=click.Path(path_type=Path))
@click.option("--stdout/--no-stdout", default=True, help="Emit markdown to stdout")
def dq_check(input_path: Path, markdown_path: Path | None, json_path: Path | None, stdout: bool) -> None:
    checker = DataQualityChecker()
    documents = list(checker.load_documents(input_path))
    result = checker.run(documents)
    if markdown_path:
        markdown_path.write_text(checker.render_markdown(result))
    if json_path:
        json_path.write_text(json.dumps(result.as_dict(), indent=2))
    if stdout:
        click.echo(checker.render_markdown(result))
    if result.schema_errors:
        sys.exit(1)


__all__ = ["DataQualityChecker", "dq_check", "CheckResult"]
