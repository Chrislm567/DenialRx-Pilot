# Phase Runner — Ingest & Normalize

This repository implements the Phase Runner ingest stack for X12 837/835 payloads.
It includes:

- **Normalized data models** for 837 claims and 835 remittances using Pydantic along
  with TypeScript types for parity under `/packages/x12`.
- **Error taxonomy** capturing normalization and payer specific failures.
- **Synthetic fixtures** used for integration tests and documentation examples.
- **COPY-based Postgres loader** capable of idempotent upserts and deduplication.
- **Deterministic PHI tokenization** utilities and log/OpenTelemetry redaction helpers.
- **`dq-check` CLI** for schema, duplicate, and payer-specific quality checks.
- **PHI scanner** wired for CI/local workflows in `scripts/phi_scan.py`.

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[tests]
```

### Running Tests

```bash
pytest
```

Integration tests rely on Docker to provision Postgres via `docker-compose.yml`.
Ensure Docker is running before executing the suite.

### Data Quality CLI

```bash
# Validate a JSONL file of normalized documents
python -m packages.dq.cli dq-check ./sample.jsonl --markdown report.md --json report.json
```

### PHI Scanner

```bash
scripts/phi_scan.py
```

### Coverage

Pytest is configured with coverage reporting and will display summary information
as part of the default test run.

## Repository Hygiene

- Commit messages follow the pattern `[ingest] 0.x <what> ✅`.
- Run `scripts/phi_scan.py` locally before committing.
- Fixtures under `packages/x12/fixtures.py` are synthetic and de-identified.
- CI executes unit/integration tests and the PHI scanner.
