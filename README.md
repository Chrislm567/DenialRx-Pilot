# DenialRx Appeals Automation

Phase runner implementation for payer appeals, including templating, API, SDK stubs, and audit guardrails.

## Quickstart

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The OpenAPI contract is published at [`openapi.yaml`](openapi.yaml).

## Tests

```bash
pytest
```

## Tooling
- Evidence packs via `app.services.evidence.bundle_evidence_pack`
- Templates under `app/templates/appeals`
- SDKs in `sdks/`
- Ops handbook in `docs/ops.md`
