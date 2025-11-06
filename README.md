# Phase Runner Rules Engine

Phase Runner is a deterministic rules engine built for revenue cycle
management (RCM) experimentation. The engine loads payer policy rule packs
written in YAML, validates them against a JSON Schema, and produces consistent
execution decisions suitable for unit testing and CLI automation.

## Features

- YAML rule packs validated against a JSON Schema before execution
- Deterministic executor with dry-run support and versioned metadata
- CLI utilities to validate, run, diff, and bulk-validate rule packs
- Extensive unit tests (>100) with golden fixtures for sample payer policies

## Getting Started

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Validate one of the sample rule packs:

```bash
python -m phase_runner.cli validate examples/rule_packs/sunrise_health.yaml
```

Execute the pack in dry-run mode against an example claim:

```bash
python -m phase_runner.cli run examples/rule_packs/sunrise_health.yaml \
  examples/inputs/patient_a.yaml --dry-run --as-json
```

Diff two rule pack versions:

```bash
python -m phase_runner.cli diff examples/rule_packs/opencare_value.yaml \
  examples/rule_packs/opencare_value.yaml --as-json
```

## Adding a New Payer

1. Review `docs/standards/rcm_rules_checklist.md` for policy inspiration.
2. Create a new YAML file under `examples/rule_packs/` with `metadata` and `rules`.
3. Use `python -m phase_runner.cli validate` to ensure the schema passes.
4. Add golden tests under `tests/golden/` mirroring expected outcomes.
5. Update `CHANGELOG.md` with the new pack version and notes.
6. Run `pytest` to ensure deterministic behavior.

## Development

- Python 3.11+
- Deterministic outputs (JSON sorting + sorted rule iteration)
- No PHI in fixtures—only synthetic patient data is included.
