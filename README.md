# DenialRx Phase Runner

The Phase Runner mono-repo hosts the foundational scaffolding for DenialRx API,
web, and worker applications together with shared domain packages.

## Repository layout

```
apps/
  api/            # Service interfaces and gateway orchestration
  web/            # Frontend experience and UX prototypes
  workers/        # Async and scheduled processing jobs
packages/
  x12/            # Healthcare X12 utilities (placeholder)
  rules/          # Rules evaluation helpers and scoring logic
  ml/             # Machine learning experiments and scoring harnesses
  observability/  # Telemetry aggregation and dashboards
infra/            # Infrastructure as code and deployment manifests
tests/            # Cross-cutting integration and regression suites
```

## Getting started

1. Install the toolchain dependencies (Node 18+ with pnpm, Python 3.11).
2. Bootstrap the workspace:
    ```bash
    make bootstrap
    ```
3. Run the quality gates locally before committing:
    ```bash
    make lint
    make type-check
    make test
    make scan
    make pii-scan
    ```

For more information on contributing and operational workflows, consult
[`CONTRIBUTING.md`](CONTRIBUTING.md) and [`RUNBOOK.md`](RUNBOOK.md).
