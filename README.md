# Phase Runner

Phase Runner is a unified portal for denial analytics, payer performance, and operations observability.

## Web Application

The frontend is a Next.js 14 dashboard with Auth0 authentication. It exposes three primary workspaces:

- **Overview** – executive pulse of approval rates, denial heatmaps, and active alerts.
- **Claims Explorer** – interactive filters, tabular drilldowns, and CSV export for operations teams.
- **Payer Scorecards** – side-by-side payer benchmarking with qualitative heatmaps.

### Getting Started

```bash
npm install
cp .env.example .env.local
# populate Auth0 settings and tracing endpoint, or set AUTH_DISABLED=true for local mocks
npm run dev
```

End-to-end tests are implemented with Playwright:

```bash
npm test
```

Lighthouse best-practices score ≥90 can be validated via:

```bash
npx lighthouse http://localhost:3000 --view
```

### Observability

- **Tracing** – OpenTelemetry Node SDK exports spans to an OTLP endpoint.
- **Logging** – Structured JSON logs emitted via Pino.
- **Metrics** – Prometheus-compatible metrics exposed at `/api/metrics`.

Grafana dashboard JSON and alert rules live under `observability/grafana/`.

## Infrastructure as Code

Terraform modules are provided under `infra/modules` for VPC, RDS, S3, KMS, and WAF with secure defaults.

Environment compositions live in `infra/environments`. The repository includes a plan-only GitHub Action workflow for pull requests.

Refer to `infra/README.md` for usage instructions.
