# Phase Runner Runbook

## Operational Budgets

| Budget            | Target | Notes                                             |
| ----------------- | ------ | ------------------------------------------------- |
| API availability  | 99.9%  | Track via service level objectives (SLOs).        |
| Worker latency    | < 2m   | Measure queue to completion per job.              |
| Web p95 load time | < 3s   | Observed from real-user monitoring dashboards.    |
| Incident response | < 15m  | Time from alert to incident commander assignment. |

## Deployment Checkpoints

1. **Readiness review** — Verify merged changes pass CI quality gates and changelog
   entries are drafted.
2. **Staging soak** — Deploy to staging, run smoke tests, validate dashboards and
   data pipelines.
3. **Change approval** — Confirm change management ticket sign-off and scheduled
   window availability.
4. **Production release** — Execute rollout via infrastructure pipelines with
   staged canaries.

## Rollback Strategy

- **Automated rollback** — Pipeline monitors key metrics and reverts to the last
  known good artifact when error budget burn > 2% in 15 minutes.
- **Manual rollback** — Incident commander triggers `make rollback` (future
  target) to redeploy the previous release tag.
- **Data considerations** — When schema changes are involved, maintain backward
  compatible migrations and prepare reversible scripts.

## Escalation Matrix

| Role               | Contact                   |
| ------------------ | ------------------------- |
| Incident Commander | oncall@denialrx.example   |
| Engineering Lead   | eng-lead@denialrx.example |
| Product Owner      | product@denialrx.example  |
| Security Officer   | security@denialrx.example |

## Observability Signals

- Error rate anomalies from API gateway logs.
- Worker queue depth spikes exceeding 500 pending jobs.
- Web application Core Web Vitals regressions.
- Alerts from PHI/PII scanner or audit events.

## Recovery Checklist

- [ ] Validate customer impact summary.
- [ ] Communicate to stakeholders via status page and Slack.
- [ ] Confirm rollback (if executed) completed successfully.
- [ ] File incident retrospective within 48 hours.
