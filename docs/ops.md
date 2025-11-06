# Appeals Automation Ops Guide

## Queues
- **drafts** — short-lived jobs triggered by intake to generate letters and email previews.
- **submissions** — durable queue that hands off finalized appeals to payer gateways; retries use exponential backoff with a 4-hour ceiling.
- **evidence** — handles PDF bundling and checksum validation before payload upload.

## Retries
- Draft generation retries twice (idempotent) when template rendering fails.
- Submissions retry up to 5 times with jitter; abort once two consecutive attempts fail PHI validation.
- Evidence packs retry three times with file-lock detection; corrupted bundles are quarantined for manual review.

## Deadlines
- Drafts SLA: 2 minutes from intake to preview delivery.
- Submission SLA: 15 minutes end-to-end, including audit persistence.
- Evidence bundling SLA: 5 minutes with alarms if checksum generation exceeds thresholds.

## Audit & Monitoring
- Immutable audit trail stored in append-only JSONL; rotate daily to object storage.
- PHI is redacted prior to log emission; metrics export only hashed identifiers.
- Stop automated retries if CI reports two consecutive red builds to prevent cascading failures.
