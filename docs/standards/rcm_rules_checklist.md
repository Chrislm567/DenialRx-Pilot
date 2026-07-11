# RCM Rules Checklist

## X12 external code lists
- **Claim Adjustment Reason Codes (CARC)** — Enumerates reasons payers adjust or deny claims; DenialRx maps CARCs to remediation playbooks and reporting dashboards.[^carc]
- **Remittance Advice Remark Codes (RARC)** — Supplies payer narrative detail to pair with CARCs; use to enrich agent guidance and templated appeals.[^rarc]
- **Claim Status Category & Codes (277C codes)** — Track lifecycle states for outstanding claims; drive follow-up workqueues and SLA monitoring.[^status]

_Update cadence_: X12 posts code list updates multiple times per year; subscribe to the X12 change log and schedule a quarterly review to import any revisions, capturing effective dates and retirement notes.[^carc]

## CMS National Correct Coding Initiative (NCCI)
- **Procedure-to-Procedure (PTP) edits** — Guard against mutually exclusive or comprehensive code pairs; DenialRx should run edits pre-bill and surface justifications when overriding.[^ptp]
- **Medically Unlikely Edits (MUEs)** — Enforce maximum units per service; use to flag outlier quantities before submission and to generate appeal rationale when medically necessary units exceed limits.[^mue]
- **NCCI Policy Manual** — Provides narrative guidance behind PTP/MUE logic; reference sections when generating clinical justification text.[^manual]

_Update cadence_: CMS releases quarterly NCCI edit files (usually mid-month prior to each calendar quarter) and annual Policy Manual updates; align ingest jobs with CMS’s publication calendar and record the CMS effective date for downstream rules.[^ptp]

## Versioning guidelines
1. Store each ruleset with source name, version identifier (publish date or CMS quarter), download URL, and checksum.
2. Maintain backward-compatible parsing so historical claims can be reprocessed with the rule set active at time of service.
3. Capture change summaries (new codes, retirements) in release notes for analysts and for regression testing scope.
4. Automate alerting when upstream URLs change or files fail checksum validation.

### YAML version stub
```yaml
- source: X12 CARC
  version: 2025-03-01
  url: https://x12.org/codes/claim-adjustment-reason-codes
  checksum: TBD
- source: CMS NCCI PTP
  version: 2025-Q2
  url: https://www.cms.gov/medicare/medicare-fee-for-service-payment/national-correct-coding-initiative-ncci-edits
  checksum: TBD
```

[^carc]: [X12 Claim Adjustment Reason Codes](https://x12.org/codes/claim-adjustment-reason-codes).
[^rarc]: [X12 Remittance Advice Remark Codes](https://x12.org/codes/remittance-advice-remark-codes).
[^status]: [X12 Claim Status Category and Codes](https://x12.org/codes/claim-status-category-codes).
[^ptp]: [CMS NCCI Procedure-to-Procedure Edits](https://www.cms.gov/medicare/medicare-fee-for-service-payment/national-correct-coding-initiative-ncci-edits).
[^mue]: [CMS NCCI Medically Unlikely Edits](https://www.cms.gov/medicare/medicare-fee-for-service-payment/national-correct-coding-initiative-ncci-mue).
[^manual]: [CMS NCCI Policy Manual](https://www.cms.gov/medicare/medicare-fee-for-service-payment/national-correct-coding-initiative-ncci-manuals).
