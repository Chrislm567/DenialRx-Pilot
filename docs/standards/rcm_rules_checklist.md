# RCM Rules Checklist

This checklist captures common utilization management requirements used to build
Phase Runner demonstration rule packs. Each policy includes a short summary and
links back to public payer documentation.

## Sample Policies

### Sunrise Health Commercial
- **Policy**: Prior authorization is required when billed charges exceed $5,000
  for outpatient imaging (CT/MRI) codes.
- **Code set**: CPT 70450, 70551-70553, 72141-72158.
- **Notes**: Patients under 18 should be routed for manual review regardless of
  billed amount.

### Horizon Medicaid Advantage
- **Policy**: Deny emergency room visits when the primary diagnosis is in the
  avoidable list *and* the patient was discharged home.
- **Diagnosis set**: ICD-10 R51, J02.9, K30.
- **Exceptions**: Approve if the patient is pregnant or is under 2 years old.

### OpenCare Value PPO
- **Policy**: Approve physical therapy visits when cumulative visits for the
  year are ≤ 12 and the diagnosis is musculoskeletal.
- **Escalation**: Visits 13-20 should pend for clinical review; anything over 20
  is denied unless pre-approved.

These policies are encoded in `examples/rule_packs/` and validated during the
Phase Runner unit and golden tests.
