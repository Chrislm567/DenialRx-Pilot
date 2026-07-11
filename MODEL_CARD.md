# Phase Runner Denial Predictor — Model Card

## Model Overview
- **Model type:** Gradient boosted decision trees (XGBoost `XGBClassifier`).
- **Primary purpose:** Predict whether a medical claim will be denied so revenue cycle teams can triage high-risk submissions.
- **Version:** 0.3 (Phase Runner pilot).
- **Owners:** Phase Runner ML team.

## Training Data
- Synthetic claims generated for development (240 rows, 2023 service dates).
- Key columns: payer, provider, CPT, diagnosis, service date, claim amount, patient age, inpatient flag, denial outcome.
- No protected health information (PHI) or real patient data are included.

## Feature Engineering
- Windowed histories for payer and payer/CPT/DX combinations (30–120 day lookbacks) using leakage-safe rolling aggregations.
- Provider volume and denial history windows (90-day lookback).
- Static features: claim amount, patient age, inpatient indicator, historical denial rates.
- Leakage checks run automatically on sampled records each build.

## Training Procedure
- Deterministic seed (`23`) for cross-validation and final training.
- 5-fold stratified CV with ROC AUC and average precision metrics tracked.
- Final model refit on all data and persisted to `artifacts/denial_xgb.json`.
- SHAP value explanations stored per-claim in `artifacts/shap_values.csv`.

## Performance (synthetic validation)
- ROC AUC: 1.00
- Average Precision: 1.00
- Threshold guidance (report in `reports/evaluation_report.md`):
  - 0.25 threshold: precision 0.95, recall 1.00, estimated net savings $3.5k over review costs.
  - 0.40+ thresholds tighten to precision 1.00 with similar coverage.
- Metrics are optimistic because they are measured on synthetic data; expect lower performance on production data.

## Monitoring & Drift
- Population Stability Index (PSI) and Kolmogorov–Smirnov (KS) drift monitoring per feature; alert when PSI ≥ 0.2 or KS ≥ 0.1.
- Latest drift snapshot saved to `reports/drift/drift_report.json` comparing the most recent 60 claims to historical baselines.
- Recommended cadence: evaluate drift weekly and retrain when multiple features show persistent `alert` status.

## Ethical Considerations & Limitations
- Synthetic data does not capture real-world reimbursement biases; deploy with caution and retrain on actual, privacy-compliant data.
- Model should augment (not replace) clinical billing expertise.
- False positives incur review costs; use threshold scenarios to align with operational capacity.
- Ensure ongoing governance for fairness if sensitive attributes are introduced in future datasets.

## Usage Notes
- Install dependencies via `pip install -r requirements.txt`.
- Run end-to-end pipeline: `python scripts/run_phase_runner.py --data <claims.csv>`.
- Review generated artifacts and reports before promoting the model downstream.
