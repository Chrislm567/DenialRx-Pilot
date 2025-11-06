# Phase Runner — ML Denial Predictor

End-to-end pilot pipeline for predicting medical claim denials with engineered window features, explainability, and drift monitoring.

## Project Structure
- `data/synthetic_claims.csv` — synthetic development dataset with 240 claims.
- `src/phase_runner/` — feature store, model training, evaluation, and drift modules.
- `scripts/run_phase_runner.py` — CLI orchestration script.
- `artifacts/` — generated model, predictions, and SHAP values (after running the pipeline).
- `reports/` — evaluation summary, drift report, and run summary.
- `MODEL_CARD.md` — model documentation.
- `RUNLOG.md` — chronological record of training runs and key metrics.

## Quickstart
1. Install dependencies (Python 3.12+):
   ```bash
   pip install -r requirements.txt
   ```
2. Generate the synthetic dataset (reproducible):
   ```bash
   python scripts/generate_synthetic_data.py
   ```
3. Train, evaluate, and monitor:
   ```bash
   python scripts/run_phase_runner.py \
       --data data/synthetic_claims.csv \
       --artifacts artifacts \
       --reports reports \
       --seed 23
   ```

The run produces:
- `artifacts/denial_xgb.json` — serialized XGBoost model.
- `artifacts/predictions.csv` — actuals, cross-validated predictions, and final probabilities.
- `artifacts/shap_values.csv` — per-claim SHAP explanations.
- `reports/evaluation_report.(json|md)` — metrics, threshold scenarios, and cost sketch.
- `reports/drift/drift_report.json` — PSI/KS drift assessment.
- `reports/RUN_SUMMARY.md` — high-level run recap.

## Testing
Run the automated checks:
```bash
pytest
```

Tests cover feature-store construction, model training, evaluation reporting, and drift monitoring using the synthetic dataset.
