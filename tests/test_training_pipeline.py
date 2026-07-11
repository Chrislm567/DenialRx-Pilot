from pathlib import Path

import pytest

from phase_runner.drift import DriftMonitor
from phase_runner.evaluation import generate_report
from phase_runner.feature_store import FeatureSpec, FeatureStore
from phase_runner.modeling import ModelTrainer


def test_training_and_drift(tmp_path):
    data_path = Path("data/synthetic_claims.csv")
    artifacts_dir = tmp_path / "artifacts"
    reports_dir = tmp_path / "reports"

    store = FeatureStore(FeatureSpec(leakage_check_sample=3))
    claims = store.load_claims(data_path)
    features, target = store.build_features(claims)

    trainer = ModelTrainer(artifacts_dir, seed=23)
    artifacts = trainer.train(features, target, claims["claim_id"])

    assert artifacts.model_path.exists()
    assert artifacts.shap_path.exists()
    assert 0.5 < artifacts.overall_metrics["auc"] <= 1.0

    report = generate_report(
        target,
        artifacts.predictions["final_prediction"],
        thresholds=[0.25, 0.4],
        report_dir=reports_dir,
    )
    assert report.auc == pytest.approx(artifacts.overall_metrics["auc"])

    monitor = DriftMonitor()
    recent = features.tail(50)
    baseline = features.iloc[:-50]
    drift_report = monitor.compare(baseline, recent, artifacts.feature_names, reports_dir / "drift")
    assert drift_report.entries
