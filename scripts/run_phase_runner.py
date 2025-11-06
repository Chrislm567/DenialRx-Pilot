import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase_runner import DriftMonitor, FeatureSpec, FeatureStore, ModelTrainer
from phase_runner.evaluation import generate_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Phase Runner denial model")
    parser.add_argument("--data", default="data/synthetic_claims.csv", help="Path to claims CSV")
    parser.add_argument("--artifacts", default="artifacts", help="Directory to write model artifacts")
    parser.add_argument("--reports", default="reports", help="Directory to write evaluation/drift reports")
    parser.add_argument("--seed", type=int, default=23, help="Deterministic seed")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[0.25, 0.4, 0.55],
        help="Probability thresholds to profile",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    artifacts_dir = Path(args.artifacts)
    reports_dir = Path(args.reports)

    store = FeatureStore(FeatureSpec())
    claims = store.load_claims(data_path)
    features, target = store.build_features(claims)

    claim_ids = claims["claim_id"] if "claim_id" in claims.columns else None

    trainer = ModelTrainer(artifacts_dir, seed=args.seed)
    training_artifacts = trainer.train(features, target, claim_ids)

    evaluation_report = generate_report(
        target,
        training_artifacts.predictions["final_prediction"],
        thresholds=args.thresholds,
        report_dir=reports_dir,
    )

    monitor = DriftMonitor()
    # Use the final 60 claims as a hold-out recent slice for drift illustration.
    recent_slice = features.tail(60)
    baseline_slice = features.iloc[:-60]
    monitor.compare(baseline_slice, recent_slice, training_artifacts.feature_names, reports_dir / "drift")

    summary_path = reports_dir / "RUN_SUMMARY.md"
    summary_lines = [
        "# Phase Runner Run Summary",
        "",
        f"* Data rows: {len(claims)}",
        f"* Features: {len(training_artifacts.feature_names)}",
        f"* ROC AUC: {evaluation_report.auc:.3f}",
        f"* Average Precision: {evaluation_report.average_precision:.3f}",
        f"* Model path: {training_artifacts.model_path}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
