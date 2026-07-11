from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass
class ThresholdScenario:
    threshold: float
    precision: float
    recall: float
    fpr: float
    tpr: float
    review_rate: float
    expected_cost_savings: float
    notes: str = ""


@dataclass
class EvaluationReport:
    auc: float
    average_precision: float
    threshold_scenarios: List[ThresholdScenario]

    def to_dict(self) -> Dict[str, object]:
        return {
            "auc": self.auc,
            "average_precision": self.average_precision,
            "threshold_scenarios": [scenario.__dict__ for scenario in self.threshold_scenarios],
        }

    def to_markdown(self) -> str:
        lines = ["# Evaluation Summary", "", f"* ROC AUC: {self.auc:.3f}", f"* Average Precision: {self.average_precision:.3f}", ""]
        lines.append("| Threshold | Precision | Recall | FPR | Review Rate | Cost Savings | Notes |")
        lines.append("|-----------|-----------|--------|-----|-------------|--------------|-------|")
        for scenario in self.threshold_scenarios:
            lines.append(
                f"| {scenario.threshold:.2f} | {scenario.precision:.3f} | {scenario.recall:.3f} | "
                f"{scenario.fpr:.3f} | {scenario.review_rate:.3f} | ${scenario.expected_cost_savings:,.0f} | {scenario.notes} |"
            )
        return "\n".join(lines)


def build_threshold_scenarios(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    thresholds: Iterable[float],
    review_cost: float = 18.0,
    prevented_denial_savings: float = 80.0,
) -> List[ThresholdScenario]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    scenarios: List[ThresholdScenario] = []

    for threshold in thresholds:
        positive_mask = y_prob >= threshold
        tp = np.logical_and(positive_mask, y_true == 1).sum()
        fp = np.logical_and(positive_mask, y_true == 0).sum()
        fn = np.logical_and(~positive_mask, y_true == 1).sum()
        tn = np.logical_and(~positive_mask, y_true == 0).sum()

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        fpr = fp / (fp + tn) if fp + tn > 0 else 0.0
        tpr = recall
        review_rate = positive_mask.mean()

        expected_savings = tp * prevented_denial_savings - (tp + fp) * review_cost
        notes = "High precision" if precision >= 0.6 else "Broad net"
        scenarios.append(
            ThresholdScenario(
                threshold=float(threshold),
                precision=float(precision),
                recall=float(recall),
                fpr=float(fpr),
                tpr=float(tpr),
                review_rate=float(review_rate),
                expected_cost_savings=float(expected_savings),
                notes=notes,
            )
        )

    return scenarios


def generate_report(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    thresholds: Iterable[float],
    report_dir: Path,
) -> EvaluationReport:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    auc = roc_auc_score(y_true, y_prob)
    average_precision = average_precision_score(y_true, y_prob)
    scenarios = build_threshold_scenarios(y_true, y_prob, thresholds)

    report = EvaluationReport(auc=float(auc), average_precision=float(average_precision), threshold_scenarios=scenarios)

    report_path = report_dir / "evaluation_report.json"
    markdown_path = report_dir / "evaluation_report.md"

    pd.Series(report.to_dict()).to_json(report_path, indent=2)
    markdown_path.write_text(report.to_markdown())

    return report
