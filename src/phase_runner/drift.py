from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


@dataclass
class DriftEntry:
    feature: str
    psi: float
    ks_stat: float
    status: str


@dataclass
class DriftReport:
    entries: List[DriftEntry]
    psi_threshold: float
    ks_threshold: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "psi_threshold": self.psi_threshold,
            "ks_threshold": self.ks_threshold,
            "entries": [entry.__dict__ for entry in self.entries],
        }


class DriftMonitor:
    def __init__(self, psi_threshold: float = 0.2, ks_threshold: float = 0.1):
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold

    @staticmethod
    def _compute_psi(expected: Sequence[float], actual: Sequence[float], bins: int = 10) -> float:
        expected = np.asarray(expected, dtype=float)
        actual = np.asarray(actual, dtype=float)

        if np.all(expected == expected[0]) and np.all(actual == actual[0]):
            return 0.0

        quantiles = np.linspace(0.0, 1.0, bins + 1)
        bin_edges = np.unique(np.quantile(expected, quantiles))
        if len(bin_edges) <= 2:
            bin_edges = np.linspace(np.min(expected), np.max(expected) + 1e-6, bins + 1)

        expected_counts, _ = np.histogram(expected, bins=bin_edges)
        actual_counts, _ = np.histogram(actual, bins=bin_edges)

        expected_pct = np.clip(expected_counts / expected_counts.sum(), 1e-6, None)
        actual_pct = np.clip(actual_counts / actual_counts.sum(), 1e-6, None)

        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi)

    @staticmethod
    def _compute_ks(expected: Sequence[float], actual: Sequence[float]) -> float:
        statistic, _ = ks_2samp(expected, actual)
        return float(statistic)

    def compare(
        self,
        baseline: pd.DataFrame,
        recent: pd.DataFrame,
        feature_names: Sequence[str],
        report_dir: Path,
    ) -> DriftReport:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

        entries: List[DriftEntry] = []
        for feature in feature_names:
            expected = baseline[feature]
            actual = recent[feature]
            psi = self._compute_psi(expected, actual)
            ks_stat = self._compute_ks(expected, actual)

            status = "ok"
            if psi >= self.psi_threshold or ks_stat >= self.ks_threshold:
                status = "alert"
            elif psi >= self.psi_threshold * 0.5 or ks_stat >= self.ks_threshold * 0.5:
                status = "warn"

            entries.append(DriftEntry(feature=feature, psi=psi, ks_stat=ks_stat, status=status))

        report = DriftReport(entries=entries, psi_threshold=self.psi_threshold, ks_threshold=self.ks_threshold)

        report_path = report_dir / "drift_report.json"
        pd.Series(report.to_dict()).to_json(report_path, indent=2)

        return report
