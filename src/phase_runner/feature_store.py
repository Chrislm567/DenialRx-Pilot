from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureSpec:
    """Configuration for the feature store."""

    payer_windows: Sequence[int] = (30, 90)
    combo_windows: Sequence[int] = (60, 120)
    provider_windows: Sequence[int] = (90,)
    leakage_check_sample: int = 10
    target_column: str = "denied"
    date_column: str = "service_date"


class FeatureStore:
    """Build windowed features for denial modelling."""

    def __init__(self, spec: Optional[FeatureSpec] = None):
        self.spec = spec or FeatureSpec()
        self.feature_columns_: List[str] = []

    @staticmethod
    def load_claims(path: str | bytes | PathLike[str]) -> pd.DataFrame:
        """Load claims with proper dtypes."""

        df = pd.read_csv(path, parse_dates=["service_date"])
        categorical_cols = ["payer", "provider", "cpt_code", "dx_code", "claim_id"]
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df

    def build_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df = df.copy()
        df = df.sort_values(self.spec.date_column).reset_index(drop=True)

        for col in [self.spec.target_column, self.spec.date_column]:
            if col not in df.columns:
                raise KeyError(f"Expected column '{col}' in source data")

        feature_frames: List[pd.DataFrame] = []

        feature_frames.append(self._make_static_features(df))

        for window in self.spec.payer_windows:
            feature_frames.append(self._window_stats(df, ["payer"], window, prefix=f"payer_{window}d"))

        combo_cols = ["payer", "cpt_code", "dx_code"]
        if all(col in df.columns for col in combo_cols):
            for window in self.spec.combo_windows:
                feature_frames.append(
                    self._window_stats(df, combo_cols, window, prefix=f"payer_cpt_dx_{window}d")
                )

        if "provider" in df.columns:
            for window in self.spec.provider_windows:
                feature_frames.append(self._window_stats(df, ["provider"], window, prefix=f"provider_{window}d"))

        feature_matrix = pd.concat(feature_frames, axis=1)
        feature_matrix = feature_matrix.fillna(0.0)

        target = df[self.spec.target_column].astype(int)
        self.feature_columns_ = list(feature_matrix.columns)

        self._run_leakage_checks(df, feature_matrix)

        return feature_matrix, target

    def _make_static_features(self, df: pd.DataFrame) -> pd.DataFrame:
        static_features = pd.DataFrame(index=df.index)
        numeric_cols = ["claim_amount", "patient_age"]
        for col in numeric_cols:
            if col in df.columns:
                static_features[col] = df[col].astype(float)
        if "is_inpatient" in df.columns:
            static_features["is_inpatient"] = df["is_inpatient"].astype(int)

        # Target encoding style prevalence for payer and cpt/dx combos using expanding counts
        if {"payer"}.issubset(df.columns):
            counts = df.groupby("payer", observed=True)[self.spec.target_column].cumcount()
            denied_counts = (
                df.groupby("payer", observed=True)[self.spec.target_column].cumsum().shift(1).fillna(0)
            )
            static_features["payer_denial_rate_all"] = np.divide(
                denied_counts,
                counts.replace(0, np.nan),
            ).fillna(0.0)

        if {"payer", "cpt_code", "dx_code"}.issubset(df.columns):
            combo_group = df.groupby(["payer", "cpt_code", "dx_code"], observed=True)
            combo_counts = combo_group.cumcount()
            combo_denied = combo_group[self.spec.target_column].cumsum().shift(1).fillna(0)
            static_features["combo_denial_rate_all"] = np.divide(
                combo_denied,
                combo_counts.replace(0, np.nan),
            ).fillna(0.0)

        return static_features

    def _window_stats(
        self,
        df: pd.DataFrame,
        group_cols: Sequence[str],
        window_days: int,
        prefix: str,
    ) -> pd.DataFrame:
        grouped = df.groupby(list(group_cols), group_keys=False, sort=False, observed=True)

        def _compute(group: pd.DataFrame) -> pd.DataFrame:
            g = group.sort_values(self.spec.date_column)
            idx = g.index
            series = g.set_index(self.spec.date_column)

            denied = series[self.spec.target_column]
            denied_roll = denied.rolling(f"{window_days}D", min_periods=1, closed="left").sum()

            volume_roll = (
                pd.Series(1.0, index=denied.index)
                .rolling(f"{window_days}D", min_periods=1, closed="left")
                .sum()
            )

            amount_roll = None
            if "claim_amount" in series:
                amount_roll = series["claim_amount"].rolling(
                    f"{window_days}D", min_periods=1, closed="left"
                ).sum()

            denied_values = denied_roll.to_numpy()
            volume_values = volume_roll.to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                rate_values = np.divide(denied_values, volume_values, where=volume_values != 0)

            data = {
                f"{prefix}_denial_rate": np.nan_to_num(rate_values, nan=0.0),
                f"{prefix}_denials": np.nan_to_num(denied_values, nan=0.0),
                f"{prefix}_volume": np.nan_to_num(volume_values, nan=0.0),
            }
            if amount_roll is not None:
                amount_values = amount_roll.to_numpy()
                with np.errstate(divide="ignore", invalid="ignore"):
                    avg_values = np.divide(amount_values, volume_values, where=volume_values != 0)
                data[f"{prefix}_amount_sum"] = np.nan_to_num(amount_values, nan=0.0)
                data[f"{prefix}_amount_avg"] = np.nan_to_num(avg_values, nan=0.0)

            return pd.DataFrame(data, index=idx)

        features = grouped.apply(_compute, include_groups=False).sort_index()
        return features

    def _run_leakage_checks(self, df: pd.DataFrame, features: pd.DataFrame) -> None:
        rng = np.random.default_rng(1234)
        sample_indices = rng.choice(len(df), size=min(self.spec.leakage_check_sample, len(df)), replace=False)
        for idx in sample_indices:
            row = df.iloc[idx]
            cutoff = row[self.spec.date_column]

            history_mask = df[self.spec.date_column] < cutoff
            history = df.loc[history_mask]

            for window in self.spec.payer_windows:
                if "payer" in df.columns:
                    mask = history["payer"] == row["payer"]
                    window_mask = history[self.spec.date_column] >= cutoff - pd.Timedelta(days=window)
                    window_hist = history.loc[mask & window_mask]
                    expected_rate = (
                        window_hist[self.spec.target_column].mean() if not window_hist.empty else 0.0
                    )
                    actual_rate = features.iloc[idx][f"payer_{window}d_denial_rate"]
                    if not np.isfinite(actual_rate):
                        actual_rate = 0.0
                    if abs(actual_rate - expected_rate) > 1e-6:
                        raise ValueError("Leakage check failed for payer window features")

            combo_cols = ["payer", "cpt_code", "dx_code"]
            if all(col in df.columns for col in combo_cols):
                for window in self.spec.combo_windows:
                    mask = (history["payer"] == row["payer"]) & (
                        history["cpt_code"] == row["cpt_code"]
                    ) & (history["dx_code"] == row["dx_code"])
                    window_mask = history[self.spec.date_column] >= cutoff - pd.Timedelta(days=window)
                    window_hist = history.loc[mask & window_mask]
                    expected_rate = (
                        window_hist[self.spec.target_column].mean() if not window_hist.empty else 0.0
                    )
                    actual_rate = features.iloc[idx][f"payer_cpt_dx_{window}d_denial_rate"]
                    if not np.isfinite(actual_rate):
                        actual_rate = 0.0
                    if abs(actual_rate - expected_rate) > 1e-6:
                        raise ValueError("Leakage check failed for combo window features")

    def get_feature_names(self) -> List[str]:
        return list(self.feature_columns_)
