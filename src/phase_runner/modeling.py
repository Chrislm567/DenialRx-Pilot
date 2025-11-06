from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

import shap


@dataclass
class TrainingArtifacts:
    model_path: Path
    prediction_path: Path
    shap_path: Path
    cv_metrics: List[Dict[str, float]]
    overall_metrics: Dict[str, float]
    feature_names: Sequence[str]
    predictions: pd.DataFrame


class ModelTrainer:
    """Train an XGBoost model for claim denial prediction."""

    def __init__(self, artifacts_dir: Path, seed: int = 23):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.model_: Optional[XGBClassifier] = None
        self.feature_names_: Sequence[str] | None = None

    def _build_model(self) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_lambda=1.0,
            reg_alpha=0.2,
            n_jobs=4,
            random_state=self.seed,
            eval_metric="logloss",
            tree_method="hist",
            use_label_encoder=False,
        )

    def cross_validate(self, X: np.ndarray, y: np.ndarray) -> tuple[List[Dict[str, float]], np.ndarray]:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        metrics: List[Dict[str, float]] = []
        oof_predictions = np.zeros_like(y, dtype=float)

        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            model = self._build_model()
            model.fit(X[train_idx], y[train_idx])
            proba = model.predict_proba(X[test_idx])[:, 1]
            auc = roc_auc_score(y[test_idx], proba)
            pr = average_precision_score(y[test_idx], proba)
            metrics.append({"fold": fold, "auc": float(auc), "average_precision": float(pr)})
            oof_predictions[test_idx] = proba

        return metrics, oof_predictions

    def train(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        claim_ids: Optional[Sequence[str]] = None,
    ) -> TrainingArtifacts:
        X = features.values.astype(float)
        y = target.values.astype(int)

        cv_metrics, oof_predictions = self.cross_validate(X, y)

        model = self._build_model()
        model.fit(X, y)
        self.model_ = model
        self.feature_names_ = features.columns

        final_proba = model.predict_proba(X)[:, 1]
        overall_auc = roc_auc_score(y, final_proba)
        overall_pr = average_precision_score(y, final_proba)

        model_path = self.artifacts_dir / "denial_xgb.json"
        model.save_model(model_path)

        prediction_df = pd.DataFrame(
            {
                "claim_id": claim_ids if claim_ids is not None else np.arange(len(y)),
                "actual": y,
                "oof_prediction": oof_predictions,
                "final_prediction": final_proba,
            }
        )
        prediction_path = self.artifacts_dir / "predictions.csv"
        prediction_df.to_csv(prediction_path, index=False)

        shap_path = self._persist_shap(features, claim_ids)

        artifacts = TrainingArtifacts(
            model_path=model_path,
            prediction_path=prediction_path,
            shap_path=shap_path,
            cv_metrics=cv_metrics,
            overall_metrics={"auc": float(overall_auc), "average_precision": float(overall_pr)},
            feature_names=list(features.columns),
            predictions=prediction_df,
        )
        return artifacts

    def _persist_shap(
        self,
        features: pd.DataFrame,
        claim_ids: Optional[Sequence[str]] = None,
    ) -> Path:
        if self.model_ is None:
            raise RuntimeError("Model must be trained before computing SHAP values")

        explainer = shap.TreeExplainer(self.model_)
        shap_values = explainer.shap_values(features)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_df = pd.DataFrame(shap_values, columns=features.columns)
        shap_df.insert(0, "claim_id", claim_ids if claim_ids is not None else np.arange(len(shap_df)))
        base_value = np.array(explainer.expected_value).reshape(-1)
        shap_df["base_value"] = base_value[0] if base_value.size else explainer.expected_value

        shap_path = self.artifacts_dir / "shap_values.csv"
        shap_df.to_csv(shap_path, index=False)
        return shap_path

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model has not been trained")
        return self.model_.predict_proba(X)[:, 1]
