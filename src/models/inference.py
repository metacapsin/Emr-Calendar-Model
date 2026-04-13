from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SlotInferenceEngine:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        bundle = joblib.load(str(self.model_path))
        self.model = bundle.get("model")
        self.feature_columns: List[str] = bundle.get("feature_columns", [])
        self.model_name: str = bundle.get("model_name", "unknown")
        self.feature_stats: Dict[str, Dict[str, float]] = bundle.get("feature_stats", {})
        self.calibration_method: Optional[str] = bundle.get("calibration_method")

        if self.model is None or not self.feature_columns:
            raise ValueError("Invalid model bundle: missing 'model' or 'feature_columns'")

        if not hasattr(self.model, "predict_proba"):
            raise ValueError("Loaded model does not support predict_proba")

        logger.info("Model loaded: name=%s features=%d calibration=%s", self.model_name, len(self.feature_columns), self.calibration_method or "none")

    def _detect_feature_drift(self, df: pd.DataFrame) -> None:
        if not self.feature_stats:
            return
        drifted = {}
        for col, stats in self.feature_stats.items():
            if col not in df.columns:
                continue
            live_mean = float(df[col].mean())
            expected_mean = float(stats.get("mean", 0.0))
            expected_std = float(stats.get("std", 0.0))
            threshold = max(expected_std * 2.0, 0.5)
            if abs(live_mean - expected_mean) > threshold:
                drifted[col] = {
                    "expected_mean": expected_mean,
                    "live_mean": live_mean,
                    "expected_std": expected_std,
                }
        if drifted:
            logger.warning(
                "Feature drift detected",
                extra={"extra": {"drifted_features": drifted}},
            )

    def _validate_feature_matrix(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Empty feature matrix")
        if (df == 0.0).all().all():
            raise ValueError("Feature matrix contains only zeros")

        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError("Feature matrix contains no numeric features")

        variances = numeric_df.var(numeric_only=True)
        varying_cols = variances[variances > 1e-8].index.tolist()
        if not varying_cols:
            raise ValueError("Feature matrix contains no varying numeric features")

        non_constant_ratio = len(varying_cols) / max(1, len(numeric_df.columns))
        if non_constant_ratio < 0.05:
            logger.warning(
                "Low variance feature matrix: only %d/%d numeric columns vary",
                len(varying_cols),
                len(numeric_df.columns),
            )

    def _prepare(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_columns if c not in features.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        df = features[self.feature_columns].copy().fillna(0.0).astype(float)
        if df.isna().any().any():
            raise ValueError("NaN values present in feature matrix after fillna")

        self._validate_feature_matrix(df)
        self._detect_feature_drift(df)

        min_val = float(df.min().min())
        max_val = float(df.max().max())
        variance = float(df.var(numeric_only=True).min()) if not df.empty else 0.0
        logger.info(
            "Inference features prepared",
            extra={"extra": {
                "rows": len(df),
                "min_value": min_val,
                "max_value": max_val,
                "min_variance": variance,
            }},
        )
        return df

    def predict_proba(self, features: pd.DataFrame) -> List[List[float]]:
        df = self._prepare(features.copy())
        try:
            proba = self.model.predict_proba(df)
            probs = proba.tolist()
            scores = [float(p[1]) for p in probs]
            self._ensure_probability_distribution(scores)
            logger.info(
                "Inference output",
                extra={"extra": {
                    "min_prob": min(scores) if scores else 0.0,
                    "max_prob": max(scores) if scores else 0.0,
                    "mean_prob": sum(scores) / max(1, len(scores)),
                    "variance_prob": float(np.var(scores)) if scores else 0.0,
                }},
            )
            return probs
        except Exception as exc:
            logger.error("predict_proba failed: %s", exc)
            raise

    def _ensure_probability_distribution(self, scores: List[float]) -> None:
        if len(scores) <= 1:
            return
        if max(scores) - min(scores) < 1e-4:
            logger.warning(
                "Model produced degenerate probability distribution; proceeding with available scores",
                extra={"extra": {"min_prob": min(scores), "max_prob": max(scores), "variance_prob": float(np.var(scores))}},
            )

    def predict(self, features: pd.DataFrame) -> List[int]:
        df = self._prepare(features.copy())
        return self.model.predict(df).tolist()

    def batch_predict(self, feature_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not feature_rows:
            return []
        df = pd.DataFrame(feature_rows)
        proba = self.predict_proba(df)
        return [
            {"index": i, "prob": float(p[1]), "prob_negative": float(p[0])}
            for i, p in enumerate(proba)
        ]
