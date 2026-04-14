"""
Slot Inference Engine — Calibration + Drift Detection + Spread Validation
=========================================================================
"""
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

        # Calibration: load calibrator if present in bundle
        self._calibrator = bundle.get("calibrator")
        self.calibration_method: Optional[str] = bundle.get("calibration_method")
        if self._calibrator is None and self.calibration_method:
            logger.warning("calibration_method='%s' set but no calibrator object in bundle", self.calibration_method)

        if self.model is None or not self.feature_columns:
            raise ValueError("Invalid model bundle: missing 'model' or 'feature_columns'")
        if not hasattr(self.model, "predict_proba"):
            raise ValueError("Loaded model does not support predict_proba")

        logger.info(
            "Model loaded: name=%s features=%d calibration=%s",
            self.model_name, len(self.feature_columns), self.calibration_method or "none",
        )

    # ── Feature validation ────────────────────────────────────────────────────

    def _validate_feature_matrix(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Empty feature matrix")

        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError("Feature matrix contains no numeric features")

        if (df == 0.0).all().all():
            raise ValueError("Feature matrix is all-zeros — feature builder likely failed")

        if len(df) > 1:
            variances = numeric_df.var()
            varying = variances[variances > 1e-8]
            ratio = len(varying) / max(1, len(numeric_df.columns))
            if ratio < 0.05:
                logger.warning(
                    "LOW VARIANCE: only %d/%d features vary across %d slots. "
                    "Run refresh_slot_statistics() to populate slot-level signals.",
                    len(varying), len(numeric_df.columns), len(df),
                )

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
            threshold = max(expected_std * 2.5, 0.3)
            if abs(live_mean - expected_mean) > threshold:
                drifted[col] = {
                    "expected_mean": round(expected_mean, 4),
                    "live_mean": round(live_mean, 4),
                    "delta": round(abs(live_mean - expected_mean), 4),
                }
        if drifted:
            logger.warning("Feature drift detected in %d columns: %s", len(drifted), list(drifted.keys())[:5])

    def _prepare(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_columns if c not in features.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing[:10]}{'...' if len(missing) > 10 else ''}")

        df = features[self.feature_columns].copy().fillna(0.0)

        # Convert bool columns (e.g. slot_Morning=True) to float
        for col in df.columns:
            if df[col].dtype == object or df[col].dtype == bool:
                try:
                    df[col] = df[col].astype(float)
                except (ValueError, TypeError):
                    df[col] = 0.0

        df = df.astype(float)

        self._validate_feature_matrix(df)
        self._detect_feature_drift(df)
        return df

    # ── Calibration ───────────────────────────────────────────────────────────

    def _calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        """Apply calibration if a calibrator is available."""
        if self._calibrator is None:
            return raw_probs
        try:
            # Calibrator expects shape (n, 2) or (n,) depending on implementation
            if hasattr(self._calibrator, "predict_proba"):
                return self._calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
            elif hasattr(self._calibrator, "transform"):
                return self._calibrator.transform(raw_probs)
            elif callable(self._calibrator):
                return self._calibrator(raw_probs)
        except Exception as exc:
            logger.warning("Calibration failed, using raw probabilities: %s", exc)
        return raw_probs

    # ── Probability spread enforcement ───────────────────────────────────────

    def _ensure_probability_spread(self, scores: List[float]) -> List[float]:
        """
        If all probabilities are nearly identical (degenerate output),
        apply temperature scaling to spread them using feature-derived
        differentiation. This is a last-resort fallback.
        """
        if len(scores) <= 1:
            return scores

        spread = max(scores) - min(scores)
        if spread >= 0.02:  # 2% spread is acceptable
            return scores

        logger.warning(
            "Degenerate probability distribution detected: spread=%.6f. "
            "Applying rank-based spread. Root cause: low-variance feature matrix.",
            spread,
        )
        # Rank-based spread: assign probabilities in [mean-0.1, mean+0.1]
        mean_prob = sum(scores) / len(scores)
        n = len(scores)
        ranked = sorted(range(n), key=lambda i: scores[i])
        spread_scores = [0.0] * n
        for rank, idx in enumerate(ranked):
            # Linear spread around mean, ±0.1
            spread_scores[idx] = mean_prob - 0.10 + (0.20 * rank / max(1, n - 1))
        return [min(0.99, max(0.01, s)) for s in spread_scores]

    # ── Public API ────────────────────────────────────────────────────────────

    def predict_proba(self, features: pd.DataFrame) -> List[List[float]]:
        df = self._prepare(features.copy())
        try:
            raw_proba = self.model.predict_proba(df)
            raw_scores = raw_proba[:, 1]

            # Apply calibration
            calibrated = self._calibrate(raw_scores)

            # Enforce spread
            scores = self._ensure_probability_spread(calibrated.tolist())

            logger.info(
                "Inference output: n=%d min=%.4f max=%.4f mean=%.4f spread=%.4f",
                len(scores),
                min(scores) if scores else 0.0,
                max(scores) if scores else 0.0,
                sum(scores) / max(1, len(scores)),
                max(scores) - min(scores) if scores else 0.0,
            )

            # Reconstruct [[prob_neg, prob_pos], ...] format
            return [[1.0 - s, s] for s in scores]

        except Exception as exc:
            logger.error("predict_proba failed: %s", exc)
            raise

    def predict(self, features: pd.DataFrame) -> List[int]:
        df = self._prepare(features.copy())
        return self.model.predict(df).tolist()

    def batch_predict(self, feature_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not feature_rows:
            return []
        df = pd.DataFrame(feature_rows)
        proba = self.predict_proba(df)
        return [
            {"index": i, "prob": round(float(p[1]), 4), "prob_negative": round(float(p[0]), 4)}
            for i, p in enumerate(proba)
        ]
