from pathlib import Path
from typing import Any, Dict, List

import joblib
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

        if self.model is None or not self.feature_columns:
            raise ValueError("Invalid model bundle: missing 'model' or 'feature_columns'")

        logger.info("Model loaded: name=%s features=%d", self.model_name, len(self.feature_columns))

    def _prepare(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_columns if c not in features.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        return features[self.feature_columns].fillna(0.0).astype(float)

    def predict_proba(self, features: pd.DataFrame) -> List[List[float]]:
        df = self._prepare(features.copy())
        try:
            proba = self.model.predict_proba(df)
            return proba.tolist()
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
            {"index": i, "prob": float(p[1]), "prob_negative": float(p[0])}
            for i, p in enumerate(proba)
        ]
