import joblib
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


class SlotInferenceEngine:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        bundle = joblib.load(str(self.model_path))
        self.model = bundle.get('model')
        self.feature_columns = bundle.get('feature_columns', [])

        if self.model is None or not self.feature_columns:
            raise ValueError('Invalid model bundle format.')

    def _validate_input(self, features: pd.DataFrame) -> pd.DataFrame:
        missing_cols = [c for c in self.feature_columns if c not in features.columns]
        if missing_cols:
            raise ValueError(f"Missing required feature columns: {missing_cols}")

        return features[self.feature_columns].astype(float)

    def predict_proba(self, features: pd.DataFrame) -> List[List[float]]:
        df_valid = self._validate_input(features.copy())
        proba = self.model.predict_proba(df_valid)
        return proba.tolist()

    def predict(self, features: pd.DataFrame) -> List[int]:
        df_valid = self._validate_input(features.copy())
        preds = self.model.predict(df_valid)
        return preds.tolist()

    def batch_predict(self, feature_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        df = pd.DataFrame(feature_rows)
        proba = self.predict_proba(df)
        results = []
        for i, row in df.iterrows():
            results.append({'index': int(i), 'prob': float(proba[i][1]), 'prob_negative': float(proba[i][0])})
        return results
