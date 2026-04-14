"""
Production Training Pipeline
=============================
Trains CatBoost on ml_ready_appointments.csv with:
- Constant-feature removal
- Isotonic calibration
- Feature stats saved for drift detection
- Full bundle saved: model + feature_columns + calibrator + feature_stats
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss

try:
    from catboost import CatBoostClassifier
except ImportError:
    print("catboost not installed. Run: pip install catboost")
    sys.exit(1)


# Columns that are labels or identifiers, never features
_EXCLUDE = {"appt_status_encoded", "status", "label", "target", "_id", "patient_id", "provider_id"}

# Columns known to be zero-variance in the training data (from audit)
_KNOWN_ZERO_VARIANCE = {"patient_total_appts", "patient_7day_appts", "patient_30day_appts",
                         "patient_7day_cancel", "patient_30day_cancel", "season"}


def load_and_clean(csv_path: str) -> tuple:
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    # Identify target
    target_col = "appt_status_encoded"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Columns: {list(df.columns)[:10]}")

    # Binarize: 1 = Confirmed (label==1), 0 = everything else
    y = (df[target_col] == 1).astype(int)

    # Drop non-feature columns
    drop_cols = [c for c in df.columns if c in _EXCLUDE]
    X = df.drop(columns=drop_cols, errors="ignore")

    # Convert bool-like string columns
    for col in X.columns:
        if X[col].dtype == object:
            try:
                X[col] = X[col].map({"True": 1, "False": 0, True: 1, False: 0}).fillna(X[col])
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
            except Exception:
                X[col] = 0.0

    X = X.fillna(0.0).astype(float)

    # Remove zero-variance features
    variances = X.var()
    zero_var  = variances[variances < 1e-8].index.tolist()
    if zero_var:
        print(f"Removing {len(zero_var)} zero-variance features: {zero_var[:10]}{'...' if len(zero_var) > 10 else ''}")
        X = X.drop(columns=zero_var)

    print(f"Dataset: {len(X)} rows, {len(X.columns)} features, {y.sum()} positives ({y.mean():.1%})")
    return X, y


def compute_feature_stats(X: pd.DataFrame) -> dict:
    """Compute mean/std per feature for drift detection at inference time."""
    stats = {}
    for col in X.columns:
        stats[col] = {
            "mean": round(float(X[col].mean()), 6),
            "std":  round(float(X[col].std()), 6),
        }
    return stats


def train(csv_path: str, output_path: str, calibrate: bool = True) -> None:
    X, y = load_and_clean(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Base CatBoost model
    base_model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
    )
    base_model.fit(X_train, y_train, eval_set=(X_test, y_test))

    # Evaluate raw model
    raw_probs = base_model.predict_proba(X_test)[:, 1]
    raw_auc   = roc_auc_score(y_test, raw_probs)
    raw_brier = brier_score_loss(y_test, raw_probs)
    print(f"\nRaw model  — ROC-AUC: {raw_auc:.4f}  Brier: {raw_brier:.4f}")

    # Calibration via isotonic regression
    calibrator = None
    calibration_method = None
    final_model = base_model

    if calibrate:
        print("\nFitting isotonic calibration...")
        calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv="prefit")
        calibrated.fit(X_test, y_test)

        cal_probs = calibrated.predict_proba(X_test)[:, 1]
        cal_auc   = roc_auc_score(y_test, cal_probs)
        cal_brier = brier_score_loss(y_test, cal_probs)
        print(f"Calibrated — ROC-AUC: {cal_auc:.4f}  Brier: {cal_brier:.4f}")

        # Use calibrated model as the inference model
        final_model        = calibrated
        calibrator         = calibrated
        calibration_method = "isotonic"

    # Probability spread check
    test_probs = final_model.predict_proba(X_test)[:, 1]
    spread = float(np.max(test_probs) - np.min(test_probs))
    print(f"\nProbability spread on test set: {spread:.4f} (min={np.min(test_probs):.4f}, max={np.max(test_probs):.4f})")
    if spread < 0.1:
        print("WARNING: Low probability spread — model may not differentiate slots well.")

    # Feature importance (top 20)
    if hasattr(base_model, "feature_importances_"):
        importances = sorted(
            zip(X.columns, base_model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        print("\nTop 20 feature importances:")
        for feat, imp in importances[:20]:
            print(f"  {feat:<45} {imp:.4f}")

    # Save bundle
    feature_stats = compute_feature_stats(X_train)
    bundle = {
        "model":              final_model,
        "feature_columns":    list(X.columns),
        "model_name":         "catboost_isotonic_calibrated" if calibrate else "catboost",
        "calibration_method": calibration_method,
        "calibrator":         calibrator,
        "feature_stats":      feature_stats,
        "trained_at":         pd.Timestamp.utcnow().isoformat(),
        "train_auc":          raw_auc,
        "n_features":         len(X.columns),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    print(f"\nModel bundle saved to: {output_path}")
    print(f"Features: {len(X.columns)}, AUC: {raw_auc:.4f}, Calibration: {calibration_method or 'none'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train slot prediction model")
    parser.add_argument("--data",      default="dataset/processed/ml_ready_appointments.csv")
    parser.add_argument("--output",    default="models/slot_prediction_model.pkl")
    parser.add_argument("--no-calibrate", action="store_true")
    args = parser.parse_args()

    train(args.data, args.output, calibrate=not args.no_calibrate)
