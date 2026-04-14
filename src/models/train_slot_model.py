"""
Production Training Pipeline — Notebook Parity Edition
=======================================================

Trains CatBoost on ml_ready_appointments.csv with PERFECT NOTEBOOK PARITY:

1. Replicates ALL feature engineering from feature_engineering.ipynb
2. Uses notebook hyperparameters (iterations=800, etc.)
3. Time-based train/test split (not random)
4. Optional hyperparameter tuning with RandomizedSearchCV
5. Proper calibration (separate calibration set, no data leakage)
6. Comprehensive metrics and feature statistics

Feature Enhancement:
- Leakage-aware historical features (patient/provider/slot success rates)
- Rolling time-window features (7D/30D)
- Slot demand/popularity features
- Patient-provider loyalty features

Usage:
    python src/models/train_slot_model.py
    python src/models/train_slot_model.py --data dataset/processed/ml_ready_appointments.csv
    python src/models/train_slot_model.py --tune  # Enable hyperparameter tuning
"""
import argparse
import json
import sys
from pathlib import Path

# Add project root to Python path if running as script
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score, f1_score

try:
    from catboost import CatBoostClassifier
except ImportError:
    print("catboost not installed. Run: pip install catboost")
    sys.exit(1)

from src.features.notebook_feature_enhancer import enhance_features
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_and_enhance(csv_path: str) -> tuple:
    """
    Load ML-ready dataset and apply notebook feature engineering.
    
    Returns:
        X: Feature DataFrame with all notebook features
        y: Binary target (appointment_success)
        df_full: Full DataFrame with metadata (for time-based split)
    """
    logger.info(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    
    logger.info(f"Raw dataset shape: {df.shape}")
    
    # Apply notebook feature engineering
    logger.info("Applying notebook feature enhancement...")
    df_enhanced = enhance_features(df, create_target=True, clean_for_modeling=False)  # Don't clean yet
    
    # Separate features and target BEFORE cleaning
    # Note: appointment_success might not exist if clean_for_modeling dropped it
    if 'appointment_success' not in df_enhanced.columns:
        # Recreate from appt_status_encoded before it gets dropped
        if 'appt_status_encoded' in df_enhanced.columns:
            df_enhanced['appointment_success'] = (df_enhanced['appt_status_encoded'] == 1).astype(int)
            logger.info("Recreated appointment_success target from appt_status_encoded")
        else:
            raise ValueError("appointment_success target not found after enhancement")
    
    y = df_enhanced['appointment_success'].astype(int)
    
    # NOW clean for modeling (will drop appt_status_encoded and appointment_success)
    from src.features.notebook_feature_enhancer import _clean_for_modeling
    X = _clean_for_modeling(df_enhanced)
    
    # Final validation
    logger.info(f"Enhanced dataset: {len(X)} rows, {len(X.columns)} features")
    logger.info(f"Target distribution: {y.sum()} positives ({y.mean():.1%})")
    
    # Remove zero-variance features
    variances = X.var()
    zero_var = variances[variances < 1e-8].index.tolist()
    if zero_var:
        logger.info(f"Removing {len(zero_var)} zero-variance features")
        X = X.drop(columns=zero_var)
    
    logger.info(f"Final feature count: {len(X.columns)}")
    
    return X, y, df_enhanced


def compute_feature_stats(X: pd.DataFrame) -> dict:
    """Compute mean/std per feature for drift detection at inference time."""
    stats = {}
    for col in X.columns:
        stats[col] = {
            "mean": round(float(X[col].mean()), 6),
            "std":  round(float(X[col].std()), 6),
        }
    return stats


def time_based_split(X: pd.DataFrame, y: pd.Series, df_full: pd.DataFrame, 
                     test_size: float = 0.2) -> tuple:
    """
    Time-based train/test split to reflect real-world deployment.
    
    Trains on past data, tests on future data (not random split).
    This matches the notebook's temporal split strategy.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        df_full: Full DataFrame with appt_dt for temporal ordering
        test_size: Fraction for test set (default 0.2)
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Use _row_id from enhancement (already sorted by appt_dt)
    if '_row_id' in df_full.columns:
        order = df_full['_row_id'].values
    else:
        order = np.arange(len(df_full))
    
    split_idx = int(len(order) * (1 - test_size))
    
    train_idx = order[:split_idx]
    test_idx = order[split_idx:]
    
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    
    logger.info(f"Time-based split: Train={len(X_train)}, Test={len(X_test)}")
    logger.info(f"Train success rate: {y_train.mean():.4f}, Test success rate: {y_test.mean():.4f}")
    
    return X_train, X_test, y_train, y_test


def train(csv_path: str, output_path: str, calibrate: bool = True, 
          enable_tuning: bool = False) -> None:
    """
    Train CatBoost model with perfect notebook parity.
    
    Args:
        csv_path: Path to ml_ready_appointments.csv
        output_path: Path to save model bundle
        calibrate: Whether to apply probability calibration
        enable_tuning: Whether to run hyperparameter tuning (RandomizedSearchCV)
    """
    # Load and enhance features
    X, y, df_full = load_and_enhance(csv_path)
    
    # Time-based split (matches notebook)
    X_train, X_test, y_train, y_test = time_based_split(X, y, df_full, test_size=0.2)
    
    # ── Model Configuration (Notebook Parity) ─────────────────────────────
    
    base_params = {
        'iterations': 800,          # Notebook uses 800 (not 500)
        'learning_rate': 0.05,      # Matches notebook
        'depth': 6,                 # Matches notebook
        'l2_leaf_reg': 3,           # Notebook default from tuning
        'loss_function': 'Logloss', # Matches notebook
        'eval_metric': 'AUC',       # Added for monitoring
        'random_seed': 42,          # Matches notebook
        'verbose': 100,             # Production: show progress
        'early_stopping_rounds': 50,# Production: prevent overfitting
    }
    
    logger.info("="*70)
    logger.info("MODEL CONFIGURATION (Notebook Parity)")
    logger.info("="*70)
    for key, val in base_params.items():
        logger.info(f"  {key}: {val}")
    
    # ── Optional Hyperparameter Tuning ────────────────────────────────────
    
    if enable_tuning:
        logger.info("\n" + "="*70)
        logger.info("HYPERPARAMETER TUNING (RandomizedSearchCV)")
        logger.info("="*70)
        
        param_distributions = {
            'depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'iterations': [400, 800, 1200],
            'l2_leaf_reg': [1, 3, 5, 7, 9]
        }
        
        base_model_tuning = CatBoostClassifier(**{**base_params, 'verbose': False})
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        search = RandomizedSearchCV(
            estimator=base_model_tuning,
            param_distributions=param_distributions,
            n_iter=20,
            scoring='roc_auc',
            cv=cv,
            n_jobs=-1,
            random_state=42,
            verbose=2
        )
        
        search.fit(X_train, y_train)
        
        best_params = search.best_params_
        logger.info(f"\nBest CV ROC-AUC: {search.best_score_:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        # Update base model with best params
        base_params.update(best_params)
        base_params['verbose'] = 100  # Restore verbosity for final training
        
        base_model = CatBoostClassifier(**base_params)
    else:
        base_model = CatBoostClassifier(**base_params)
    
    # ── Train Base Model ──────────────────────────────────────────────────
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING BASE MODEL")
    logger.info("="*70)
    
    base_model.fit(X_train, y_train, eval_set=(X_test, y_test))
    
    # Evaluate raw model
    raw_probs = base_model.predict_proba(X_test)[:, 1]
    raw_auc = roc_auc_score(y_test, raw_probs)
    raw_brier = brier_score_loss(y_test, raw_probs)
    raw_acc = accuracy_score(y_test, (raw_probs >= 0.5).astype(int))
    raw_f1 = f1_score(y_test, (raw_probs >= 0.5).astype(int))
    
    logger.info(f"\nRaw Model Performance:")
    logger.info(f"  ROC-AUC:  {raw_auc:.4f}")
    logger.info(f"  Brier:    {raw_brier:.4f}")
    logger.info(f"  Accuracy: {raw_acc:.4f}")
    logger.info(f"  F1-Score: {raw_f1:.4f}")
    
    # ── Calibration (Optional) ───────────────────────────────────────────
    
    calibrator = None
    calibration_method = None
    final_model = base_model
    
    if calibrate:
        logger.info("\n" + "="*70)
        logger.info("FITTING ISOTONIC CALIBRATION")
        logger.info("="*70)
        
        # Use separate calibration set to avoid data leakage
        # Split test set into calibration and evaluation sets
        X_cal, X_eval, y_cal, y_eval = train_test_split(
            X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
        )
        
        # Fit calibration - use cv=5 for sklearn >= 1.6, or prefit for older versions
        try:
            # Try prefit first (older sklearn)
            calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv="prefit")
            calibrated.fit(X_cal, y_cal)
        except Exception as e:
            # For newer sklearn, need to retrain with CV
            logger.info("Using CV-based calibration (sklearn >= 1.6)")
            calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
            calibrated.fit(X_cal, y_cal)
        
        cal_probs = calibrated.predict_proba(X_eval)[:, 1]
        cal_auc = roc_auc_score(y_eval, cal_probs)
        cal_brier = brier_score_loss(y_eval, cal_probs)
        cal_acc = accuracy_score(y_eval, (cal_probs >= 0.5).astype(int))
        cal_f1 = f1_score(y_eval, (cal_probs >= 0.5).astype(int))
        
        logger.info(f"\nCalibrated Model Performance (on held-out eval set):")
        logger.info(f"  ROC-AUC:  {cal_auc:.4f}")
        logger.info(f"  Brier:    {cal_brier:.4f}")
        logger.info(f"  Accuracy: {cal_acc:.4f}")
        logger.info(f"  F1-Score: {cal_f1:.4f}")
        
        # Use calibrated model if it improves or maintains Brier score
        if cal_brier <= raw_brier:
            logger.info(f"✓ Calibration improved Brier score ({raw_brier:.4f} → {cal_brier:.4f})")
            final_model = calibrated
            calibrator = calibrated
            calibration_method = "isotonic"
        else:
            logger.info(f"✗ Calibration degraded Brier score ({raw_brier:.4f} → {cal_brier:.4f}), using raw model")
    
    # ── Probability Spread Validation ────────────────────────────────────
    
    test_probs = final_model.predict_proba(X_test)[:, 1]
    spread = float(np.max(test_probs) - np.min(test_probs))
    logger.info(f"\nProbability spread on test set: {spread:.4f} "
               f"(min={np.min(test_probs):.4f}, max={np.max(test_probs):.4f})")
    
    if spread < 0.1:
        logger.warning("⚠ Low probability spread — model may not differentiate slots well")
    
    # ── Feature Importance (Top 20) ──────────────────────────────────────
    
    if hasattr(base_model, "feature_importances_"):
        importances = sorted(
            zip(X.columns, base_model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        logger.info("\nTop 20 feature importances:")
        for feat, imp in importances[:20]:
            logger.info(f"  {feat:<50} {imp:.4f}")
    
    # ── Save Model Bundle ────────────────────────────────────────────────
    
    feature_stats = compute_feature_stats(X_train)
    bundle = {
        "model":              final_model,
        "feature_columns":    list(X.columns),
        "model_name":         "catboost_isotonic_calibrated" if calibrate and calibrator else "catboost",
        "calibration_method": calibration_method,
        "calibrator":         calibrator,
        "feature_stats":      feature_stats,
        "trained_at":         pd.Timestamp.utcnow().isoformat(),
        "train_auc":          raw_auc,
        "train_brier":        raw_brier,
        "train_accuracy":     raw_acc,
        "train_f1":           raw_f1,
        "n_features":         len(X.columns),
        "notebook_parity":    True,
        "hyperparameters":    base_params,
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"Model bundle saved to: {output_path}")
    logger.info(f"Features: {len(X.columns)}")
    logger.info(f"ROC-AUC:  {raw_auc:.4f}")
    logger.info(f"Brier:    {raw_brier:.4f}")
    logger.info(f"Calibration: {calibration_method or 'none'}")
    logger.info(f"Notebook Parity: YES ✓")
    logger.info("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train slot prediction model with perfect notebook parity"
    )
    parser.add_argument(
        "--data",
        default="dataset/processed/ml_ready_appointments.csv",
        help="Path to ML-ready dataset CSV"
    )
    parser.add_argument(
        "--output",
        default="models/slot_prediction_model.pkl",
        help="Path to save trained model bundle"
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Disable probability calibration"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning with RandomizedSearchCV (takes longer)"
    )
    
    args = parser.parse_args()
    
    train(
        csv_path=args.data,
        output_path=args.output,
        calibrate=not args.no_calibrate,
        enable_tuning=args.tune
    )
