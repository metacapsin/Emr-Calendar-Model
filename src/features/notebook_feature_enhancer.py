"""
Notebook Feature Enhancer — Production-Grade Replication
=========================================================

Replicates ALL feature engineering from feature_engineering.ipynb to ensure
perfect parity between notebook research and production training pipeline.

Features added:
- Synthetic datetime for temporal ordering
- Slot-level demand/popularity features
- Leakage-aware historical features (shifted expanding stats)
- Rolling time-window features (7D, 30D)
- Patient-provider loyalty features
- All interaction features from notebook

Usage:
    from src.features.notebook_feature_enhancer import enhance_features
    df_enhanced = enhance_features(df_ml_ready)
"""
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def safe_div(numer, denom):
    """Safe division that avoids divide-by-zero errors."""
    numer = np.asarray(numer, dtype=float)
    denom = np.asarray(denom, dtype=float)
    out = np.zeros_like(numer, dtype=float)
    m = denom != 0
    out[m] = numer[m] / denom[m]
    return out


def _create_synthetic_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create synthetic appointment datetime anchored to year 2026.
    
    This is a pragmatic approximation used in the notebook for:
    - Sorting events consistently
    - Computing historical (shifted) patient/provider/slot statistics
    - Computing time-window rollups (7D/30D) without using current label
    """
    df = df.copy()
    
    df['appt_dt'] = pd.to_datetime(
        {
            'year': 2026,
            'month': df['appt_month'].astype(int),
            'day': df['appt_day'].astype(int),
            'hour': df['appt_hour'].astype(int),
            'minute': 0,
        },
        errors='coerce'
    )
    
    nat_rate = df['appt_dt'].isna().mean()
    if nat_rate > 0:
        logger.warning(f"NaT rate in synthetic datetime: {nat_rate:.3%}")
    
    # Stable ordering key even if appt_dt has NaT
    df['_row_id'] = np.arange(len(df))
    df = df.sort_values(['appt_dt', '_row_id']).reset_index(drop=True)
    
    return df


def _add_slot_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add slot-level demand, popularity, and success rate features.
    
    These are global (not time-shifted) features computed per slot key.
    """
    df = df.copy()
    
    # Slot key: hour x weekday combination
    df['slot_key'] = df['appt_weekday'].astype(int).astype(str) + '_' + df['appt_hour'].astype(int).astype(str)
    
    # Slot demand count: how many appointments in this slot
    slot_counts = df.groupby('slot_key').size().rename('slot_demand_count')
    df = df.join(slot_counts, on='slot_key')
    
    # Popularity score: min-max scaled demand
    c = df['slot_demand_count'].astype(float)
    df['slot_popularity_score'] = (c - c.min()) / (c.max() - c.min() + 1e-9)
    
    # Historical success rate per slot (global, not shifted)
    slot_success = df.groupby('slot_key')['appointment_success'].mean().rename('slot_historical_success_rate')
    df = df.join(slot_success, on='slot_key')
    
    # Provider workload by slot (how often provider works that slot)
    prov_slot = df.groupby(['provider_encoded', 'slot_key']).size().rename('provider_slot_volume')
    df = df.join(prov_slot, on=['provider_encoded', 'slot_key'])
    
    return df


def _add_patient_provider_loyalty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add patient-provider loyalty feature.
    
    Measures how often a patient visits a specific provider relative to their total visits.
    """
    df = df.copy()
    
    if 'patient_provider_history' in df.columns and 'patient_total_appts' in df.columns:
        df['patient_provider_loyalty'] = safe_div(
            df['patient_provider_history'], 
            df['patient_total_appts']
        )
    else:
        df['patient_provider_loyalty'] = 0.0
    
    return df


def _add_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add leakage-aware historical features (shifted expanding stats).
    
    These features approximate what you could know AT SCHEDULING TIME:
    - Patient historical success rate
    - Provider historical success rate
    - Patient-provider historical success rate
    - Slot historical success rate
    
    Implementation: sort by appt_dt and use shift(1) so current label is NEVER used.
    """
    df = df.copy()
    
    # Expanding (all-time) historical success rates, shifted to avoid leakage
    df['patient_hist_success_rate'] = (
        df.groupby('patient_encoded')['appointment_success']
          .apply(lambda s: s.shift(1).expanding(min_periods=1).mean())
          .reset_index(level=0, drop=True)
    )
    
    df['provider_hist_success_rate'] = (
        df.groupby('provider_encoded')['appointment_success']
          .apply(lambda s: s.shift(1).expanding(min_periods=1).mean())
          .reset_index(level=0, drop=True)
    )
    
    # Patient-provider pair history
    pair_key = ['patient_encoded', 'provider_encoded']
    df['patient_provider_hist_success_rate'] = (
        df.groupby(pair_key)['appointment_success']
          .apply(lambda s: s.shift(1).expanding(min_periods=1).mean())
          .reset_index(level=pair_key, drop=True)
    )
    
    # Slot history
    df['slot_hist_success_rate_shifted'] = (
        df.groupby('slot_key')['appointment_success']
          .apply(lambda s: s.shift(1).expanding(min_periods=1).mean())
          .reset_index(level=0, drop=True)
    )
    
    # Historical volumes (how many past appointments) as stability indicators
    df['patient_hist_appt_count'] = df.groupby('patient_encoded').cumcount()
    df['provider_hist_appt_count'] = df.groupby('provider_encoded').cumcount()
    df['patient_provider_hist_appt_count'] = df.groupby(pair_key).cumcount()
    df['slot_hist_appt_count'] = df.groupby('slot_key').cumcount()
    
    # Fill NaNs from first-ever events with global mean
    hist_cols = [
        'patient_hist_success_rate', 'provider_hist_success_rate',
        'patient_provider_hist_success_rate', 'slot_hist_success_rate_shifted'
    ]
    global_mean = df['appointment_success'].mean()
    df[hist_cols] = df[hist_cols].fillna(global_mean)
    
    return df


def _add_time_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling 7-day and 30-day metrics (time-windowed).
    
    Computes rolling window features per patient/provider over PREVIOUS 7 and 30 days:
    - Appointment volume
    - Success rate
    
    These are computed using appt_dt and are SHIFTED to avoid leakage.
    """
    df = df.copy()
    
    def add_time_rolling(df_in: pd.DataFrame, group_col: str, time_col: str, 
                         label_col: str, windows=('7D', '30D')):
        """Add rolling features for a specific group column."""
        # Ensure unique index
        df_in = df_in.reset_index(drop=True)
        df_in = df_in.sort_values([time_col, '_row_id']).reset_index(drop=True).copy()
        
        out = df_in[[group_col, time_col, label_col]].copy()
        out[label_col] = out.groupby(group_col)[label_col].shift(1)
        out_valid = out.dropna(subset=[time_col]).copy()
        
        for w in windows:
            rolled = (
                out_valid.groupby(group_col)
                         .rolling(w, on=time_col)[label_col]
                         .agg(['count', 'mean'])
            )
            # Reset index to align with original dataframe
            rolled = rolled.reset_index()
            rolled = rolled.rename(columns={'level_1': 'orig_idx'} if 'level_1' in rolled.columns else {})
            
            # Map back to original index
            if 'orig_idx' in rolled.columns:
                rolled = rolled.set_index('orig_idx')
            
            df_in[f'{group_col}_roll_{w}_count'] = rolled['count'].reindex(df_in.index).fillna(0)
            df_in[f'{group_col}_roll_{w}_success_rate'] = rolled['mean'].reindex(df_in.index).fillna(df_in[label_col].mean())
        
        return df_in
    
    # Patient rolling features
    df = add_time_rolling(df, group_col='patient_encoded', time_col='appt_dt', 
                          label_col='appointment_success')
    
    # Provider rolling features
    df = add_time_rolling(df, group_col='provider_encoded', time_col='appt_dt', 
                          label_col='appointment_success')
    
    return df


def _clean_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean enhanced dataframe for modeling.
    
    - Drop internal columns (appt_dt, slot_key, _row_id)
    - Drop target leakage columns (appt_status_encoded)
    - Convert bool to int
    - Coerce non-numeric to numeric
    - Drop all-NaN columns
    - Fill remaining NaNs with median
    """
    df = df.copy()
    
    # Drop internal/temporary columns AND target leakage columns
    drop_cols = {
        'appt_dt', 'slot_key', '_row_id', 'appointment_success',
        'appt_status_encoded'  # CRITICAL: This is the source of the target, must drop!
    }
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # Convert True/False to 0/1
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)
    
    # Coerce non-numeric (if any) to numeric via pandas
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    for c in non_numeric:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # Drop columns that are entirely NaN after coercion
    all_nan = [c for c in df.columns if df[c].isna().all()]
    if all_nan:
        logger.warning(f"Dropping all-NaN columns: {all_nan}")
        df = df.drop(columns=all_nan)
    
    # Fill remaining NaNs with median (simple + robust)
    df = df.apply(lambda s: s.fillna(s.median()) if pd.api.types.is_numeric_dtype(s) else s)
    
    return df


def enhance_features(df: pd.DataFrame, 
                     create_target: bool = True, 
                     clean_for_modeling: bool = True) -> pd.DataFrame:
    """
    Master function to replicate ALL feature engineering from feature_engineering.ipynb.
    
    Args:
        df: Input DataFrame from ml_ready_appointments.csv
        create_target: If True, create appointment_success binary target
        clean_for_modeling: If True, clean and prepare for modeling (drop internals, fill NaN, etc.)
    
    Returns:
        Enhanced DataFrame with all notebook features
    
    Raises:
        ValueError: If required columns are missing
    """
    logger.info("Starting notebook feature enhancement...")
    initial_cols = len(df.columns)
    
    # Validate required columns
    required_cols = ['patient_encoded', 'provider_encoded', 'appt_hour', 
                     'appt_weekday', 'appt_month', 'appt_day']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    df = df.copy()
    
    # Step 1: Create binary target if requested
    if create_target:
        if 'appt_status_encoded' in df.columns:
            # Binarize: 1 = Confirmed (label==1), 0 = everything else
            df['appointment_success'] = (df['appt_status_encoded'] == 1).astype(int)
            logger.info(f"Created appointment_success target. "
                       f"Success rate: {df['appointment_success'].mean():.1%}")
        else:
            logger.warning("appt_status_encoded not found, skipping target creation")
    
    # Step 2: Create synthetic datetime
    logger.info("Creating synthetic datetime...")
    df = _create_synthetic_datetime(df)
    
    # Step 3: Add patient-provider loyalty
    logger.info("Adding patient-provider loyalty...")
    df = _add_patient_provider_loyalty(df)
    
    # Step 4: Add slot-level features
    logger.info("Adding slot-level features...")
    df = _add_slot_level_features(df)
    
    # Step 5: Add leakage-aware historical features
    logger.info("Adding historical features (leakage-aware)...")
    df = _add_historical_features(df)
    
    # Step 6: Add rolling time-window features
    logger.info("Adding rolling time-window features...")
    df = _add_time_rolling_features(df)
    
    # Step 7: Clean for modeling (optional)
    if clean_for_modeling:
        logger.info("Cleaning for modeling...")
        df = _clean_for_modeling(df)
    
    final_cols = len(df.columns)
    logger.info(f"Feature enhancement complete: {initial_cols} → {final_cols} features "
               f"(+{final_cols - initial_cols} new features)")
    
    return df


def get_new_feature_names() -> list:
    """
    Get list of all feature names added by notebook enhancer.
    
    Useful for validation and debugging.
    """
    return [
        # Slot-level features
        'slot_demand_count',
        'slot_popularity_score',
        'slot_historical_success_rate',
        'provider_slot_volume',
        
        # Patient-provider features
        'patient_provider_loyalty',
        
        # Historical features
        'patient_hist_success_rate',
        'provider_hist_success_rate',
        'patient_provider_hist_success_rate',
        'slot_hist_success_rate_shifted',
        'patient_hist_appt_count',
        'provider_hist_appt_count',
        'patient_provider_hist_appt_count',
        'slot_hist_appt_count',
        
        # Rolling features
        'patient_encoded_roll_7D_count',
        'patient_encoded_roll_7D_success_rate',
        'patient_encoded_roll_30D_count',
        'patient_encoded_roll_30D_success_rate',
        'provider_encoded_roll_7D_count',
        'provider_encoded_roll_7D_success_rate',
        'provider_encoded_roll_30D_count',
        'provider_encoded_roll_30D_success_rate',
    ]
