"""
Dataset Regeneration Script — Enhanced with Notebook Features
==============================================================

This script regenerates the ML-ready dataset with ALL notebook feature engineering
applied, creating a comprehensive enhanced dataset for training.

Features added:
- Leakage-aware historical features (patient/provider/slot success rates)
- Rolling time-window features (7D/30D)
- Slot demand/popularity features
- Patient-provider loyalty features

Output:
    dataset/processed/enhanced_ml_ready_appointments.csv

Usage:
    python scripts/regenerate_enhanced_dataset.py
    python scripts/regenerate_enhanced_dataset.py --input dataset/processed/ml_ready_appointments.csv
"""
import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.features.notebook_feature_enhancer import enhance_features, get_new_feature_names
from src.utils.logger import get_logger

logger = get_logger(__name__)


def analyze_dataset(df: pd.DataFrame, name: str = "Dataset"):
    """Print comprehensive dataset statistics."""
    logger.info(f"\n{'='*70}")
    logger.info(f"{name} ANALYSIS")
    logger.info(f"{'='*70}")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Rows: {len(df):,}")
    logger.info(f"Columns: {len(df.columns)}")
    
    # Check for missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        logger.info(f"\nMissing values in {len(missing_cols)} columns:")
        for col, count in missing_cols.head(10).items():
            logger.info(f"  {col}: {count} ({count/len(df):.1%})")
    else:
        logger.info(f"\n✓ No missing values")
    
    # Check for zero-variance columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    zero_var = []
    for col in numeric_cols:
        if df[col].var() < 1e-8:
            zero_var.append(col)
    
    if zero_var:
        logger.info(f"\n⚠ Zero-variance columns: {len(zero_var)}")
        for col in zero_var[:10]:
            logger.info(f"  {col}")
    else:
        logger.info(f"\n✓ No zero-variance columns")
    
    # Data types
    logger.info(f"\nData types:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        logger.info(f"  {dtype}: {count} columns")


def regenerate_dataset(input_path: str, output_path: str):
    """
    Regenerate dataset with notebook feature engineering.
    
    Args:
        input_path: Path to original ml_ready_appointments.csv
        output_path: Path to save enhanced dataset
    """
    logger.info("="*70)
    logger.info("DATASET REGENERATION — NOTEBOOK FEATURE ENHANCEMENT")
    logger.info("="*70)
    
    # Load original dataset
    logger.info(f"\nLoading dataset from: {input_path}")
    df_original = pd.read_csv(input_path, skipinitialspace=True)
    df_original.columns = [c.strip() for c in df_original.columns]
    
    logger.info(f"Original dataset loaded: {df_original.shape}")
    analyze_dataset(df_original, "ORIGINAL")
    
    # Apply notebook feature engineering
    logger.info("\n" + "="*70)
    logger.info("APPLYING NOTEBOOK FEATURE ENGINEERING")
    logger.info("="*70)
    
    df_enhanced = enhance_features(
        df_original, 
        create_target=True, 
        clean_for_modeling=False  # Keep metadata for analysis
    )
    
    # Analyze enhanced dataset
    logger.info("\n" + "="*70)
    analyze_dataset(df_enhanced, "ENHANCED")
    
    # Report new features
    new_features = get_new_feature_names()
    actual_new = [f for f in new_features if f in df_enhanced.columns]
    logger.info(f"\nNew features added: {len(actual_new)}")
    for feat in actual_new:
        logger.info(f"  ✓ {feat}")
    
    # Feature count summary
    logger.info(f"\n{'='*70}")
    logger.info("FEATURE SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"Original features: {len(df_original.columns)}")
    logger.info(f"Enhanced features: {len(df_enhanced.columns)}")
    logger.info(f"New features: +{len(df_enhanced.columns) - len(df_original.columns)}")
    
    if 'appointment_success' in df_enhanced.columns:
        success_rate = df_enhanced['appointment_success'].mean()
        logger.info(f"\nTarget distribution (appointment_success):")
        logger.info(f"  Success (1): {(df_enhanced['appointment_success'] == 1).sum():,} ({success_rate:.1%})")
        logger.info(f"  Failure (0): {(df_enhanced['appointment_success'] == 0).sum():,} ({1-success_rate:.1%})")
    
    # Save enhanced dataset
    logger.info(f"\n{'='*70}")
    logger.info(f"SAVING ENHANCED DATASET")
    logger.info(f"{'='*70}")
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_enhanced.to_csv(output_path, index=False)
    logger.info(f"✓ Saved to: {output_path}")
    logger.info(f"  File size: {Path(output_path).stat().st_size / 1024:.1f} KB")
    
    # Validation
    logger.info(f"\n{'='*70}")
    logger.info("VALIDATION")
    logger.info(f"{'='*70}")
    
    # Reload and verify
    df_verify = pd.read_csv(output_path)
    assert df_verify.shape == df_enhanced.shape, "Shape mismatch after save/load!"
    assert list(df_verify.columns) == list(df_enhanced.columns), "Column mismatch!"
    logger.info(f"✓ Dataset integrity verified (shape: {df_verify.shape})")
    logger.info(f"✓ All {len(df_verify.columns)} columns present")
    logger.info(f"✓ No data corruption detected")
    
    logger.info(f"\n{'='*70}")
    logger.info("REGENERATION COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Review enhanced dataset: {output_path}")
    logger.info(f"  2. Train model: python src/models/train_slot_model.py")
    logger.info(f"  3. Test inference: python scripts/run_inference.py")
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regenerate ML-ready dataset with notebook feature engineering"
    )
    parser.add_argument(
        "--input",
        default="dataset/processed/ml_ready_appointments.csv",
        help="Path to original ML-ready dataset"
    )
    parser.add_argument(
        "--output",
        default="dataset/processed/enhanced_ml_ready_appointments.csv",
        help="Path to save enhanced dataset"
    )
    
    args = parser.parse_args()
    
    regenerate_dataset(args.input, args.output)
