"""Tests for slot feature builder."""
import joblib
import pandas as pd
import pytest

from src.features.slot_feature_builder import build_slot_features, build_slots_feature_dataframe

bundle = joblib.load("models/slot_prediction_model.pkl")
FEATURE_COLUMNS = bundle["feature_columns"]

PATIENT = {"patient_encoded": 5, "patient_age": 35, "sex_encoded": 1, "patient_total_appts": 3, "patient_cancel_rate": 0.1}
PROVIDER = {"provider_encoded": 2, "provider_total_appts": 100, "provider_avg_duration": 30, "provider_cancel_rate": 0.05}
SLOT = {"date": "2026-04-01", "hour": 9, "weekday": 0, "month": 4, "day": 1, "year": 2026, "slot_duration_minutes": 60}


def test_build_slot_features_returns_all_columns():
    features = build_slot_features(PATIENT, PROVIDER, SLOT, FEATURE_COLUMNS)
    for col in FEATURE_COLUMNS:
        assert col in features, f"Missing column: {col}"


def test_build_slot_features_types():
    features = build_slot_features(PATIENT, PROVIDER, SLOT, FEATURE_COLUMNS)
    for col, val in features.items():
        assert isinstance(val, (int, float)), f"Column {col} has non-numeric value: {val}"


def test_build_slots_feature_dataframe_shape():
    slots = [SLOT, {**SLOT, "hour": 10}, {**SLOT, "hour": 11}]
    df = build_slots_feature_dataframe(slots, PATIENT, PROVIDER, FEATURE_COLUMNS)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, len(FEATURE_COLUMNS))


def test_feature_columns_match_model():
    assert len(FEATURE_COLUMNS) == 105


def test_lead_time_non_negative():
    features = build_slot_features(PATIENT, PROVIDER, SLOT, FEATURE_COLUMNS)
    assert features["lead_time_days"] >= 0.0


def test_weekend_flag():
    saturday_slot = {**SLOT, "weekday": 5}
    features = build_slot_features(PATIENT, PROVIDER, saturday_slot, FEATURE_COLUMNS)
    assert features["is_weekend"] == 1

    monday_slot = {**SLOT, "weekday": 0}
    features2 = build_slot_features(PATIENT, PROVIDER, monday_slot, FEATURE_COLUMNS)
    assert features2["is_weekend"] == 0
