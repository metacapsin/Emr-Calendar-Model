import math
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


TIME_SLOT_RANGES = {
    'morning': range(8, 11),
    'midday': range(11, 13),
    'afternoon': range(13, 16),
    'evening': range(16, 20),
}


def get_time_of_day(hour: int) -> str:
    for when, rng in TIME_SLOT_RANGES.items():
        if hour in rng:
            return when
    return 'other'


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def build_slot_features(
    patient_info: Dict[str, Any],
    provider_info: Dict[str, Any],
    slot_info: Dict[str, Any],
    feature_columns: List[str],
) -> Dict[str, Any]:
    """Build one exact feature vector row for the saved slot prediction model."""
    features: Dict[str, Any] = {c: 0.0 for c in feature_columns}

    # Basic patient + provider mapping. Keep original columns.
    patient_defaults = {
        'patient_age': patient_info.get('patient_age', 35),
        'sex_encoded': patient_info.get('sex_encoded', 1),
        'patient_total_appts': patient_info.get('patient_total_appts', 1),
        'patient_cancel_rate': patient_info.get('patient_cancel_rate', 0.0),
        'patient_7day_appts': patient_info.get('patient_7day_appts', 0),
        'patient_30day_appts': patient_info.get('patient_30day_appts', 0),
        'patient_7day_cancel': patient_info.get('patient_7day_cancel', 0),
        'patient_30day_cancel': patient_info.get('patient_30day_cancel', 0),
        'patient_provider_history': patient_info.get('patient_provider_history', 0),
    }

    provider_defaults = {
        'provider_total_appts': provider_info.get('provider_total_appts', 1),
        'provider_avg_duration': provider_info.get('provider_avg_duration', 30),
        'provider_cancel_rate': provider_info.get('provider_cancel_rate', 0.0),
        'provider_7day_util': provider_info.get('provider_7day_util', 0.0),
        'provider_30day_util': provider_info.get('provider_30day_util', 0.0),
    }

    # Conservative default logic if some provider or patient ID available.
    if 'patient_encoded' in feature_columns:
        features['patient_encoded'] = patient_info.get('patient_encoded', 0)
    if 'provider_encoded' in feature_columns:
        features['provider_encoded'] = provider_info.get('provider_encoded', 0)

    for k, v in {**patient_defaults, **provider_defaults}.items():
        if k in features:
            features[k] = v

    # Slot-specific features
    features['appt_hour'] = slot_info.get('hour', 9)
    features['appt_weekday'] = slot_info.get('weekday', 0)
    features['appt_month'] = slot_info.get('month', 1)
    features['appt_day'] = slot_info.get('day', 1)
    features['appt_quarter'] = slot_info.get('appt_quarter', (features['appt_month'] - 1) // 3 + 1)

    features['is_weekend'] = 1 if features['appt_weekday'] >= 5 else 0
    features['is_holiday'] = slot_info.get('is_holiday', 0)
    features['is_peak_day'] = slot_info.get('is_peak_day', 0)
    features['season'] = slot_info.get('season', math.floor((features['appt_month'] - 1) / 3))

    now = datetime.utcnow()
    target_date = datetime(
        slot_info.get('year', now.year),
        int(features['appt_month']),
        int(features['appt_day']),
        int(features['appt_hour']),
    )
    lead_time = (target_date - now).total_seconds() / 86400.0
    features['lead_time_days'] = max(0.0, lead_time)

    features['was_updated'] = slot_info.get('was_updated', 0)

    # Location & contact values
    for key in [
        'visit_reason_encoded',
        'location_encoded',
        'has_overlap',
        'has_primary_insurance',
        'has_secondary_insurance',
        'has_dual_insurance',
        'is_medicare',
        'is_medicaid',
        'is_hmo',
        'secondary_insurance_active',
        'insurance_coverage_days',
        'has_copay',
        'patient_avg_copay',
        'is_home_visit',
        'is_new_patient',
        'is_established_patient',
        'is_telehealth',
        'is_wellness',
        'is_followup',
        'is_lab',
        'is_emergency',
        'is_paid',
        'paidAmount',
        'days_since_last_payment',
        'contact_method_encoded',
        'note_length',
        'insurance_medicalGroup_freq',
        'providerName_freq',
        'serviceLocationName_freq',
    ]:
        if key in features:
            features[key] = slot_info.get(key, patient_info.get(key, provider_info.get(key, 0)))

    # Time slot one-hot
    tod = get_time_of_day(int(features['appt_hour']))
    for key in ['slot_Morning', 'slot_Midday', 'slot_Afternoon', 'slot_Evening']:
        features[key] = 1 if key.lower().endswith(tod) else 0

    # Duration categorization
    dur = slot_info.get('slot_duration_minutes', 60)
    features['dur_Short'] = 1 if dur <= 30 else 0
    features['dur_Medium'] = 1 if 30 < dur <= 60 else 0
    features['dur_Long'] = 1 if dur > 60 else 0

    # Request TF-IDF fields if present
    for i in range(0, 9):
        key = f'note_tfidf_{i}'
        if key in features:
            features[key] = _safe_float(patient_info.get(key, 0.0))

    for i in range(0, 19):
        key = f'reason_tfidf_{i}'
        if key in features:
            features[key] = _safe_float(patient_info.get(key, 0.0))

    # Derived plugs
    features['patient_provider_loyalty'] = _safe_float(
        features.get('patient_provider_history', 0)
        / max(1.0, features.get('patient_total_appts', 1)),
    )

    features['slot_demand_count'] = slot_info.get('slot_demand_count', 1)
    features['slot_popularity_score'] = slot_info.get('slot_popularity_score', 0.0)
    features['slot_historical_success_rate'] = slot_info.get('slot_historical_success_rate', 0.5)
    features['provider_slot_volume'] = slot_info.get('provider_slot_volume', 1)

    features['patient_hist_success_rate'] = patient_info.get('patient_hist_success_rate', 0.5)
    features['provider_hist_success_rate'] = provider_info.get('provider_hist_success_rate', 0.5)
    features['patient_provider_hist_success_rate'] = patient_info.get('patient_provider_hist_success_rate', 0.5)
    features['slot_hist_success_rate_shifted'] = slot_info.get('slot_hist_success_rate_shifted', 0.5)

    # Historical counts fallback
    features['patient_hist_appt_count'] = patient_info.get('patient_hist_appt_count', 0)
    features['provider_hist_appt_count'] = provider_info.get('provider_hist_appt_count', 0)
    features['patient_provider_hist_appt_count'] = patient_info.get('patient_provider_hist_appt_count', 0)
    features['slot_hist_appt_count'] = slot_info.get('slot_hist_appt_count', 0)

    if 'patient_encoded_roll_7D_count' in features:
        features['patient_encoded_roll_7D_count'] = patient_info.get('patient_encoded_roll_7D_count', 0)
        features['patient_encoded_roll_7D_success_rate'] = patient_info.get('patient_encoded_roll_7D_success_rate', 0.5)
        features['patient_encoded_roll_30D_count'] = patient_info.get('patient_encoded_roll_30D_count', 0)
        features['patient_encoded_roll_30D_success_rate'] = patient_info.get('patient_encoded_roll_30D_success_rate', 0.5)
        features['provider_encoded_roll_7D_count'] = provider_info.get('provider_encoded_roll_7D_count', 0)
        features['provider_encoded_roll_7D_success_rate'] = provider_info.get('provider_encoded_roll_7D_success_rate', 0.5)
        features['provider_encoded_roll_30D_count'] = provider_info.get('provider_encoded_roll_30D_count', 0)
        features['provider_encoded_roll_30D_success_rate'] = provider_info.get('provider_encoded_roll_30D_success_rate', 0.5)

    return features


def build_slots_feature_dataframe(
    slots: List[Dict[str, Any]],
    patient_info: Dict[str, Any],
    provider_info: Dict[str, Any],
    feature_columns: List[str],
) -> pd.DataFrame:
    """Build DataFrame for multiple slots."""
    rows = []
    for slot in slots:
        features = build_slot_features(patient_info, provider_info, slot, feature_columns)
        rows.append(features)
    return pd.DataFrame(rows, columns=feature_columns)
