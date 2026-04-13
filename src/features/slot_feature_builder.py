import math
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


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
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _normalize_rate(value: Any, default: float = 0.0) -> float:
    return min(1.0, max(0.0, _safe_float(value, default)))


def _time_bucket_score(hour: int, preferred_bucket: str) -> float:
    return 1.0 if get_time_of_day(hour) == preferred_bucket else 0.0


def _encode_time_slot(hour: int, bucket: str) -> int:
    return 1 if get_time_of_day(hour) == bucket else 0


def _days_ahead(slot_date: str) -> float:
    try:
        target = datetime.fromisoformat(slot_date)
        return max(0.0, (target - datetime.utcnow()).days)
    except Exception:
        return 0.0


def _compute_reliability_score(cancel_rate: float, no_show_rate: float) -> float:
    score = 1.0 - (cancel_rate * 0.55 + no_show_rate * 0.45)
    return min(1.0, max(0.0, score))


def build_slot_features(
    patient_info: Dict[str, Any],
    provider_info: Dict[str, Any],
    slot_info: Dict[str, Any],
    feature_columns: List[str],
) -> Dict[str, Any]:
    """Build one feature vector row for the slot prediction model."""
    features: Dict[str, Any] = {c: 0.0 for c in feature_columns}

    patient_cancel_rate = _normalize_rate(patient_info.get('patient_cancel_rate', 0.0))
    patient_no_show_rate = _normalize_rate(patient_info.get('patient_no_show_rate', patient_info.get('patient_no_show_rate', 0.0)))
    patient_total_appts = max(1.0, _safe_float(patient_info.get('patient_total_appts', 1.0)))
    patient_avg_lead_time = _safe_float(patient_info.get('patient_avg_booking_lead_time', patient_info.get('patient_avg_lead_time', 7.0)))
    patient_visit_frequency = _normalize_rate(patient_info.get('patient_visit_frequency', min(1.0, patient_total_appts / 12.0)))
    patient_reliability_score = _compute_reliability_score(patient_cancel_rate, patient_no_show_rate)
    patient_preferred_time = patient_info.get('patient_preferred_time', '')
    if not patient_preferred_time:
        patient_preferred_time = 'morning' if patient_total_appts < 3 else 'afternoon'

    provider_utilization = _normalize_rate(provider_info.get('provider_utilization_rate', provider_info.get('provider_7day_util', provider_info.get('provider_utilization', 0.5))))
    provider_overbooking_ratio = _normalize_rate(provider_info.get('provider_overbooking_ratio', 0.0))
    provider_avg_daily_appts = _safe_float(provider_info.get('provider_avg_daily_appointments', provider_info.get('provider_avg_daily_appointments', 3.0)))
    provider_total_appts = max(1.0, _safe_float(provider_info.get('provider_total_appts', 1.0)))
    provider_cancellation_rate = _normalize_rate(provider_info.get('provider_cancellation_rate', provider_info.get('provider_cancel_rate', 0.0)))
    provider_peak_hours = provider_info.get('provider_peak_hours', [])
    provider_peak_hour_score = 1.0 if slot_info.get('hour') in provider_peak_hours else 0.0

    slot_hour = int(slot_info.get('hour', 9))
    slot_weekday = int(slot_info.get('weekday', 0))
    slot_days_ahead = _days_ahead(slot_info.get('date', datetime.utcnow().date().isoformat()))
    slot_popularity_score = _normalize_rate(slot_info.get('slot_popularity_score', 0.0))
    slot_success_rate = _normalize_rate(slot_info.get('slot_historical_success_rate', slot_info.get('slot_success_rate', 0.5)))
    slot_demand_count = max(0.0, _safe_float(slot_info.get('slot_demand_count', 0.0)))
    slot_quality_score = _normalize_rate(slot_info.get('slot_quality_score', min(1.0, 0.2 + slot_popularity_score * 0.6 + slot_success_rate * 0.4)))
    patient_provider_history = _safe_float(patient_info.get('patient_provider_history', patient_info.get('patient_provider_hist_appt_count', 0)))
    provider_hist_appt_count = _safe_float(provider_info.get('provider_hist_appt_count', provider_info.get('provider_total_appts', 1)))
    patient_provider_history_score = min(1.0, patient_provider_history / max(1.0, patient_total_appts))
    time_slot_affinity_score = max(patient_preferred_time == get_time_of_day(slot_hour), provider_peak_hour_score)
    patient_time_preference_match = _time_bucket_score(slot_hour, patient_preferred_time)

    if 'patient_encoded' in features:
        features['patient_encoded'] = _safe_float(patient_info.get('patient_encoded', 0.0))
    if 'provider_encoded' in features:
        features['provider_encoded'] = _safe_float(provider_info.get('provider_encoded', 0.0))

    slot_defaults = {
        'appt_hour': slot_hour,
        'appt_weekday': slot_weekday,
        'appt_month': int(slot_info.get('month', datetime.utcnow().month)),
        'appt_day': int(slot_info.get('day', datetime.utcnow().day)),
        'appt_quarter': int(slot_info.get('appt_quarter', (int(slot_info.get('month', datetime.utcnow().month)) - 1) // 3 + 1)),
        'is_weekend': 1 if slot_weekday >= 5 else 0,
        'is_holiday': _normalize_rate(slot_info.get('is_holiday', 0.0)),
        'is_peak_hour': 1 if slot_hour in (8, 9, 10, 14, 15) else 0,
        'slot_hour_of_day': slot_hour,
        'slot_day_of_week': slot_weekday,
        'slot_is_peak_hour': 1 if slot_hour in (8, 9, 10, 14, 15) else 0,
        'slot_days_ahead': slot_days_ahead,
        'slot_popularity_score': slot_popularity_score,
        'slot_success_rate': slot_success_rate,
        'slot_quality_score': slot_quality_score,
        'slot_demand_count': slot_demand_count,
        'provider_avg_daily_appointments': provider_avg_daily_appts,
        'provider_utilization_rate': provider_utilization,
        'provider_overbooking_ratio': provider_overbooking_ratio,
        'provider_cancellation_rate': provider_cancellation_rate,
        'provider_peak_hour_score': provider_peak_hour_score,
        'patient_no_show_rate': patient_no_show_rate,
        'patient_cancel_rate': patient_cancel_rate,
        'patient_avg_booking_lead_time': patient_avg_lead_time,
        'patient_visit_frequency': patient_visit_frequency,
        'patient_reliability_score': patient_reliability_score,
        'patient_time_preference_match': patient_time_preference_match,
        'patient_provider_history_score': patient_provider_history_score,
        'time_slot_affinity_score': float(time_slot_affinity_score),
    }

    patient_defaults = {
        'patient_age': _safe_float(patient_info.get('patient_age', 35)),
        'sex_encoded': _safe_float(patient_info.get('sex_encoded', 1)),
        'patient_total_appts': patient_total_appts,
        'patient_7day_appts': _safe_float(patient_info.get('patient_7day_appts', 0)),
        'patient_30day_appts': _safe_float(patient_info.get('patient_30day_appts', 0)),
        'patient_7day_cancel': _safe_float(patient_info.get('patient_7day_cancel', 0)),
        'patient_30day_cancel': _safe_float(patient_info.get('patient_30day_cancel', 0)),
        'patient_provider_history': patient_provider_history,
        'patient_preferred_time': patient_preferred_time,
    }

    provider_defaults = {
        'provider_total_appts': _safe_float(provider_info.get('provider_total_appts', 1)),
        'provider_avg_duration': _safe_float(provider_info.get('provider_avg_duration', 30)),
        'provider_cancel_rate': provider_cancellation_rate,
        'provider_7day_util': _normalize_rate(provider_info.get('provider_7day_util', provider_utilization)),
        'provider_30day_util': _normalize_rate(provider_info.get('provider_30day_util', provider_utilization)),
        'provider_utilization': provider_utilization,
        'provider_slot_volume': _safe_float(provider_info.get('provider_slot_volume', max(1.0, provider_total_appts))),
        'provider_peak_hours': provider_peak_hours,
    }

    for key, value in {**slot_defaults, **patient_defaults, **provider_defaults}.items():
        if key in features:
            features[key] = value

    for bucket in TIME_SLOT_RANGES:
        key = f'patient_pref_{bucket}'
        if key in features:
            features[key] = _encode_time_slot(slot_hour, bucket)
        key = f'provider_peak_{bucket}'
        if key in features:
            features[key] = 1 if get_time_of_day(slot_hour) == bucket and provider_peak_hour_score else 0

    for i in range(0, 9):
        key = f'note_tfidf_{i}'
        if key in features:
            features[key] = _safe_float(patient_info.get(key, 0.0))

    for i in range(0, 19):
        key = f'reason_tfidf_{i}'
        if key in features:
            features[key] = _safe_float(patient_info.get(key, 0.0))

    if 'patient_provider_loyalty' in features:
        features['patient_provider_loyalty'] = _normalize_rate(patient_provider_history_score)

    if 'lead_time_days' in features:
        features['lead_time_days'] = slot_days_ahead

    if 'slot_popularity_score' in features and features['slot_popularity_score'] == 0.0:
        features['slot_popularity_score'] = max(0.05, 0.15 + slot_success_rate * 0.4)

    return features


def build_slots_feature_dataframe(
    slots: List[Dict[str, Any]],
    patient_info: Dict[str, Any],
    provider_info: Dict[str, Any],
    feature_columns: List[str],
) -> pd.DataFrame:
    """Build DataFrame for multiple slots."""
    rows = [build_slot_features(patient_info, provider_info, slot, feature_columns) for slot in slots]
    df = pd.DataFrame(rows, columns=feature_columns)
    if not df.empty:
        variance = df.var(numeric_only=True)
        zero_variance = [col for col, val in variance.items() if val == 0]
        logger.info(
            "Built slot feature dataframe",
            extra={
                "extra": {
                    "rows": len(df),
                    "min_variance": float(variance.min()) if not variance.empty else 0.0,
                    "max_variance": float(variance.max()) if not variance.empty else 0.0,
                    "zero_variance_features": len(zero_variance),
                    "zero_variance_columns": zero_variance,
                }
            },
        )
    return df
