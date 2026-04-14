"""
Slot Feature Builder — Production-Grade Intelligence Layer
==========================================================
Every slot must produce a UNIQUE, high-variance feature vector.
Patient-level features are constant across slots (correct).
Provider-level features are constant across slots (correct).
Slot-level + interaction features MUST vary per slot.
"""
import math
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

TIME_SLOT_RANGES = {
    "morning":   range(8, 11),
    "midday":    range(11, 13),
    "afternoon": range(13, 16),
    "evening":   range(16, 20),
}

# Hour-of-day demand weights derived from healthcare scheduling research
# (morning peak, post-lunch dip, afternoon recovery)
_HOUR_DEMAND_WEIGHTS = {
    8: 0.90, 9: 1.00, 10: 0.95, 11: 0.80, 12: 0.55,
    13: 0.60, 14: 0.85, 15: 0.80, 16: 0.65, 17: 0.45,
    18: 0.30, 19: 0.20,
}

# Weekday demand weights (Mon/Fri higher, Wed lower)
_WEEKDAY_DEMAND_WEIGHTS = {0: 0.90, 1: 0.85, 2: 0.75, 3: 0.80, 4: 0.88, 5: 0.30, 6: 0.20}


def get_time_of_day(hour: int) -> str:
    for name, rng in TIME_SLOT_RANGES.items():
        if hour in rng:
            return name
    return "other"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _clip(value: Any, lo: float = 0.0, hi: float = 1.0, default: float = 0.0) -> float:
    return min(hi, max(lo, _safe_float(value, default)))


def _days_ahead(slot_date: str) -> float:
    try:
        return max(0.0, (datetime.fromisoformat(slot_date) - datetime.utcnow()).days)
    except Exception:
        return 0.0


# ── Slot-level signals (MUST vary per slot) ───────────────────────────────────

def _hour_demand_score(hour: int) -> float:
    """Intrinsic demand weight for this hour based on healthcare patterns."""
    return _HOUR_DEMAND_WEIGHTS.get(hour, 0.4)


def _weekday_demand_score(weekday: int) -> float:
    """Intrinsic demand weight for this weekday."""
    return _WEEKDAY_DEMAND_WEIGHTS.get(weekday, 0.5)


def _hour_weekday_interaction(hour: int, weekday: int) -> float:
    """Cross-feature: combined demand signal for (hour, weekday) pair."""
    return round(_hour_demand_score(hour) * _weekday_demand_score(weekday), 4)


def _provider_hour_load(hour: int, hourly_util_curve: Dict[int, float]) -> float:
    """Provider's historical load at this specific hour (0–1)."""
    return _clip(hourly_util_curve.get(hour, 0.0))


def _provider_hour_overload(hour: int, hourly_util_curve: Dict[int, float]) -> float:
    """1.0 if this hour is above provider's mean utilization, else 0."""
    if not hourly_util_curve:
        return 0.0
    mean_load = sum(hourly_util_curve.values()) / len(hourly_util_curve)
    return 1.0 if hourly_util_curve.get(hour, 0.0) > mean_load else 0.0


def _congestion_index(
    hour: int,
    weekday: int,
    slot_demand_count: float,
    provider_avg_daily_appts: float,
) -> float:
    """Slot congestion: how busy is this slot relative to provider capacity."""
    base = _hour_weekday_interaction(hour, weekday)
    volume_factor = min(1.0, slot_demand_count / max(1.0, provider_avg_daily_appts))
    return round(min(1.0, base * 0.6 + volume_factor * 0.4), 4)


def _lead_time_urgency(days_ahead: float) -> float:
    """Urgency score: same-day=1.0, 2 weeks=0.0 (exponential decay)."""
    return round(math.exp(-days_ahead / 7.0), 4)


def _patient_provider_affinity(
    patient_provider_history: float,
    patient_total_appts: float,
) -> float:
    """Loyalty score: fraction of patient's appointments with this provider."""
    return _clip(patient_provider_history / max(1.0, patient_total_appts))


def _patient_hour_affinity(hour: int, patient_preferred_time: str) -> float:
    """1.0 if slot hour matches patient's historically preferred time bucket."""
    return 1.0 if get_time_of_day(hour) == patient_preferred_time else 0.0


def _provider_peak_affinity(hour: int, provider_peak_hour_set: List[int]) -> float:
    """1.0 if this hour is in provider's historically busy hours."""
    return 1.0 if hour in provider_peak_hour_set else 0.0


def _slot_success_confidence(success_rate: float, total_count: float) -> float:
    """Bayesian-smoothed success rate: low-count slots regress toward 0.5."""
    prior = 0.5
    weight = min(1.0, total_count / 20.0)  # full confidence at 20+ observations
    return round(prior * (1 - weight) + success_rate * weight, 4)


def _patient_noshow_risk(no_show_rate: float, cancel_rate: float, lead_time_days: float) -> float:
    """Composite no-show risk: higher lead time increases risk."""
    base_risk = no_show_rate * 0.6 + cancel_rate * 0.4
    lead_factor = min(1.0, lead_time_days / 14.0) * 0.2  # up to +20% for far-future slots
    return _clip(base_risk + lead_factor)


# ── Main builder ──────────────────────────────────────────────────────────────

def build_slot_features(
    patient_info: Dict[str, Any],
    provider_info: Dict[str, Any],
    slot_info: Dict[str, Any],
    feature_columns: List[str],
) -> Dict[str, Any]:
    """Build one feature vector row. Every slot produces a unique vector."""
    features: Dict[str, Any] = {c: 0.0 for c in feature_columns}

    # ── Patient signals ───────────────────────────────────────────────────────
    p_cancel_rate    = _clip(patient_info.get("patient_cancel_rate", 0.0))
    p_noshow_rate    = _clip(patient_info.get("patient_no_show_rate", 0.0))
    p_total_appts    = max(1.0, _safe_float(patient_info.get("patient_total_appts", 1.0)))
    p_avg_lead_time  = _safe_float(patient_info.get("patient_avg_booking_lead_time", 7.0))
    p_preferred_time = patient_info.get("patient_preferred_time", "morning") or "morning"
    p_7day_appts     = _safe_float(patient_info.get("patient_7day_appts", 0.0))
    p_30day_appts    = _safe_float(patient_info.get("patient_30day_appts", 0.0))
    p_7day_cancel    = _safe_float(patient_info.get("patient_7day_cancel", 0.0))
    p_30day_cancel   = _safe_float(patient_info.get("patient_30day_cancel", 0.0))
    p_provider_hist  = _safe_float(patient_info.get("patient_provider_history", 0.0))
    p_reliability    = _clip(patient_info.get("patient_reliability_score",
                             1.0 - min(1.0, p_cancel_rate * 0.55 + p_noshow_rate * 0.45)))
    p_visit_freq     = _clip(patient_info.get("patient_visit_frequency",
                             min(1.0, p_total_appts / 12.0)))

    # ── Provider signals ──────────────────────────────────────────────────────
    pv_util          = _clip(provider_info.get("provider_utilization_rate",
                             provider_info.get("provider_7day_util",
                             provider_info.get("provider_utilization", 0.5))))
    pv_util_7d       = _clip(provider_info.get("provider_7day_util", pv_util))
    pv_util_30d      = _clip(provider_info.get("provider_30day_util", pv_util))
    pv_cancel_rate   = _clip(provider_info.get("provider_cancellation_rate",
                             provider_info.get("provider_cancel_rate", 0.0)))
    pv_avg_daily     = _safe_float(provider_info.get("provider_avg_daily_appointments", 3.0))
    pv_total_appts   = max(1.0, _safe_float(provider_info.get("provider_total_appts", 1.0)))
    pv_overbook      = _clip(provider_info.get("provider_overbooking_ratio", 0.0))
    pv_peak_hours    = provider_info.get("provider_peak_hours", []) or []
    pv_peak_hour_set = provider_info.get("provider_peak_hour_set", []) or []
    pv_hourly_curve  = provider_info.get("provider_hourly_util_curve", {}) or {}

    # ── Slot signals (VARY per slot) ──────────────────────────────────────────
    slot_hour    = int(slot_info.get("hour", 9))
    slot_weekday = int(slot_info.get("weekday", 0))
    slot_date    = slot_info.get("date", datetime.utcnow().date().isoformat())
    slot_month   = int(slot_info.get("month", datetime.utcnow().month))
    slot_day     = int(slot_info.get("day", datetime.utcnow().day))
    slot_quarter = int(slot_info.get("appt_quarter", (slot_month - 1) // 3 + 1))
    days_ahead   = _safe_float(slot_info.get("slot_days_ahead", _days_ahead(slot_date)))

    # Slot-level DB stats (vary per slot when slot_statistics is populated)
    raw_success_rate  = _clip(slot_info.get("slot_historical_success_rate",
                              slot_info.get("slot_success_rate", 0.5)))
    raw_demand_count  = max(0.0, _safe_float(slot_info.get("slot_demand_count", 0.0)))
    raw_popularity    = _clip(slot_info.get("slot_popularity_score", 0.0))

    # ── Derived slot-level features (always vary by hour/weekday) ─────────────
    hour_demand       = _hour_demand_score(slot_hour)
    weekday_demand    = _weekday_demand_score(slot_weekday)
    hw_interaction    = _hour_weekday_interaction(slot_hour, slot_weekday)
    pv_hour_load      = _provider_hour_load(slot_hour, pv_hourly_curve)
    pv_hour_overload  = _provider_hour_overload(slot_hour, pv_hourly_curve)
    congestion_idx    = _congestion_index(slot_hour, slot_weekday, raw_demand_count, pv_avg_daily)
    lead_urgency      = _lead_time_urgency(days_ahead)
    success_conf      = _slot_success_confidence(raw_success_rate, raw_demand_count)

    # Popularity: if DB has data use it; otherwise derive from hour/weekday demand
    if raw_popularity > 0.0:
        slot_popularity = raw_popularity
    else:
        slot_popularity = round(0.3 + hw_interaction * 0.5 + success_conf * 0.2, 4)

    # ── Interaction features (patient × slot, provider × slot) ────────────────
    p_hour_affinity   = _patient_hour_affinity(slot_hour, p_preferred_time)
    pv_peak_affinity  = _provider_peak_affinity(slot_hour, pv_peak_hour_set or pv_peak_hours)
    p_pv_affinity     = _patient_provider_affinity(p_provider_hist, p_total_appts)
    p_noshow_risk     = _patient_noshow_risk(p_noshow_rate, p_cancel_rate, days_ahead)

    # Provider load × patient reliability interaction
    load_reliability_interaction = round(pv_hour_load * (1.0 - p_reliability), 4)
    # Hour demand × patient preference match
    demand_pref_interaction = round(hw_interaction * p_hour_affinity, 4)

    # ── Assemble feature dict ─────────────────────────────────────────────────
    values = {
        # IDs
        "patient_encoded":                  _safe_float(patient_info.get("patient_encoded", 0.0)),
        "provider_encoded":                 _safe_float(provider_info.get("provider_encoded", 0.0)),

        # Temporal (vary per slot)
        "appt_hour":                        slot_hour,
        "appt_weekday":                     slot_weekday,
        "appt_month":                       slot_month,
        "appt_day":                         slot_day,
        "appt_quarter":                     slot_quarter,
        "is_weekend":                       1 if slot_weekday >= 5 else 0,
        "is_holiday":                       _clip(slot_info.get("is_holiday", 0.0)),
        "is_peak_hour":                     1 if slot_hour in (8, 9, 10, 14, 15) else 0,
        "slot_hour_of_day":                 slot_hour,
        "slot_day_of_week":                 slot_weekday,
        "slot_is_peak_hour":                1 if slot_hour in (8, 9, 10, 14, 15) else 0,
        "slot_days_ahead":                  days_ahead,
        "lead_time_days":                   days_ahead,

        # Slot demand signals (vary per slot)
        "slot_popularity_score":            slot_popularity,
        "slot_success_rate":                success_conf,
        "slot_quality_score":               round(0.3 + success_conf * 0.4 + slot_popularity * 0.3, 4),
        "slot_demand_count":                raw_demand_count,

        # Hour/weekday intrinsic demand (vary per slot)
        "hour_demand_score":                hour_demand,
        "weekday_demand_score":             weekday_demand,
        "hour_weekday_interaction":         hw_interaction,
        "lead_time_urgency":                lead_urgency,
        "congestion_index":                 congestion_idx,

        # Provider signals (constant per request, but real values)
        "provider_avg_daily_appointments":  pv_avg_daily,
        "provider_utilization_rate":        pv_util,
        "provider_utilization":             pv_util,
        "provider_overbooking_ratio":       pv_overbook,
        "provider_cancellation_rate":       pv_cancel_rate,
        "provider_cancel_rate":             pv_cancel_rate,
        "provider_peak_hour_score":         pv_peak_affinity,
        "provider_7day_util":               pv_util_7d,
        "provider_30day_util":              pv_util_30d,
        "provider_total_appts":             pv_total_appts,
        "provider_avg_duration":            _safe_float(provider_info.get("provider_avg_duration", 30)),
        "provider_slot_volume":             max(1.0, pv_total_appts),
        "provider_hist_success_rate":       _clip(provider_info.get("provider_hist_success_rate", 0.5)),

        # Provider × slot interactions (vary per slot)
        "provider_hour_load":               pv_hour_load,
        "provider_hour_overload":           pv_hour_overload,

        # Patient signals (constant per request, but real values)
        "patient_age":                      _safe_float(patient_info.get("patient_age", 35)),
        "sex_encoded":                      _safe_float(patient_info.get("sex_encoded", 1)),
        "patient_total_appts":              p_total_appts,
        "patient_cancel_rate":              p_cancel_rate,
        "patient_no_show_rate":             p_noshow_rate,
        "patient_7day_appts":               p_7day_appts,
        "patient_30day_appts":              p_30day_appts,
        "patient_7day_cancel":              p_7day_cancel,
        "patient_30day_cancel":             p_30day_cancel,
        "patient_avg_booking_lead_time":    p_avg_lead_time,
        "patient_visit_frequency":          p_visit_freq,
        "patient_reliability_score":        p_reliability,
        "patient_provider_history":         p_provider_hist,
        "patient_provider_loyalty":         p_pv_affinity,
        "patient_provider_history_score":   p_pv_affinity,
        "patient_hist_success_rate":        _clip(patient_info.get("patient_hist_success_rate", 0.5)),
        "patient_avg_duration":             _safe_float(patient_info.get("patient_avg_duration", 30.0)),

        # Patient × slot interactions (vary per slot)
        "patient_time_preference_match":    p_hour_affinity,
        "time_slot_affinity_score":         max(p_hour_affinity, pv_peak_affinity),
        "patient_noshow_risk":              p_noshow_risk,
        "patient_provider_affinity":        p_pv_affinity,
        "load_reliability_interaction":     load_reliability_interaction,
        "demand_pref_interaction":          demand_pref_interaction,

        # Insurance / demographics
        "has_primary_insurance":            int(patient_info.get("has_primary_insurance", 1)),
        "has_secondary_insurance":          int(patient_info.get("has_secondary_insurance", 0)),
        "is_medicare":                      int(patient_info.get("is_medicare", 0)),
        "is_medicaid":                      int(patient_info.get("is_medicaid", 0)),
        "is_hmo":                           int(patient_info.get("is_hmo", 0)),
        "patient_avg_copay":                _safe_float(patient_info.get("patient_avg_copay", 0.0)),

        # Rolling windows
        "patient_encoded_roll_7D_count":    _safe_float(patient_info.get("patient_encoded_roll_7D_count", p_7day_appts)),
        "patient_encoded_roll_7D_success_rate": _clip(patient_info.get("patient_encoded_roll_7D_success_rate", 0.5)),
        "patient_encoded_roll_30D_count":   _safe_float(patient_info.get("patient_encoded_roll_30D_count", p_30day_appts)),
        "patient_encoded_roll_30D_success_rate": _clip(patient_info.get("patient_encoded_roll_30D_success_rate", 0.5)),

        # Time bucket one-hot (vary per slot)
        "slot_Morning":   1 if slot_hour in TIME_SLOT_RANGES["morning"] else 0,
        "slot_Midday":    1 if slot_hour in TIME_SLOT_RANGES["midday"] else 0,
        "slot_Afternoon": 1 if slot_hour in TIME_SLOT_RANGES["afternoon"] else 0,
        "slot_Evening":   1 if slot_hour in TIME_SLOT_RANGES["evening"] else 0,
    }

    # Patient time-bucket preferences (constant per request)
    for bucket in TIME_SLOT_RANGES:
        values[f"patient_pref_{bucket}"] = int(patient_info.get(f"patient_pref_{bucket}", 0))
        values[f"provider_peak_{bucket}"] = (
            1 if get_time_of_day(slot_hour) == bucket and pv_peak_affinity else 0
        )

    # TF-IDF pass-through (constant per patient)
    for i in range(9):
        values[f"note_tfidf_{i}"] = _safe_float(patient_info.get(f"note_tfidf_{i}", 0.0))
    for i in range(19):
        values[f"reason_tfidf_{i}"] = _safe_float(patient_info.get(f"reason_tfidf_{i}", 0.0))

    # Write only columns that exist in the model's feature_columns
    for key, val in values.items():
        if key in features:
            features[key] = val

    return features


def build_slots_feature_dataframe(
    slots: List[Dict[str, Any]],
    patient_info: Dict[str, Any],
    provider_info: Dict[str, Any],
    feature_columns: List[str],
) -> pd.DataFrame:
    """Build DataFrame for multiple slots with variance validation."""
    rows = [build_slot_features(patient_info, provider_info, slot, feature_columns) for slot in slots]
    df = pd.DataFrame(rows, columns=feature_columns)

    if not df.empty:
        numeric_df = df.select_dtypes(include="number")
        variance = numeric_df.var()
        zero_var_cols = variance[variance == 0].index.tolist()
        varying_cols  = variance[variance > 1e-8].index.tolist()

        logger.info(
            "Feature matrix built",
            extra={"extra": {
                "rows": len(df),
                "total_features": len(numeric_df.columns),
                "varying_features": len(varying_cols),
                "zero_variance_features": len(zero_var_cols),
                "min_variance": float(variance.min()) if not variance.empty else 0.0,
                "max_variance": float(variance.max()) if not variance.empty else 0.0,
            }},
        )

        if len(slots) > 1 and len(varying_cols) < 3:
            logger.warning(
                "CRITICAL: Only %d features vary across %d slots. "
                "Ensure slot_statistics collection is populated via refresh_slot_statistics().",
                len(varying_cols), len(slots),
            )

    return df
