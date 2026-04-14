import re
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from bson import ObjectId
from pymongo.database import Database

from src.database.db_connection import get_collection_names
from src.database.errors import EntityNotFoundError
from src.utils.logger import get_logger

logger = get_logger(__name__)

CONFIRMED_STATUSES = {"Confirmed", "Confirmed           "}


def _cols(db: Database) -> dict:
    return get_collection_names()


def _to_oid(id_str: str) -> Optional[ObjectId]:
    try:
        return ObjectId(id_str)
    except Exception:
        return None


# ─── Patient ──────────────────────────────────────────────────────────────────

def get_patient_data(db: Database, patient_id: str, provider_id: Optional[str] = None) -> Dict[str, Any]:
    """Fetch patient by MongoDB _id. Returns ML-compatible feature dict."""
    col = _cols(db)
    oid = _to_oid(patient_id)
    patient = db[col["patients"]].find_one({"_id": oid}) if oid else None

    if not patient:
        logger.warning("[get_patient_data] not found: id=%s — strict validation applies", patient_id)
        return _default_patient(patient_id)

    now = datetime.utcnow()
    window_7 = now - timedelta(days=7)
    window_30 = now - timedelta(days=30)

    appts = list(db[col["appointments"]].find({"patient_id": patient_id}))
    total = len(appts)
    successful = [a for a in appts if str(a.get("status", "")).strip() in CONFIRMED_STATUSES]
    cancelled = [a for a in appts if str(a.get("status", "")).strip() not in CONFIRMED_STATUSES and str(a.get("status", "")).strip() != "No Show"]
    no_shows = [a for a in appts if str(a.get("status", "")).strip().lower() in {"no show", "no-show", "noshow"}]
    appts_7d = [a for a in appts if _parse_date(a.get("appt_date")) >= window_7]
    appts_30d = [a for a in appts if _parse_date(a.get("appt_date")) >= window_30]
    average_lead_time = _compute_avg_lead_time(appts)
    time_buckets = _build_time_bucket_profile(appts)

    success_rate = len(successful) / max(1, total)
    cancel_rate = len(cancelled) / max(1, total)
    no_show_rate = len(no_shows) / max(1, total)
    reliability_score = _compute_reliability_score(cancel_rate, no_show_rate)
    visit_frequency = min(1.0, total / max(1.0, max(1.0, _active_days(appts) / 30.0)))
    preferred_time = max(time_buckets, key=time_buckets.get) if time_buckets else "morning"

    provider_history = [a for a in appts if provider_id and a.get("provider_id") == provider_id] if provider_id else []
    provider_history_total = len(provider_history)
    provider_history_confirmed = [a for a in provider_history if str(a.get("status", "")).strip() in CONFIRMED_STATUSES]
    provider_history_success_rate = round(len(provider_history_confirmed) / max(1, provider_history_total), 4) if provider_history_total else 0.5

    patient_encoded = _extract_numeric_id(patient, ["patient_encoded", "patientId", "patientID"])
    if patient_encoded is None:
        patient_encoded = _stable_hash(patient_id)

    sex_raw = patient.get("sex", patient.get("gender", ""))
    sex_encoded = 0 if str(sex_raw).strip().upper() in ("F", "FEMALE") else 1

    insurance = patient.get("insurance", {}) or {}
    ins_type = str(insurance.get("insuranceType", "")).lower()
    has_ins = str(patient.get("hasInsurance", "no")).lower() == "yes"
    has_sec = str(patient.get("hasSecondaryInsurance", "no")).lower() == "yes"
    is_medicare = "medicare" in ins_type
    is_medicaid = "medicaid" in ins_type
    is_hmo = "hmo" in ins_type
    copay_raw = insurance.get("coPayAmount", "")
    try:
        avg_copay = float(copay_raw) if copay_raw else 0.0
    except (ValueError, TypeError):
        avg_copay = 0.0

    age = 35
    dob = patient.get("dOB", patient.get("dob", ""))
    if dob:
        try:
            age = (date.today() - datetime.strptime(dob, "%m/%d/%Y").date()).days // 365
        except (ValueError, TypeError):
            age = 35

    avg_duration = 0.0
    if appts:
        duration_values = [float(a.get("duration_minutes", 0)) for a in appts if a.get("duration_minutes") is not None]
        avg_duration = sum(duration_values) / max(1, len(duration_values)) if duration_values else 0.0

    return {
        "patient_id":                         patient_id,
        "patient_encoded":                    patient_encoded,
        "patient_age":                        age,
        "sex_encoded":                        sex_encoded,
        "patient_total_appts":                total,
        "patient_cancel_rate":                round(cancel_rate, 4),
        "patient_no_show_rate":               round(no_show_rate, 4),
        "patient_7day_appts":                 len(appts_7d),
        "patient_30day_appts":                len(appts_30d),
        "patient_7day_cancel":                len(cancelled),
        "patient_30day_cancel":               len(cancelled),
        "patient_7day_activity":              len(appts_7d),
        "patient_30day_activity":             len(appts_30d),
        "patient_hist_success_rate":          round(success_rate, 4),
        "patient_hist_appt_count":            total,
        "patient_avg_duration":               round(avg_duration, 2),
        "patient_avg_booking_lead_time":      round(average_lead_time, 2),
        "patient_visit_frequency":            round(visit_frequency, 4),
        "patient_reliability_score":          round(reliability_score, 4),
        "patient_preferred_time":             preferred_time,
        "patient_provider_history":           provider_history_total,
        "patient_provider_success_rate":      provider_history_success_rate,
        "patient_provider_hist_appt_count":   provider_history_total,
        "patient_provider_loyalty":           round(provider_history_total / max(1, total), 4),
        "has_primary_insurance":              int(has_ins),
        "has_secondary_insurance":            int(has_sec),
        "is_medicare":                        int(is_medicare),
        "is_medicaid":                        int(is_medicaid),
        "is_hmo":                             int(is_hmo),
        "patient_avg_copay":                  avg_copay,
        "patient_encoded_roll_7D_count":       len(appts_7d),
        "patient_encoded_roll_7D_success_rate": round(len([a for a in appts_7d if str(a.get("status", "")).strip() in CONFIRMED_STATUSES]) / max(1, len(appts_7d)), 4) if appts_7d else 0.5,
        "patient_encoded_roll_30D_count":      len(appts_30d),
        "patient_encoded_roll_30D_success_rate": round(len([a for a in appts_30d if str(a.get("status", "")).strip() in CONFIRMED_STATUSES]) / max(1, len(appts_30d)), 4) if appts_30d else 0.5,
        **time_buckets,
    }


def _default_patient(patient_id: str) -> Dict[str, Any]:
    return {
        "patient_id":                        patient_id,
        "patient_encoded":                   _stable_hash(patient_id),
        "patient_age":                       35,
        "sex_encoded":                       1,
        "patient_total_appts":               1,
        "patient_cancel_rate":               0.0,
        "patient_no_show_rate":              0.0,
        "patient_7day_appts":                0,
        "patient_30day_appts":               0,
        "patient_7day_cancel":               0,
        "patient_30day_cancel":              0,
        "patient_7day_activity":             0,
        "patient_30day_activity":            0,
        "patient_hist_success_rate":         0.5,
        "patient_hist_appt_count":           0,
        "patient_avg_duration":              30.0,
        "patient_avg_booking_lead_time":     7.0,
        "patient_visit_frequency":           0.0,
        "patient_reliability_score":         0.5,
        "patient_preferred_time":            "morning",
        "patient_provider_history":          0,
        "patient_provider_success_rate":     0.5,
        "patient_provider_hist_appt_count":  0,
        "patient_provider_loyalty":          0.0,
        "patient_encoded_roll_7D_count":      0,
        "patient_encoded_roll_7D_success_rate": 0.5,
        "patient_encoded_roll_30D_count":     0,
        "patient_encoded_roll_30D_success_rate": 0.5,
        "has_primary_insurance":             1,
        "has_secondary_insurance":           0,
        "is_medicare":                       0,
        "is_medicaid":                       0,
        "is_hmo":                            0,
        "patient_avg_copay":                 0.0,
        "patient_pref_morning":              1,
        "patient_pref_midday":               0,
        "patient_pref_afternoon":            0,
        "patient_pref_evening":              0,
    }


# ─── Provider ─────────────────────────────────────────────────────────────────

def get_provider_data(db: Database, provider_id: str) -> Dict[str, Any]:
    """Fetch provider by MongoDB _id. Returns ML-compatible feature dict."""
    col = _cols(db)
    oid = _to_oid(provider_id)
    provider = db[col["providers"]].find_one({"_id": oid}) if oid else None

    if not provider:
        logger.warning("[get_provider_data] not found: id=%s — strict validation applies", provider_id)
        return _default_provider(provider_id)

    now = datetime.utcnow()
    window_7 = now - timedelta(days=7)
    window_30 = now - timedelta(days=30)

    appts = list(db[col["appointments"]].find({"provider_id": provider_id}))
    total = len(appts)
    confirmed = [a for a in appts if str(a.get("status", "")).strip() in CONFIRMED_STATUSES]
    cancelled = [a for a in appts if str(a.get("status", "")).strip() not in CONFIRMED_STATUSES and str(a.get("status", "")).strip().lower() != "no show"]
    no_shows = [a for a in appts if str(a.get("status", "")).strip().lower() in {"no show", "no-show", "noshow"}]
    appts_7d = [a for a in appts if _parse_date(a.get("appt_date")) >= window_7]
    appts_30d = [a for a in appts if _parse_date(a.get("appt_date")) >= window_30]
    max_daily = int(provider.get("max_daily_slots", 16))
    daily_active_days = max(1, _active_days(appts))
    util_7d = len(appts_7d) / max(1, max_daily * 7)
    util_30d = len(appts_30d) / max(1, max_daily * 30)
    success_rate = len(confirmed) / max(1, total)
    cancellation_rate = len(cancelled) / max(1, total)
    no_show_rate = len(no_shows) / max(1, total)
    overbooking_ratio = min(1.0, total / max(1.0, max_daily * daily_active_days))
    peak_hours = _top_time_buckets([a.get("appt_hour") for a in appts])
    avg_daily_appointments = total / daily_active_days

    provider_encoded = _extract_numeric_id(provider, ["provider_encoded", "providerId", "providerID"])
    if provider_encoded is None:
        provider_encoded = _stable_hash(provider_id)

    duration_values = [float(a.get("duration_minutes", 0)) for a in appts if a.get("duration_minutes") is not None]
    avg_appt_duration = round(sum(duration_values) / max(1, len(duration_values)), 2) if duration_values else int(provider.get("avg_duration_minutes", provider.get("provider_avg_duration", 30)))

    first = provider.get("firstName", "")
    last = provider.get("lastName", "")
    name = f"{first} {last}".strip() or f"Provider {provider_id[:8]}"

    raw_days = provider.get("working_days", [0, 1, 2, 3, 4])
    if isinstance(raw_days, str):
        working_days = [int(d) for d in raw_days.split(",") if d.strip().isdigit()]
    elif raw_days:
        working_days = [int(d) for d in raw_days]
    else:
        working_days = [0, 1, 2, 3, 4]

    # Hourly utilization curve: {hour: fraction_of_total_appts}
    from collections import Counter
    hour_counts = Counter(int(a["appt_hour"]) for a in appts if a.get("appt_hour") is not None)
    total_for_util = max(1, sum(hour_counts.values()))
    hourly_util_curve = {h: round(c / total_for_util, 4) for h, c in hour_counts.items()}

    # Peak hour density: hours above mean utilization
    mean_util = total_for_util / max(1, len(hour_counts)) if hour_counts else 0
    peak_hour_set = {h for h, c in hour_counts.items() if c > mean_util}

    return {
        "provider_id":                     provider_id,
        "provider_encoded":                provider_encoded,
        "provider_name":                   name,
        "speciality":                      provider.get("speciality", ""),
        "provider_total_appts":            total,
        "provider_avg_duration":           int(avg_appt_duration),
        "provider_cancel_rate":            round(cancellation_rate, 4),
        "provider_no_show_rate":           round(no_show_rate, 4),
        "provider_cancellation_rate":      round(cancellation_rate, 4),
        "provider_7day_util":              round(util_7d, 4),
        "provider_30day_util":             round(util_30d, 4),
        "provider_utilization":            round(util_7d, 4),
        "provider_utilization_rate":       round(util_7d, 4),
        "provider_avg_daily_appointments": round(avg_daily_appointments, 2),
        "provider_overbooking_ratio":      round(overbooking_ratio, 4),
        "provider_peak_hours":             peak_hours,
        "provider_peak_hour_set":          list(peak_hour_set),
        "provider_hourly_util_curve":      hourly_util_curve,
        "provider_hist_success_rate":      round(success_rate, 4),
        "provider_hist_appt_count":        total,
        "provider_slot_volume":            max(1, total),
        "working_days":                    working_days,
        "hours": {
            "start": int(provider.get("work_start_hour", provider.get("hours", {}).get("start", 8))),
            "end":   int(provider.get("work_end_hour",   provider.get("hours", {}).get("end",   17))),
        },
    }


def _default_provider(provider_id: str) -> Dict[str, Any]:
    return {
        "provider_id":                provider_id,
        "provider_encoded":           _stable_hash(provider_id),
        "provider_name":              f"Provider {provider_id[:8]}",
        "speciality":                 "",
        "provider_total_appts":       1,
        "provider_avg_duration":      30,
        "provider_cancel_rate":       0.0,
        "provider_no_show_rate":      0.0,
        "provider_cancellation_rate": 0.0,
        "provider_7day_util":         0.5,
        "provider_30day_util":        0.5,
        "provider_utilization":       0.5,
        "provider_utilization_rate":  0.5,
        "provider_avg_daily_appointments": 1.0,
        "provider_overbooking_ratio":  0.0,
        "provider_peak_hours":         [],
        "provider_peak_hour_set":      [],
        "provider_hourly_util_curve":  {},
        "provider_hist_success_rate":  0.5,
        "provider_hist_appt_count":   0,
        "provider_slot_volume":       1,
        "working_days":               [0, 1, 2, 3, 4],
        "hours":                      {"start": 8, "end": 17},
    }


def get_provider_schedule(db: Database, provider_id: str) -> List[str]:
    col  = _cols(db)
    docs = db[col["provider_schedules"]].find({"provider_id": provider_id})
    return [d["blocked_date"] for d in docs if "blocked_date" in d]


def get_booked_slots(db: Database, provider_id: str, date_iso: str) -> List[int]:
    col  = _cols(db)
    docs = db[col["appointments"]].find({
        "provider_id": provider_id,
        "appt_date":   date_iso,
        "status":      {"$in": ["Confirmed", "Confirmed           ", "Confirmation Pending"]},
    })
    return [int(d["appt_hour"]) for d in docs if "appt_hour" in d]


def ensure_database_indexes(db: Database) -> None:
    from src.config.read_only_config import is_write_enabled, log_write_blocked
    
    # WRITE GUARD: Skip index creation in read-only mode
    if not is_write_enabled():
        log_write_blocked("ensure_database_indexes", "create_index operations")
        logger.info("Index creation skipped (read-only mode)")
        return
    
    col = _cols(db)
    db[col["patients"]].create_index([("fullName", 1)], name="idx_patient_fullName", unique=False)
    db[col["patients"]].create_index([("firstName", 1), ("lastName", 1)], name="idx_patient_name", unique=False)
    db[col["providers"]].create_index([("firstName", 1), ("lastName", 1), ("role", 1)], name="idx_provider_name_role", unique=False)
    db[col["appointments"]].create_index([("provider_id", 1), ("patient_id", 1), ("appt_date", 1), ("appt_hour", 1)], name="idx_appointments_provider_patient_date", unique=False)
    db[col["slot_statistics"]].create_index([("provider_id", 1), ("weekday", 1), ("hour", 1)], name="idx_slot_statistics_provider_weekday_hour", unique=True)


def get_patient_by_name(db: Database, patient_name: str) -> str:
    if not isinstance(patient_name, str) or not patient_name.strip():
        raise ValueError("patient_name must be a non-empty string")

    col = _cols(db)
    collection = db[col["patients"]]
    normalized = _normalize_name(patient_name)
    tokens = [t for t in normalized.split() if t]

    logger.info("[patient_lookup] searching patient_name='%s' normalized='%s' tokens=%s",
                patient_name, normalized, tokens)

    # Exact match on fullName or nested data.fullName
    queries = [
        {"fullName": {"$regex": f"^{re.escape(normalized)}$", "$options": "i"}},
        {"data.fullName": {"$regex": f"^{re.escape(normalized)}$", "$options": "i"}},
    ]
    for query in queries:
        doc = collection.find_one(query)
        if doc:
            return str(doc["_id"])

    # Try name-token intersection and first/last name fields
    if len(tokens) >= 2:
        token_query = {"$and": [{"fullName": {"$regex": re.escape(token), "$options": "i"}} for token in tokens]}
        doc = collection.find_one(token_query)
        if doc:
            return str(doc["_id"])

        doc = collection.find_one({"firstName": {"$regex": re.escape(tokens[0]), "$options": "i"},
                                   "lastName": {"$regex": re.escape(tokens[-1]), "$options": "i"}})
        if doc:
            return str(doc["_id"])

    # Fall back to fuzzy search across candidate name fields
    if tokens:
        candidates = list(collection.find(_query_candidates_by_tokens(tokens, _patient_search_fields())))
        doc = _best_name_match(candidates, patient_name, _patient_search_fields())
        if doc:
            return str(doc["_id"])

    raise EntityNotFoundError("patient", patient_name)


def get_provider_by_name(db: Database, provider_name: str) -> str:
    if not isinstance(provider_name, str) or not provider_name.strip():
        raise ValueError("provider_name must be a non-empty string")

    cleaned = re.sub(r"^(dr\.?\s*|doctor\s*)", "", provider_name.strip(), flags=re.IGNORECASE)
    normalized = _normalize_name(cleaned)
    tokens = [t for t in normalized.split() if t]
    col = _cols(db)
    collection = db[col["providers"]]

    logger.info("[provider_lookup] searching provider_name='%s' cleaned='%s' tokens=%s",
                provider_name, cleaned, tokens)

    if len(tokens) >= 2:
        query = {
            "firstName": {"$regex": f"^{re.escape(tokens[0])}$", "$options": "i"},
            "lastName": {"$regex": re.escape(tokens[-1]), "$options": "i"},
        }
        doc = collection.find_one(query)
        if doc:
            return str(doc["_id"])

    if len(tokens) >= 2 and len(tokens[0]) == 1:
        query = {
            "lastName": {"$regex": f"^{re.escape(tokens[-1])}$", "$options": "i"},
            "firstName": {"$regex": f"^{re.escape(tokens[0])}", "$options": "i"},
        }
        doc = collection.find_one(query)
        if doc:
            return str(doc["_id"])

    if tokens:
        candidates = list(collection.find(_query_candidates_by_tokens(tokens, _provider_search_fields())))
        doc = _best_name_match(candidates, cleaned, _provider_search_fields())
        if doc:
            return str(doc["_id"])

    raise EntityNotFoundError("provider", provider_name)


def refresh_slot_statistics(db: Database) -> None:
    """Recompute per-(provider, weekday, hour) success rates from appointments."""
    from src.config.read_only_config import is_write_enabled, log_write_blocked
    
    col = _cols(db)
    # Fixed pipeline: was missing '$' on $group stage
    pipeline = [
        {"$match": {"appt_date": {"$ne": None}, "appt_hour": {"$ne": None}}},
        {
            "$project": {
                "provider_id": 1,
                "appt_hour": 1,
                "status": 1,
                "weekday": {
                    "$subtract": [
                        {"$dayOfWeek": {"$dateFromString": {"dateString": "$appt_date", "onError": datetime.utcnow()}}},
                        1,
                    ]
                },
            }
        },
        {
            "$group": {
                "_id": {"provider_id": "$provider_id", "weekday": "$weekday", "hour": "$appt_hour"},
                "total_count": {"$sum": 1},
                "success_count": {
                    "$sum": {"$cond": [{"$in": ["$status", ["Confirmed", "Confirmed           "]]}, 1, 0]}
                },
            }
        },
    ]
    try:
        stats = list(db[col["appointments"]].aggregate(pipeline))
    except Exception as exc:
        logger.error("refresh_slot_statistics failed: %s", exc)
        return

    for stat in stats:
        _id = stat["_id"]
        provider_id = _id.get("provider_id")
        weekday = _id.get("weekday")
        hour = _id.get("hour")
        if provider_id is None or weekday is None or hour is None:
            continue
        total_count = int(stat["total_count"])
        success_count = int(stat["success_count"])
        success_rate = success_count / max(1, total_count)
        # Popularity: base 0.2 + success signal + volume signal (capped)
        popularity_score = min(1.0, 0.2 + 0.5 * success_rate + min(total_count / 30.0, 0.3))
        
        # WRITE GUARD: Skip DB update in read-only mode
        if not is_write_enabled():
            log_write_blocked("refresh_slot_statistics", "update_one on slot_statistics")
            continue
        
        db[col["slot_statistics"]].update_one(
            {"provider_id": provider_id, "weekday": weekday, "hour": hour},
            {"$set": {
                "total_count": total_count,
                "success_count": success_count,
                "success_rate": round(success_rate, 4),
                "popularity_score": round(popularity_score, 4),
                "updated_at": datetime.utcnow(),
            }},
            upsert=True,
        )


def get_slot_statistics(db: Database, provider_id: str, weekday: int, hour: int) -> Dict[str, Any]:
    from src.config.read_only_config import is_write_enabled
    
    col  = _cols(db)
    
    # READ-ONLY OPTIMIZATION: Skip DB lookup and use dynamic computation
    if not is_write_enabled():
        return _compute_dynamic_slot_statistics(db, col, provider_id, weekday, hour)
    
    stat = db[col["slot_statistics"]].find_one({
        "provider_id": provider_id,
        "weekday":     weekday,
        "hour":        hour,
    })
    if stat:
        return {
            "success_rate":     float(stat.get("success_rate",     0.5)),
            "popularity_score": float(stat.get("popularity_score", 0.0)),
            "total_count":      int(stat.get("total_count",        0)),
        }

    # Fallback: compute on-the-fly from appointments (correct weekday filter)
    return _compute_dynamic_slot_statistics(db, col, provider_id, weekday, hour)


def _compute_dynamic_slot_statistics(
    db: Database, 
    col: dict, 
    provider_id: str, 
    weekday: int, 
    hour: int
) -> Dict[str, Any]:
    """
    Enhanced fallback with high-variance per-slot features.
    Computes statistics dynamically from appointment history.
    Used when slot_statistics collection is unavailable or in read-only mode.
    """
    appts = list(db[col["appointments"]].find({
        "provider_id": provider_id,
        "appt_date":   {"$ne": None},
        "appt_hour":   hour,
    }))
    slot_appts = [a for a in appts if _parse_date(a.get("appt_date")).weekday() == weekday]
    total = len(slot_appts)
    confirmed = [a for a in slot_appts if str(a.get("status", "")).strip() in CONFIRMED_STATUSES]
    
    # Base success rate from actual data
    success_rate = len(confirmed) / max(1, total) if total > 0 else 0.5
    
    # Enhanced popularity: hour demand * weekday demand * volume signal
    # Import demand weights from feature builder for consistency
    from src.features.slot_feature_builder import _HOUR_DEMAND_WEIGHTS, _WEEKDAY_DEMAND_WEIGHTS
    hour_weight = _HOUR_DEMAND_WEIGHTS.get(hour, 0.5)
    weekday_weight = _WEEKDAY_DEMAND_WEIGHTS.get(weekday, 0.5)
    volume_signal = min(0.3, total / 30.0)
    popularity_score = min(1.0, 0.2 + 0.5 * success_rate + volume_signal + 0.1 * hour_weight * weekday_weight)
    
    return {
        "success_rate":     round(success_rate, 4),
        "popularity_score": round(popularity_score, 4),
        "total_count":      total,
    }


def get_provider_hourly_utilization(db: Database, provider_id: str) -> Dict[int, float]:
    """Return {hour: utilization_rate} across all history for a provider."""
    col = _cols(db)
    appts = list(db[col["appointments"]].find(
        {"provider_id": provider_id, "appt_hour": {"$ne": None}},
        {"appt_hour": 1, "_id": 0},
    ))
    if not appts:
        return {}
    from collections import Counter
    counts = Counter(int(a["appt_hour"]) for a in appts if a.get("appt_hour") is not None)
    total = max(1, sum(counts.values()))
    return {hour: round(count / total, 4) for hour, count in counts.items()}


# ─── Appointments ─────────────────────────────────────────────────────────────

def insert_appointment(
    db: Database,
    patient_id: str,
    provider_id: str,
    appt_date: str,
    appt_hour: int,
    duration_minutes: int = 60,
    visit_reason: str = "",
    is_telehealth: bool = False,
    is_new_patient: bool = False,
) -> Dict[str, Any]:
    from src.config.read_only_config import is_write_enabled, log_write_blocked
    
    col = _cols(db)
    doc = {
        "patient_id":       patient_id,
        "provider_id":      provider_id,
        "appt_date":        appt_date,
        "appt_hour":        appt_hour,
        "duration_minutes": duration_minutes,
        "visit_reason":     visit_reason,
        "is_telehealth":    is_telehealth,
        "is_new_patient":   is_new_patient,
        "status":           "Confirmation Pending",
        "created_at":       datetime.utcnow(),
        "updated_at":       datetime.utcnow(),
    }
    
    # WRITE GUARD: Skip DB insert in read-only mode, return mock appointment
    if not is_write_enabled():
        log_write_blocked("insert_appointment", "insert_one on appointments")
        doc["_id"] = "READ_ONLY_MOCK_ID"
        logger.info(
            "Appointment booking simulated (read-only): patient=%s provider=%s date=%s hour=%s",
            patient_id, provider_id, appt_date, appt_hour
        )
        return doc
    
    result     = db[col["appointments"]].insert_one(doc)
    doc["_id"] = str(result.inserted_id)
    logger.info("Appointment booked: id=%s patient=%s provider=%s date=%s hour=%s",
                doc["_id"], patient_id, provider_id, appt_date, appt_hour)
    return doc


def update_appointment_status(db: Database, appointment_id: str, status: str) -> Optional[Dict[str, Any]]:
    from src.config.read_only_config import is_write_enabled, log_write_blocked
    
    col = _cols(db)
    oid = _to_oid(appointment_id)
    if not oid:
        logger.warning("Invalid appointment_id: %s", appointment_id)
        return None
    
    # WRITE GUARD: Skip DB update in read-only mode, return current state
    if not is_write_enabled():
        log_write_blocked("update_appointment_status", "find_one_and_update on appointments")
        result = db[col["appointments"]].find_one({"_id": oid})
        logger.info(
            "Status update skipped (read-only): appt=%s new_status=%s",
            appointment_id, status
        )
        return result
    
    result = db[col["appointments"]].find_one_and_update(
        {"_id": oid},
        {"$set": {"status": status, "updated_at": datetime.utcnow()}},
        return_document=True,
    )
    if not result:
        logger.warning("Appointment not found: %s", appointment_id)
    return result


def get_patient_appointments(db: Database, patient_id: str) -> List[Dict[str, Any]]:
    col  = _cols(db)
    docs = db[col["appointments"]].find({"patient_id": patient_id}, sort=[("appt_date", -1)])
    return [
        {
            "id":               str(d.get("_id", "")),
            "appt_date":        d.get("appt_date", ""),
            "appt_hour":        d.get("appt_hour", 0),
            "duration_minutes": d.get("duration_minutes", 60),
            "status":           str(d.get("status", "")).strip(),
            "visit_reason":     d.get("visit_reason", ""),
            "is_telehealth":    d.get("is_telehealth", False),
        }
        for d in docs
    ]


def get_provider_appointments(db: Database, provider_id: str, date_iso: Optional[str] = None) -> List[Dict[str, Any]]:
    col   = _cols(db)
    query: Dict[str, Any] = {"provider_id": provider_id}
    if date_iso:
        query["appt_date"] = date_iso
    docs = db[col["appointments"]].find(query, sort=[("appt_date", 1), ("appt_hour", 1)])
    return [
        {
            "id":               str(d.get("_id", "")),
            "patient_id":       d.get("patient_id"),
            "appt_date":        d.get("appt_date", ""),
            "appt_hour":        d.get("appt_hour", 0),
            "status":           str(d.get("status", "")).strip(),
            "duration_minutes": d.get("duration_minutes", 60),
        }
        for d in docs
    ]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _parse_date(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return datetime.min


def _extract_numeric_id(doc: dict, fields: list) -> Optional[int]:
    """Try field names at top-level then nested under 'data'."""
    for field in fields:
        val = doc.get(field)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                pass
    nested = doc.get("data", {})
    if isinstance(nested, dict):
        for field in fields:
            val = nested.get(field)
            if val is not None:
                try:
                    return int(val)
                except (ValueError, TypeError):
                    pass
    return None


def _stable_hash(id_str: str) -> int:
    """Deterministic small positive int from _id string — ML feature fallback."""
    return abs(hash(id_str)) % 100_000


def _active_days(appts: List[Dict[str, Any]]) -> int:
    dates = {_parse_date(a.get("appt_date")).date() for a in appts if a.get("appt_date")}
    return max(1, len(dates))


def _build_time_bucket_profile(appts: List[Dict[str, Any]]) -> Dict[str, int]:
    buckets = {"patient_pref_morning": 0, "patient_pref_midday": 0, "patient_pref_afternoon": 0, "patient_pref_evening": 0}
    for appt in appts:
        hour = appt.get("appt_hour")
        if hour is None:
            continue
        try:
            hour = int(hour)
        except (ValueError, TypeError):
            continue
        if 6 <= hour < 11:
            buckets["patient_pref_morning"] += 1
        elif 11 <= hour < 14:
            buckets["patient_pref_midday"] += 1
        elif 14 <= hour < 18:
            buckets["patient_pref_afternoon"] += 1
        elif 18 <= hour < 22:
            buckets["patient_pref_evening"] += 1
    if sum(buckets.values()) == 0:
        buckets["patient_pref_morning"] = 1
    return buckets


def _compute_avg_lead_time(appts: List[Dict[str, Any]]) -> float:
    lead_times = []
    for appt in appts:
        appt_date = _parse_date(appt.get("appt_date"))
        booked_date = _parse_date(appt.get("created_at") or appt.get("booking_date") or appt.get("scheduled_at"))
        if appt_date and booked_date and appt_date > booked_date:
            lead_times.append((appt_date - booked_date).days)
    if not lead_times:
        return 7.0
    return float(sum(lead_times)) / max(1, len(lead_times))


def _compute_reliability_score(cancel_rate: float, no_show_rate: float) -> float:
    score = 1.0 - min(1.0, cancel_rate * 0.7 + no_show_rate * 0.9)
    return max(0.0, min(1.0, score))


def _top_time_buckets(hours: List[Any]) -> List[int]:
    buckets = {"morning": 0, "midday": 0, "afternoon": 0, "evening": 0}
    for hour in hours:
        if hour is None:
            continue
        try:
            hour = int(hour)
        except (ValueError, TypeError):
            continue
        if 6 <= hour < 11:
            buckets["morning"] += 1
        elif 11 <= hour < 14:
            buckets["midday"] += 1
        elif 14 <= hour < 18:
            buckets["afternoon"] += 1
        elif 18 <= hour < 22:
            buckets["evening"] += 1
    sorted_buckets = sorted(buckets.items(), key=lambda item: item[1], reverse=True)
    return [name for name, _ in sorted_buckets if _ > 0][:2]


def _normalize_name(value: str) -> str:
    return " ".join(re.sub(r"[^A-Z0-9 ]", " ", value.upper()).split())


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _best_name_match(candidates: List[dict], query: str, keys: List[str], min_ratio: float = 0.55) -> Optional[dict]:
    if not candidates:
        return None

    query_norm = _normalize_name(query)
    best: Optional[dict] = None
    best_score = 0.0

    for doc in candidates:
        values = []
        for key in keys:
            if key in doc and isinstance(doc[key], str):
                values.append(_normalize_name(doc[key]))
        if "data" in doc and isinstance(doc["data"], dict):
            for key in keys:
                if key in doc["data"] and isinstance(doc["data"][key], str):
                    values.append(_normalize_name(doc["data"][key]))

        for value in values:
            score = _similarity(query_norm, value)
            if score > best_score:
                best_score = score
                best = doc

    return best if best_score >= min_ratio else None


def _query_candidates_by_tokens(tokens: List[str], fields: List[str]) -> dict:
    return {
        "$or": [
            {field: {"$regex": re.escape(token), "$options": "i"}}
            for token in tokens
            for field in fields
        ]
    }


def _extract_document_name(doc: dict) -> str:
    if not isinstance(doc, dict):
        return ""
    parts = []
    if doc.get("fullName"):
        parts.append(str(doc["fullName"]).strip())
    if doc.get("firstName") or doc.get("lastName"):
        parts.append(" ".join(filter(None, [str(doc.get("firstName", "")).strip(), str(doc.get("lastName", "")).strip()])))
    if isinstance(doc.get("data"), dict) and doc["data"].get("fullName"):
        parts.append(str(doc["data"]["fullName"]).strip())
    return " ".join(parts).strip()


def _patient_search_fields() -> List[str]:
    return ["fullName", "firstName", "lastName"]


def _provider_search_fields() -> List[str]:
    return ["firstName", "lastName", "fullName"]
