from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pymongo.database import Database

from src.database.db_connection import get_collection_names
from src.utils.logger import get_logger

logger = get_logger(__name__)

CONFIRMED_STATUSES = {"Confirmed", "Confirmed           "}


def _cols(db: Database) -> dict:
    return get_collection_names()


# ─── Patient ──────────────────────────────────────────────────────────────────

def get_patient_data(db: Database, patient_encoded: int) -> Dict[str, Any]:
    col = _cols(db)
    patient = db[col["patients"]].find_one({"patient_encoded": patient_encoded})

    if not patient:
        logger.warning("Patient not found: %s — using defaults", patient_encoded)
        return _default_patient(patient_encoded)

    now = datetime.utcnow()
    window_7  = now - timedelta(days=7)
    window_30 = now - timedelta(days=30)

    appts = list(db[col["appointments"]].find({"patient_encoded": patient_encoded}))
    total     = len(appts)
    confirmed = [a for a in appts if str(a.get("status", "")).strip() in CONFIRMED_STATUSES]

    appts_7d   = [a for a in appts if _parse_date(a.get("appt_date")) >= window_7]
    appts_30d  = [a for a in appts if _parse_date(a.get("appt_date")) >= window_30]
    cancel_7d  = [a for a in appts_7d  if str(a.get("status", "")).strip() not in CONFIRMED_STATUSES]
    cancel_30d = [a for a in appts_30d if str(a.get("status", "")).strip() not in CONFIRMED_STATUSES]

    success_rate = len(confirmed) / max(1, total)

    return {
        "patient_encoded":            patient_encoded,
        "patient_age":                patient.get("age", patient.get("patient_age", 35)),
        "sex_encoded":                patient.get("sex_encoded", 1),
        "patient_total_appts":        total,
        "patient_cancel_rate":        round(1 - success_rate, 4),
        "patient_7day_appts":         len(appts_7d),
        "patient_30day_appts":        len(appts_30d),
        "patient_7day_cancel":        len(cancel_7d),
        "patient_30day_cancel":       len(cancel_30d),
        "patient_hist_success_rate":  round(success_rate, 4),
        "patient_hist_appt_count":    total,
        "has_primary_insurance":      int(patient.get("has_primary_insurance", True)),
        "has_secondary_insurance":    int(patient.get("has_secondary_insurance", False)),
        "is_medicare":                int(patient.get("is_medicare", False)),
        "is_medicaid":                int(patient.get("is_medicaid", False)),
        "is_hmo":                     int(patient.get("is_hmo", False)),
        "patient_avg_copay":          float(patient.get("patient_avg_copay", 0.0)),
    }


def _default_patient(patient_encoded: int) -> Dict[str, Any]:
    return {
        "patient_encoded":            patient_encoded,
        "patient_age":                35,
        "sex_encoded":                1,
        "patient_total_appts":        1,
        "patient_cancel_rate":        0.0,
        "patient_7day_appts":         0,
        "patient_30day_appts":        0,
        "patient_7day_cancel":        0,
        "patient_30day_cancel":       0,
        "patient_hist_success_rate":  0.5,
        "patient_hist_appt_count":    0,
        "has_primary_insurance":      1,
        "has_secondary_insurance":    0,
        "is_medicare":                0,
        "is_medicaid":                0,
        "is_hmo":                     0,
        "patient_avg_copay":          0.0,
    }


# ─── Provider ─────────────────────────────────────────────────────────────────

def get_provider_data(db: Database, provider_encoded: int) -> Dict[str, Any]:
    col = _cols(db)
    provider = db[col["providers"]].find_one({"provider_encoded": provider_encoded})

    if not provider:
        logger.warning("Provider not found: %s — using defaults", provider_encoded)
        return _default_provider(provider_encoded)

    now = datetime.utcnow()
    window_7  = now - timedelta(days=7)
    window_30 = now - timedelta(days=30)

    appts = list(db[col["appointments"]].find({"provider_encoded": provider_encoded}))
    total     = len(appts)
    confirmed = [a for a in appts if str(a.get("status", "")).strip() in CONFIRMED_STATUSES]
    appts_7d  = [a for a in appts if _parse_date(a.get("appt_date")) >= window_7]
    appts_30d = [a for a in appts if _parse_date(a.get("appt_date")) >= window_30]

    max_daily = int(provider.get("max_daily_slots", 16))
    util_7d   = len(appts_7d)  / max(1, max_daily * 7)
    util_30d  = len(appts_30d) / max(1, max_daily * 30)
    success_rate = len(confirmed) / max(1, total)

    # working_days stored as list [0,1,2,3,4] or CSV string "0,1,2,3,4"
    raw_days = provider.get("working_days", [0, 1, 2, 3, 4])
    if isinstance(raw_days, str):
        working_days = [int(d) for d in raw_days.split(",") if d.strip().isdigit()]
    else:
        working_days = [int(d) for d in raw_days]

    return {
        "provider_encoded":           provider_encoded,
        "provider_name":              provider.get("name", f"Dr {provider_encoded}"),
        "provider_total_appts":       total,
        "provider_avg_duration":      int(provider.get("avg_duration_minutes", provider.get("provider_avg_duration", 30))),
        "provider_cancel_rate":       round(1 - success_rate, 4),
        "provider_7day_util":         round(util_7d, 4),
        "provider_30day_util":        round(util_30d, 4),
        "provider_hist_success_rate": round(success_rate, 4),
        "provider_hist_appt_count":   total,
        "working_days":               working_days,
        "hours": {
            "start": int(provider.get("work_start_hour", provider.get("hours", {}).get("start", 8))),
            "end":   int(provider.get("work_end_hour",   provider.get("hours", {}).get("end",   17))),
        },
    }


def _default_provider(provider_encoded: int) -> Dict[str, Any]:
    return {
        "provider_encoded":           provider_encoded,
        "provider_name":              f"Dr {provider_encoded}",
        "provider_total_appts":       1,
        "provider_avg_duration":      30,
        "provider_cancel_rate":       0.0,
        "provider_7day_util":         0.5,
        "provider_30day_util":        0.5,
        "provider_hist_success_rate": 0.5,
        "provider_hist_appt_count":   0,
        "working_days":               [0, 1, 2, 3, 4],
        "hours":                      {"start": 8, "end": 17},
    }


def get_provider_schedule(db: Database, provider_encoded: int) -> List[str]:
    """Return list of blocked ISO date strings for a provider."""
    col = _cols(db)
    docs = db[col["provider_schedules"]].find({"provider_encoded": provider_encoded})
    return [d["blocked_date"] for d in docs if "blocked_date" in d]


def get_booked_slots(db: Database, provider_encoded: int, date_iso: str) -> List[int]:
    """Return list of already-booked hours for a provider on a given date."""
    col = _cols(db)
    docs = db[col["appointments"]].find({
        "provider_encoded": provider_encoded,
        "appt_date":        date_iso,
        "status":           {"$in": ["Confirmed", "Confirmed           ", "Confirmation Pending"]},
    })
    return [int(d["appt_hour"]) for d in docs if "appt_hour" in d]


# ─── Slot Statistics ──────────────────────────────────────────────────────────

def get_slot_statistics(db: Database, provider_encoded: int, weekday: int, hour: int) -> Dict[str, Any]:
    col = _cols(db)
    stat = db[col["slot_statistics"]].find_one({
        "provider_encoded": provider_encoded,
        "weekday":          weekday,
        "hour":             hour,
    })
    if not stat:
        return {"success_rate": 0.5, "popularity_score": 0.0, "total_count": 0}
    return {
        "success_rate":     float(stat.get("success_rate",     0.5)),
        "popularity_score": float(stat.get("popularity_score", 0.0)),
        "total_count":      int(stat.get("total_count",        0)),
    }


# ─── Appointments ─────────────────────────────────────────────────────────────

def insert_appointment(
    db: Database,
    patient_encoded: int,
    provider_encoded: int,
    appt_date: str,
    appt_hour: int,
    duration_minutes: int = 60,
    visit_reason: str = "",
    is_telehealth: bool = False,
    is_new_patient: bool = False,
) -> Dict[str, Any]:
    col = _cols(db)
    doc = {
        "patient_encoded":  patient_encoded,
        "provider_encoded": provider_encoded,
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
    result = db[col["appointments"]].insert_one(doc)
    doc["_id"] = str(result.inserted_id)

    logger.info(
        "Appointment booked: id=%s patient=%s provider=%s date=%s hour=%s",
        doc["_id"], patient_encoded, provider_encoded, appt_date, appt_hour,
    )
    return doc


def update_appointment_status(db: Database, appointment_id: str, status: str) -> Optional[Dict[str, Any]]:
    from bson import ObjectId
    col = _cols(db)
    try:
        oid = ObjectId(appointment_id)
    except Exception:
        logger.warning("Invalid appointment_id: %s", appointment_id)
        return None

    result = db[col["appointments"]].find_one_and_update(
        {"_id": oid},
        {"$set": {"status": status, "updated_at": datetime.utcnow()}},
        return_document=True,
    )
    if not result:
        logger.warning("Appointment not found: %s", appointment_id)
    return result


def get_patient_appointments(db: Database, patient_encoded: int) -> List[Dict[str, Any]]:
    col = _cols(db)
    docs = db[col["appointments"]].find(
        {"patient_encoded": patient_encoded},
        sort=[("appt_date", -1)],
    )
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


def get_provider_appointments(db: Database, provider_encoded: int, date_iso: Optional[str] = None) -> List[Dict[str, Any]]:
    col = _cols(db)
    query: Dict[str, Any] = {"provider_encoded": provider_encoded}
    if date_iso:
        query["appt_date"] = date_iso

    docs = db[col["appointments"]].find(
        query,
        sort=[("appt_date", 1), ("appt_hour", 1)],
    )
    return [
        {
            "id":               str(d.get("_id", "")),
            "patient_encoded":  d.get("patient_encoded"),
            "appt_date":        d.get("appt_date", ""),
            "appt_hour":        d.get("appt_hour", 0),
            "status":           str(d.get("status", "")).strip(),
            "duration_minutes": d.get("duration_minutes", 60),
        }
        for d in docs
    ]


# ─── Helper ───────────────────────────────────────────────────────────────────

def _parse_date(value: Any) -> datetime:
    """Safely parse appt_date — handles str ISO, datetime, or None."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return datetime.min
