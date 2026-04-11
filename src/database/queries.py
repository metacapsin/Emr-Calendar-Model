from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from bson import ObjectId
from pymongo.database import Database

from src.database.db_connection import get_collection_names
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

def get_patient_data(db: Database, patient_id: str) -> Dict[str, Any]:
    """Fetch patient by MongoDB _id. Returns ML-compatible feature dict."""
    col = _cols(db)
    oid = _to_oid(patient_id)
    patient = db[col["patients"]].find_one({"_id": oid}) if oid else None

    if not patient:
        logger.warning("[get_patient_data] not found: id=%s — using defaults", patient_id)
        return _default_patient(patient_id)

    now       = datetime.utcnow()
    window_7  = now - timedelta(days=7)
    window_30 = now - timedelta(days=30)

    appts      = list(db[col["appointments"]].find({"patient_id": patient_id}))
    total      = len(appts)
    confirmed  = [a for a in appts if str(a.get("status", "")).strip() in CONFIRMED_STATUSES]
    appts_7d   = [a for a in appts if _parse_date(a.get("appt_date")) >= window_7]
    appts_30d  = [a for a in appts if _parse_date(a.get("appt_date")) >= window_30]
    cancel_7d  = [a for a in appts_7d  if str(a.get("status", "")).strip() not in CONFIRMED_STATUSES]
    cancel_30d = [a for a in appts_30d if str(a.get("status", "")).strip() not in CONFIRMED_STATUSES]
    success_rate = len(confirmed) / max(1, total)

    # Numeric ML ID — fallback to stable hash of _id string
    patient_encoded = _extract_numeric_id(patient, ["patient_encoded", "patientId", "patientID"])
    if patient_encoded is None:
        patient_encoded = _stable_hash(patient_id)

    # Real EMR patient-details: sex = "F"/"M"
    sex_raw     = patient.get("sex", patient.get("gender", ""))
    sex_encoded = 0 if str(sex_raw).strip().upper() in ("F", "FEMALE") else 1

    # Real EMR: insurance nested object
    insurance   = patient.get("insurance", {}) or {}
    ins_type    = str(insurance.get("insuranceType", "")).lower()
    has_ins     = str(patient.get("hasInsurance", "no")).lower() == "yes"
    has_sec     = str(patient.get("hasSecondaryInsurance", "no")).lower() == "yes"
    is_medicare = "medicare" in ins_type
    is_medicaid = "medicaid" in ins_type
    is_hmo      = "hmo" in ins_type
    copay_raw   = insurance.get("coPayAmount", "")
    try:
        avg_copay = float(copay_raw) if copay_raw else 0.0
    except (ValueError, TypeError):
        avg_copay = 0.0

    # Real EMR: dOB = "MM/DD/YYYY"
    age = 35
    dob = patient.get("dOB", "")
    if dob:
        try:
            age = (date.today() - datetime.strptime(dob, "%m/%d/%Y").date()).days // 365
        except (ValueError, TypeError):
            age = 35

    return {
        "patient_id":                patient_id,
        "patient_encoded":           patient_encoded,
        "patient_age":               age,
        "sex_encoded":               sex_encoded,
        "patient_total_appts":       total,
        "patient_cancel_rate":       round(1 - success_rate, 4),
        "patient_7day_appts":        len(appts_7d),
        "patient_30day_appts":       len(appts_30d),
        "patient_7day_cancel":       len(cancel_7d),
        "patient_30day_cancel":      len(cancel_30d),
        "patient_hist_success_rate": round(success_rate, 4),
        "patient_hist_appt_count":   total,
        "has_primary_insurance":     int(has_ins),
        "has_secondary_insurance":   int(has_sec),
        "is_medicare":               int(is_medicare),
        "is_medicaid":               int(is_medicaid),
        "is_hmo":                    int(is_hmo),
        "patient_avg_copay":         avg_copay,
    }


def _default_patient(patient_id: str) -> Dict[str, Any]:
    return {
        "patient_id":                patient_id,
        "patient_encoded":           _stable_hash(patient_id),
        "patient_age":               35,
        "sex_encoded":               1,
        "patient_total_appts":       1,
        "patient_cancel_rate":       0.0,
        "patient_7day_appts":        0,
        "patient_30day_appts":       0,
        "patient_7day_cancel":       0,
        "patient_30day_cancel":      0,
        "patient_hist_success_rate": 0.5,
        "patient_hist_appt_count":   0,
        "has_primary_insurance":     1,
        "has_secondary_insurance":   0,
        "is_medicare":               0,
        "is_medicaid":               0,
        "is_hmo":                    0,
        "patient_avg_copay":         0.0,
    }


# ─── Provider ─────────────────────────────────────────────────────────────────

def get_provider_data(db: Database, provider_id: str) -> Dict[str, Any]:
    """Fetch provider by MongoDB _id. Returns ML-compatible feature dict."""
    col = _cols(db)
    oid = _to_oid(provider_id)
    provider = db[col["providers"]].find_one({"_id": oid}) if oid else None

    if not provider:
        logger.warning("[get_provider_data] not found: id=%s — using defaults", provider_id)
        return _default_provider(provider_id)

    now       = datetime.utcnow()
    window_7  = now - timedelta(days=7)
    window_30 = now - timedelta(days=30)

    appts        = list(db[col["appointments"]].find({"provider_id": provider_id}))
    total        = len(appts)
    confirmed    = [a for a in appts if str(a.get("status", "")).strip() in CONFIRMED_STATUSES]
    appts_7d     = [a for a in appts if _parse_date(a.get("appt_date")) >= window_7]
    appts_30d    = [a for a in appts if _parse_date(a.get("appt_date")) >= window_30]
    max_daily    = int(provider.get("max_daily_slots", 16))
    util_7d      = len(appts_7d)  / max(1, max_daily * 7)
    util_30d     = len(appts_30d) / max(1, max_daily * 30)
    success_rate = len(confirmed) / max(1, total)

    provider_encoded = _extract_numeric_id(provider, ["provider_encoded", "providerId", "providerID"])
    if provider_encoded is None:
        provider_encoded = _stable_hash(provider_id)

    first = provider.get("firstName", "")
    last  = provider.get("lastName", "")
    name  = f"{first} {last}".strip() or f"Provider {provider_id[:8]}"

    # Real EMR schema has no working_days/hours fields — use safe defaults
    raw_days = provider.get("working_days", [0, 1, 2, 3, 4])
    if isinstance(raw_days, str):
        working_days = [int(d) for d in raw_days.split(",") if d.strip().isdigit()]
    elif raw_days:
        working_days = [int(d) for d in raw_days]
    else:
        working_days = [0, 1, 2, 3, 4]

    return {
        "provider_id":                provider_id,
        "provider_encoded":           provider_encoded,
        "provider_name":              name,
        "speciality":                 provider.get("speciality", ""),
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


def _default_provider(provider_id: str) -> Dict[str, Any]:
    return {
        "provider_id":                provider_id,
        "provider_encoded":           _stable_hash(provider_id),
        "provider_name":              f"Provider {provider_id[:8]}",
        "speciality":                 "",
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


# ─── Slot Statistics ──────────────────────────────────────────────────────────

def get_slot_statistics(db: Database, provider_id: str, weekday: int, hour: int) -> Dict[str, Any]:
    col  = _cols(db)
    stat = db[col["slot_statistics"]].find_one({
        "provider_id": provider_id,
        "weekday":     weekday,
        "hour":        hour,
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
    patient_id: str,
    provider_id: str,
    appt_date: str,
    appt_hour: int,
    duration_minutes: int = 60,
    visit_reason: str = "",
    is_telehealth: bool = False,
    is_new_patient: bool = False,
) -> Dict[str, Any]:
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
    result     = db[col["appointments"]].insert_one(doc)
    doc["_id"] = str(result.inserted_id)
    logger.info("Appointment booked: id=%s patient=%s provider=%s date=%s hour=%s",
                doc["_id"], patient_id, provider_id, appt_date, appt_hour)
    return doc


def update_appointment_status(db: Database, appointment_id: str, status: str) -> Optional[Dict[str, Any]]:
    col = _cols(db)
    oid = _to_oid(appointment_id)
    if not oid:
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
