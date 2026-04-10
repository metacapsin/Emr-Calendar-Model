"""
MongoDB Document Schemas (reference only — no ORM needed with pymongo)

These are the expected field structures for each collection.
Your real MongoDB documents may have more fields — queries.py handles
flexible field access with .get() and safe defaults.
"""

# ── patients collection ───────────────────────────────────────────────────────
PATIENT_SCHEMA = {
    "patient_encoded":        int,    # unique identifier (used as lookup key)
    "age":                    int,    # or "patient_age"
    "sex_encoded":            int,    # 0=F, 1=M
    "has_primary_insurance":  bool,
    "has_secondary_insurance": bool,
    "is_medicare":            bool,
    "is_medicaid":            bool,
    "is_hmo":                 bool,
    "patient_avg_copay":      float,
    "created_at":             "datetime",
}

# ── providers collection ──────────────────────────────────────────────────────
PROVIDER_SCHEMA = {
    "provider_encoded":    int,       # unique identifier
    "name":                str,
    "specialty":           str,
    "avg_duration_minutes": int,      # or "provider_avg_duration"
    "working_days":        list,      # [0,1,2,3,4] or "0,1,2,3,4"
    "work_start_hour":     int,       # or nested hours.start
    "work_end_hour":       int,       # or nested hours.end
    "max_daily_slots":     int,
    "created_at":          "datetime",
}

# ── appointments collection ───────────────────────────────────────────────────
APPOINTMENT_SCHEMA = {
    "patient_encoded":  int,
    "provider_encoded": int,
    "appt_date":        str,          # ISO date "YYYY-MM-DD"
    "appt_hour":        int,
    "duration_minutes": int,
    "status":           str,          # "Confirmed", "Confirmation Pending", "Rescheduled"
    "visit_reason":     str,
    "is_telehealth":    bool,
    "is_new_patient":   bool,
    "created_at":       "datetime",
    "updated_at":       "datetime",
}

# ── provider_schedules collection ─────────────────────────────────────────────
PROVIDER_SCHEDULE_SCHEMA = {
    "provider_encoded": int,
    "blocked_date":     str,          # ISO date "YYYY-MM-DD"
    "reason":           str,          # "holiday", "leave", etc.
}

# ── slot_statistics collection ────────────────────────────────────────────────
SLOT_STATISTICS_SCHEMA = {
    "provider_encoded": int,
    "weekday":          int,          # 0=Mon … 6=Sun
    "hour":             int,          # 0–23
    "total_count":      int,
    "success_count":    int,
    "success_rate":     float,
    "popularity_score": float,
    "updated_at":       "datetime",
}
