"""
Feedback Loop — Booking Outcome Tracker
========================================
Records predicted probability vs actual outcome for every booked slot.
This data feeds the retraining pipeline to close the learning loop.

Usage:
    from src.models.feedback import record_booking_outcome, get_feedback_stats

    # Call after booking is confirmed/cancelled/no-showed:
    record_booking_outcome(db, appointment_id, actual_status, predicted_prob)
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from pymongo.database import Database

from src.utils.logger import get_logger

logger = get_logger(__name__)

_FEEDBACK_COLLECTION = "slot_prediction_feedback"


def record_booking_outcome(
    db: Database,
    appointment_id: str,
    actual_status: str,
    predicted_prob: float,
    patient_id: Optional[str] = None,
    provider_id: Optional[str] = None,
    slot_hour: Optional[int] = None,
    slot_weekday: Optional[int] = None,
    appt_date: Optional[str] = None,
) -> None:
    """
    Record the outcome of a predicted slot booking.
    actual_status: 'Confirmed', 'Cancelled', 'No Show', etc.
    predicted_prob: the probability the model assigned at booking time.
    """
    from src.config.read_only_config import is_write_enabled, log_write_blocked
    
    confirmed_statuses = {"Confirmed", "Confirmed           "}
    actual_label = 1 if actual_status.strip() in confirmed_statuses else 0
    error = abs(predicted_prob - actual_label)

    doc = {
        "appointment_id":  appointment_id,
        "patient_id":      patient_id,
        "provider_id":     provider_id,
        "slot_hour":       slot_hour,
        "slot_weekday":    slot_weekday,
        "appt_date":       appt_date,
        "predicted_prob":  round(predicted_prob, 4),
        "actual_status":   actual_status.strip(),
        "actual_label":    actual_label,
        "prediction_error": round(error, 4),
        "recorded_at":     datetime.utcnow(),
    }

    try:
        # WRITE GUARD: Skip DB insert in read-only mode
        if not is_write_enabled():
            log_write_blocked("record_booking_outcome", "insert_one on slot_prediction_feedback")
            logger.info(
                "Feedback skipped (read-only): appt=%s predicted=%.3f actual=%d error=%.3f",
                appointment_id, predicted_prob, actual_label, error,
            )
            return
        
        db[_FEEDBACK_COLLECTION].insert_one(doc)
        logger.info(
            "Feedback recorded: appt=%s predicted=%.3f actual=%d error=%.3f",
            appointment_id, predicted_prob, actual_label, error,
        )
    except Exception as exc:
        logger.error("Failed to record feedback: %s", exc)


def get_feedback_stats(db: Database, days: int = 30) -> Dict[str, Any]:
    """
    Return calibration quality metrics from recent feedback.
    Use this to trigger retraining when drift is detected.
    """
    from datetime import timedelta
    cutoff = datetime.utcnow() - timedelta(days=days)

    docs = list(db[_FEEDBACK_COLLECTION].find(
        {"recorded_at": {"$gte": cutoff}},
        {"predicted_prob": 1, "actual_label": 1, "prediction_error": 1, "_id": 0},
    ))

    if not docs:
        return {"n": 0, "mean_error": None, "calibration_drift": False}

    n = len(docs)
    mean_error = sum(d["prediction_error"] for d in docs) / n
    mean_pred  = sum(d["predicted_prob"] for d in docs) / n
    mean_actual = sum(d["actual_label"] for d in docs) / n

    # Calibration drift: predicted mean vs actual mean diverges by >10%
    calibration_drift = abs(mean_pred - mean_actual) > 0.10

    stats = {
        "n":                n,
        "mean_error":       round(mean_error, 4),
        "mean_predicted":   round(mean_pred, 4),
        "mean_actual":      round(mean_actual, 4),
        "calibration_bias": round(mean_pred - mean_actual, 4),
        "calibration_drift": calibration_drift,
        "retrain_recommended": calibration_drift or mean_error > 0.35,
    }

    if stats["retrain_recommended"]:
        logger.warning(
            "RETRAINING RECOMMENDED: mean_error=%.3f calibration_bias=%.3f (n=%d, days=%d)",
            mean_error, stats["calibration_bias"], n, days,
        )

    return stats


def export_feedback_for_retraining(
    db: Database,
    output_path: str,
    days: int = 90,
) -> int:
    """
    Export feedback records as CSV for retraining pipeline.
    Returns number of records exported.
    """
    import pandas as pd
    from datetime import timedelta

    cutoff = datetime.utcnow() - timedelta(days=days)
    docs = list(db[_FEEDBACK_COLLECTION].find(
        {"recorded_at": {"$gte": cutoff}},
        {"_id": 0, "appointment_id": 0},
    ))

    if not docs:
        logger.info("No feedback records found for last %d days", days)
        return 0

    df = pd.DataFrame(docs)
    df.to_csv(output_path, index=False)
    logger.info("Exported %d feedback records to %s", len(df), output_path)
    return len(df)
