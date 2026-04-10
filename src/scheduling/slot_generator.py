from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from src.utils.logger import get_logger

logger = get_logger(__name__)

TIME_OF_DAY: Dict[str, range] = {
    "morning": range(8, 11),
    "midday": range(11, 13),
    "afternoon": range(13, 16),
    "evening": range(16, 19),
}


def generate_candidate_slots(
    start_date: datetime,
    end_date: datetime,
    provider_availability: Optional[Dict[str, Any]] = None,
    preferred_time_of_day: Optional[str] = None,
    slot_duration_minutes: int = 60,
    slot_step_minutes: int = 60,
    working_hours: Optional[Dict[str, int]] = None,
    blocked_dates: Optional[List[str]] = None,
    booked_slots_by_date: Optional[Dict[str, List[int]]] = None,
) -> List[Dict[str, Any]]:
    """Generate available candidate slots respecting provider schedule and bookings.

    Args:
        start_date: inclusive start datetime
        end_date: inclusive end datetime
        provider_availability: dict with provider_encoded, working_days, hours
        preferred_time_of_day: morning/midday/afternoon/evening filter
        slot_duration_minutes: length of each appointment
        slot_step_minutes: step between slot start times
        working_hours: fallback working hours {start, end}
        blocked_dates: ISO date strings that are holidays/blocked
        booked_slots_by_date: {date_iso: [hour, ...]} already booked hours
    """
    if working_hours is None:
        working_hours = {"start": 8, "end": 17}

    blocked: Set[str] = set(blocked_dates or [])
    booked: Dict[str, Set[int]] = {
        d: set(hours) for d, hours in (booked_slots_by_date or {}).items()
    }

    working_days: List[int] = list(range(5))  # Mon–Fri default
    day_start = working_hours["start"]
    day_end = working_hours["end"]

    if provider_availability:
        working_days = provider_availability.get("working_days", working_days)
        hours_cfg = provider_availability.get("hours", {})
        day_start = hours_cfg.get("start", day_start)
        day_end = hours_cfg.get("end", day_end)

    preferred_range = TIME_OF_DAY.get(preferred_time_of_day) if preferred_time_of_day else None

    candidate_slots: List[Dict[str, Any]] = []
    current = start_date.replace(hour=0, minute=0, second=0, microsecond=0)

    while current.date() <= end_date.date():
        date_iso = current.date().isoformat()
        weekday = current.weekday()

        if weekday not in working_days:
            current += timedelta(days=1)
            continue

        if date_iso in blocked:
            logger.debug("Skipping blocked date: %s", date_iso)
            current += timedelta(days=1)
            continue

        booked_hours = booked.get(date_iso, set())
        hour = day_start

        while hour < day_end:
            if preferred_range and hour not in preferred_range:
                hour += slot_step_minutes // 60 if slot_step_minutes >= 60 else 1
                continue

            if hour in booked_hours:
                hour += 1
                continue

            slot_end_hour = hour + slot_duration_minutes // 60
            if slot_end_hour > day_end:
                break

            candidate_slots.append(
                {
                    "date": date_iso,
                    "hour": hour,
                    "minute": 0,
                    "weekday": weekday,
                    "month": current.month,
                    "day": current.day,
                    "year": current.year,
                    "appt_quarter": (current.month - 1) // 3 + 1,
                    "slot_duration_minutes": slot_duration_minutes,
                    "provider_encoded": provider_availability.get("provider_encoded") if provider_availability else None,
                    "is_holiday": 0,
                    "is_peak_day": 1 if weekday in (0, 4) else 0,
                }
            )

            hour += max(1, slot_step_minutes // 60)

        current += timedelta(days=1)

    logger.debug("Generated %d candidate slots", len(candidate_slots))
    return candidate_slots
