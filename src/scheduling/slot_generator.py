from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


def generate_candidate_slots(
    start_date: datetime,
    end_date: datetime,
    provider_availability: Optional[Dict[str, Any]] = None,
    preferred_time_of_day: Optional[str] = None,
    slot_duration_minutes: int = 60,
    slot_step_minutes: int = 60,
    working_hours: Dict[str, int] = None,
) -> List[Dict[str, Any]]:
    """Generate candidate slots between start_date and end_date.

    This generator is intentionally simple and fast.

    Args:
        start_date: start datetime (inclusive)
        end_date: end datetime (inclusive)
        provider_availability: optional provider availability structure:
            {
                'provider_encoded': 3,
                'working_days': [0,1,2,3,4],
                'hours': {'start':8, 'end':17}
            }
        preferred_time_of_day: one of morning/midday/afternoon/evening. If set,
            only return slots in that window.
        slot_duration_minutes: length of each slot.
        slot_step_minutes: step between available slot start times.
        working_hours: fallback working hours mapping.

    Returns:
        List of candidate slot dicts.
    """
    if working_hours is None:
        working_hours = {'start': 8, 'end': 17}

    time_of_day = {
        'morning': range(8, 11),
        'midday': range(11, 13),
        'afternoon': range(13, 16),
        'evening': range(16, 19),
    }

    candidate_slots: List[Dict[str, Any]] = []

    current = start_date.replace(hour=0, minute=0, second=0, microsecond=0)

    while current.date() <= end_date.date():
        weekday = current.weekday()

        if provider_availability:
            days = provider_availability.get('working_days', list(range(7)))
        else:
            days = list(range(7))

        if weekday in days:
            if provider_availability and 'hours' in provider_availability:
                hours = provider_availability['hours']
                day_start = hours.get('start', working_hours['start'])
                day_end = hours.get('end', working_hours['end'])
            else:
                day_start = working_hours['start']
                day_end = working_hours['end']

            for hour in range(day_start, day_end):
                if preferred_time_of_day and preferred_time_of_day in time_of_day:
                    if hour not in time_of_day[preferred_time_of_day]:
                        continue

                slot_start = current.replace(hour=hour, minute=0, second=0, microsecond=0)
                slot_end = slot_start + timedelta(minutes=slot_duration_minutes)

                if slot_end.date() > current.date() and slot_end.hour > day_end + 1:
                    continue

                candidate_slots.append(
                    {
                        'date': slot_start.date().isoformat(),
                        'hour': slot_start.hour,
                        'minute': slot_start.minute,
                        'weekday': slot_start.weekday(),
                        'month': slot_start.month,
                        'day': slot_start.day,
                        'appt_quarter': (slot_start.month - 1) // 3 + 1,
                        'slot_duration_minutes': slot_duration_minutes,
                        'provider_encoded': provider_availability.get('provider_encoded')
                        if provider_availability
                        else None,
                    }
                )
                # step by slot_step
                if slot_step_minutes != 60:
                    step = timedelta(minutes=slot_step_minutes)
                    slot_start += step

        current += timedelta(days=1)

    return candidate_slots
