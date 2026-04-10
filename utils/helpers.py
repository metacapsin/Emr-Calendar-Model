from datetime import datetime
from typing import Any, Dict, List


def format_slot_time(hour: int, minute: int = 0) -> str:
    return f"{hour:02d}:{minute:02d}"


def iso_to_weekday_name(date_iso: str) -> str:
    return datetime.fromisoformat(date_iso).strftime("%A")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator else default


def filter_slots_by_date(slots: List[Dict[str, Any]], date_iso: str) -> List[Dict[str, Any]]:
    return [s for s in slots if s.get("date") == date_iso]


def deduplicate_by_key(items: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    seen = set()
    result = []
    for item in items:
        val = item.get(key)
        if val not in seen:
            seen.add(val)
            result.append(item)
    return result
