import re
from datetime import datetime, timedelta
from typing import Any, Dict, Optional


WEEKDAY_MAP = {
    "monday": 0, "tuesday": 1, "wednesday": 2,
    "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
}

TIME_KEYWORDS: Dict[str, list] = {
    "morning": ["morning", "early morning", "early", "am"],
    "midday": ["midday", "noon", "lunch", "12"],
    "afternoon": ["afternoon", "pm"],
    "evening": ["evening", "late"],
}

RELATIVE_DATE_PATTERNS = {
    "today": 0,
    "tomorrow": 1,
    "day after tomorrow": 2,
}


def _next_weekday(today: datetime, target_weekday: int) -> datetime:
    """Return the next occurrence of target_weekday after today."""
    days_ahead = (target_weekday - today.weekday() + 7) % 7
    if days_ahead == 0:
        days_ahead = 7
    return today + timedelta(days=days_ahead)


def parse_appointment_request(text: str) -> Dict[str, Optional[Any]]:
    """Parse natural language appointment request into structured constraints.

    Handles:
    - Relative dates: today, tomorrow, next monday, next week
    - Time of day: morning, afternoon, evening, noon
    - Provider: Dr 3, Dr. Smith, doctor 5
    - Explicit ISO dates: 2026-04-01
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string")

    out: Dict[str, Optional[Any]] = {
        "weekday": None,
        "preferred_time": None,
        "provider_encoded": None,
        "date": None,
        "raw_text": text,
    }

    normalized = text.lower().strip()
    today = datetime.utcnow()

    # ── Explicit ISO date ──────────────────────────────────────────────────────
    iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", normalized)
    if iso_match:
        out["date"] = iso_match.group(1)

    # ── Relative dates ─────────────────────────────────────────────────────────
    if not out["date"]:
        for phrase, offset in RELATIVE_DATE_PATTERNS.items():
            if phrase in normalized:
                out["date"] = (today + timedelta(days=offset)).date().isoformat()
                break

    # ── "next week" ────────────────────────────────────────────────────────────
    if not out["date"] and "next week" in normalized and not any(d in normalized for d in WEEKDAY_MAP):
        out["date"] = (today + timedelta(days=7)).date().isoformat()

    # ── "next <weekday>" or just "<weekday>" ───────────────────────────────────
    for day_name, day_idx in WEEKDAY_MAP.items():
        if day_name in normalized:
            out["weekday"] = day_idx
            if "next" in normalized or not out["date"]:
                out["date"] = _next_weekday(today, day_idx).date().isoformat()
            break

    # ── Time of day ────────────────────────────────────────────────────────────
    for time_name, keywords in TIME_KEYWORDS.items():
        if any(re.search(rf"\b{re.escape(k)}\b", normalized) for k in keywords):
            out["preferred_time"] = time_name
            break

    # ── Provider extraction: "Dr 3", "Dr. Smith", "doctor 5", "provider 2" ────
    provider_match = re.search(
        r"(?:dr\.?\s*|doctor\s*|provider\s*)(\d+|[a-z]+(?:\s+[a-z]+)?)",
        normalized,
    )
    if provider_match:
        raw = provider_match.group(1).strip()
        if raw.isdigit():
            out["provider_encoded"] = int(raw)
        else:
            # Named provider — store name for lookup; encoded resolved later
            out["provider_name"] = raw.title()

    return out
