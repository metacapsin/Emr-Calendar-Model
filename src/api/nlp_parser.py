import re
from datetime import datetime, timedelta
from typing import Any, Dict, Optional


WEEKDAY_MAP = {
    'monday': 0,
    'tuesday': 1,
    'wednesday': 2,
    'thursday': 3,
    'friday': 4,
    'saturday': 5,
    'sunday': 6,
}

TIME_KEYWORDS = {
    'morning': ['morning', 'am', 'early'],
    'midday': ['midday', 'noon', 'lunch'],
    'afternoon': ['afternoon', 'pm'],
    'evening': ['evening', 'late'],
}


def parse_appointment_request(text: str) -> Dict[str, Optional[Any]]:
    """Parse natural language scheduler text into structured constraints."""
    if not isinstance(text, str):
        raise ValueError('Text input must be a string')

    out = {
        'weekday': None,
        'preferred_time': None,
        'provider_encoded': None,
        'date': None,
    }

    normalized = text.lower()

    for word, index in WEEKDAY_MAP.items():
        if word in normalized:
            out['weekday'] = index
            break

    for time_name, keywords in TIME_KEYWORDS.items():
        if any(k in normalized for k in keywords):
            out['preferred_time'] = time_name
            break

    doctor_match = re.search(r'dr(?:\s+|\.\s*)?(\d+)', normalized)
    if doctor_match:
        out['provider_encoded'] = int(doctor_match.group(1))

    tomorrow = datetime.utcnow().date() + timedelta(days=1)
    if 'tomorrow' in normalized:
        out['date'] = tomorrow.isoformat()

    if 'next week' in normalized:
        out['date'] = (datetime.utcnow().date() + timedelta(days=7)).isoformat()

    for word, idx in WEEKDAY_MAP.items():
        if word in normalized and 'next' in normalized:
            today = datetime.utcnow().date()
            current_weekday = today.weekday()
            offset = (idx - current_weekday + 7) % 7
            if offset == 0:
                offset = 7
            out['date'] = (today + timedelta(days=offset)).isoformat()
            break

    return out
