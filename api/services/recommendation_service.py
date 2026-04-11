from typing import Any, Dict, List, Optional

from pymongo.database import Database

from src.recommendation.recommender import AppointmentRecommender
from src.utils.logger import get_logger

logger = get_logger(__name__)

_recommender: Optional[AppointmentRecommender] = None


def get_recommender() -> AppointmentRecommender:
    global _recommender
    if _recommender is None:
        _recommender = AppointmentRecommender()
    return _recommender


def recommend(
    text: str,
    patient_data: Dict[str, Any],
    provider_data: Dict[str, Any],
    top_k: int = 5,
    db: Optional[Database] = None,
) -> List[Dict[str, Any]]:
    recommender = get_recommender()
    return recommender.recommend_slots(
        request_text=text,
        patient_data=patient_data,
        provider_data=provider_data,
        top_k=top_k,
        db=db,
    )
