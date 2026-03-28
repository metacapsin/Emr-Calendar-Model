from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.recommendation.recommender import AppointmentRecommender

router = APIRouter(prefix="", tags=["slot_recommendations"])


class PredictSlotRequest(BaseModel):
    text: str
    patient_data: dict
    provider_data: dict
    top_k: int = 5


class SlotItem(BaseModel):
    date: str
    hour: int
    prob: float
    provider_encoded: int


class PredictSlotResponse(BaseModel):
    recommended_slots: list


@router.post('/recommend-slots', response_model=PredictSlotResponse)
def recommend_slots(payload: PredictSlotRequest):
    try:
        recommender = AppointmentRecommender()
        slots = recommender.recommend_slots(
            request_text=payload.text,
            patient_data=payload.patient_data,
            provider_data=payload.provider_data,
            top_k=payload.top_k,
        )

        return {'recommended_slots': slots}

    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
