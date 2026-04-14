from fastapi import APIRouter, Depends, HTTPException
from pymongo.database import Database

from api.schemas import (
    RecommendSlotsRequest,
    RecommendSlotsResponse,
    SlotItem,
)
from api.services.recommendation_service import recommend
from src.api.nlp_parser import parse_appointment_request
from src.database import get_db
from src.database.errors import EntityNotFoundError
from src.utils.logger import get_logger, log_request

logger = get_logger(__name__)
router = APIRouter(prefix="", tags=["slots"])


@router.post("/recommend-slots", response_model=RecommendSlotsResponse)
def recommend_slots(payload: RecommendSlotsRequest, db: Database = Depends(get_db)):
    log_request(logger, "/recommend-slots", payload.model_dump())
    try:
        parsed       = parse_appointment_request(payload.text)
        patient_dict = payload.patient_data.model_dump()
        provider_dict = payload.provider_data.model_dump()

        # Inject patient_name from NLP if not already resolved
        if parsed.get("patient_name") and not patient_dict.get("patient_id"):
            patient_dict["patient_name"] = parsed["patient_name"]

        slots = recommend(
            text=payload.text,
            patient_data=patient_dict,
            provider_data=provider_dict,
            top_k=payload.top_k,
            db=db,
        )

        items = [
            SlotItem(
                date=s["date"],
                time=s.get("time", f"{s['hour']:02d}:00"),
                hour=s["hour"],
                prob=s["prob"],
                patient_id=s.get("patient_id"),
                provider_id=s.get("provider_id"),
                score=s.get("score"),
            )
            for s in slots
        ]
        return RecommendSlotsResponse(
            recommended_slots=items,
            total=len(items),
            parsed_intent=parsed,
        )
    except EntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail=exc.to_dict())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("recommend-slots error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal recommendation error")
