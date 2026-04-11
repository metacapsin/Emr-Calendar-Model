from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pymongo.database import Database

from api.schemas import (
    BookAppointmentRequest,
    BookAppointmentResponse,
    PatientHistoryResponse,
    ProviderSlotsResponse,
    RecommendSlotsRequest,
    RecommendSlotsResponse,
    SlotItem,
)
from api.services.recommendation_service import recommend
from src.api.nlp_parser import parse_appointment_request
from src.database import (
    get_db,
    get_patient_appointments,
    get_provider_appointments,
    insert_appointment,
)
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
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("recommend-slots error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal recommendation error")


@router.post("/book-appointment", response_model=BookAppointmentResponse)
def book_appointment(payload: BookAppointmentRequest, db: Database = Depends(get_db)):
    log_request(logger, "/book-appointment", payload.model_dump())
    try:
        appt = insert_appointment(
            db=db,
            patient_id=payload.patient_id,
            provider_id=payload.provider_id,
            appt_date=payload.appt_date,
            appt_hour=payload.appt_hour,
            duration_minutes=payload.duration_minutes,
            visit_reason=payload.visit_reason,
            is_telehealth=payload.is_telehealth,
            is_new_patient=payload.is_new_patient,
        )
        return BookAppointmentResponse(
            appointment_id=str(appt.get("_id", "")),
            status=str(appt.get("status", "")).strip(),
            appt_date=appt["appt_date"],
            appt_hour=appt["appt_hour"],
            patient_id=payload.patient_id,
            provider_id=payload.provider_id,
        )
    except Exception as exc:
        logger.error("book-appointment error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/provider-slots", response_model=ProviderSlotsResponse)
def provider_slots(
    provider_id: str = Query(..., description="MongoDB _id of provider"),
    date: Optional[str] = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    db: Database = Depends(get_db),
):
    appts = get_provider_appointments(db, provider_id, date)
    return ProviderSlotsResponse(provider_id=provider_id, date=date, appointments=appts)


@router.get("/patient-history", response_model=PatientHistoryResponse)
def patient_history(
    patient_id: str = Query(..., description="MongoDB _id of patient"),
    db: Database = Depends(get_db),
):
    appts = get_patient_appointments(db, patient_id)
    return PatientHistoryResponse(patient_id=patient_id, appointments=appts, total=len(appts))
