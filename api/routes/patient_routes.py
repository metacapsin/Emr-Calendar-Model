from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.schemas import PatientHistoryResponse
from src.database import get_db, get_patient_appointments, get_patient_data
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/patients", tags=["patients"])


@router.get("/{patient_encoded}", response_model=dict)
def get_patient(patient_encoded: int, db: Session = Depends(get_db)):
    data = get_patient_data(db, patient_encoded)
    if not data:
        raise HTTPException(status_code=404, detail="Patient not found")
    return data


@router.get("/{patient_encoded}/history", response_model=PatientHistoryResponse)
def get_patient_history(patient_encoded: int, db: Session = Depends(get_db)):
    appts = get_patient_appointments(db, patient_encoded)
    return PatientHistoryResponse(
        patient_encoded=patient_encoded,
        appointments=appts,
        total=len(appts),
    )
