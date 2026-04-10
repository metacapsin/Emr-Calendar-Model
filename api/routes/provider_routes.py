from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.schemas import ProviderSlotsResponse
from src.database import get_db, get_provider_appointments, get_provider_data, get_provider_schedule
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/providers", tags=["providers"])


@router.get("/{provider_encoded}", response_model=dict)
def get_provider(provider_encoded: int, db: Session = Depends(get_db)):
    data = get_provider_data(db, provider_encoded)
    if not data:
        raise HTTPException(status_code=404, detail="Provider not found")
    return data


@router.get("/{provider_encoded}/schedule", response_model=dict)
def get_schedule(provider_encoded: int, db: Session = Depends(get_db)):
    blocked = get_provider_schedule(db, provider_encoded)
    return {"provider_encoded": provider_encoded, "blocked_dates": blocked}


@router.get("/{provider_encoded}/appointments", response_model=ProviderSlotsResponse)
def get_appointments(
    provider_encoded: int,
    date: Optional[str] = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    db: Session = Depends(get_db),
):
    appts = get_provider_appointments(db, provider_encoded, date)
    return ProviderSlotsResponse(
        provider_encoded=provider_encoded,
        date=date,
        appointments=appts,
    )
