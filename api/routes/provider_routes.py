from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pymongo.database import Database

from api.schemas import ProviderSlotsResponse
from src.database import get_db, get_provider_appointments, get_provider_data, get_provider_schedule
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/providers", tags=["providers"])


@router.get("/{provider_id}", response_model=dict)
def get_provider(provider_id: str, db: Database = Depends(get_db)):
    data = get_provider_data(db, provider_id)
    if not data:
        raise HTTPException(status_code=404, detail="Provider not found")
    return data


@router.get("/{provider_id}/schedule", response_model=dict)
def get_schedule(provider_id: str, db: Database = Depends(get_db)):
    blocked = get_provider_schedule(db, provider_id)
    return {"provider_id": provider_id, "blocked_dates": blocked}


@router.get("/{provider_id}/appointments", response_model=ProviderSlotsResponse)
def get_appointments(
    provider_id: str,
    date: Optional[str] = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    db: Database = Depends(get_db),
):
    appts = get_provider_appointments(db, provider_id, date)
    return ProviderSlotsResponse(provider_id=provider_id, date=date, appointments=appts)
