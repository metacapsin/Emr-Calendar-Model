from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ── Shared ─────────────────────────────────────────────────────────────────────

class SlotItem(BaseModel):
    date: str
    time: str
    hour: int
    prob: float
    provider_encoded: Optional[int] = None
    score: Optional[float] = None


# ── Recommend Slots ────────────────────────────────────────────────────────────

class PatientData(BaseModel):
    patient_encoded: Optional[int] = None
    patient_age: int = Field(default=35, ge=0, le=120)
    sex_encoded: int = Field(default=1, ge=0, le=1)
    patient_total_appts: int = Field(default=1, ge=0)
    patient_cancel_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    patient_7day_appts: int = Field(default=0, ge=0)
    patient_30day_appts: int = Field(default=0, ge=0)
    patient_7day_cancel: int = Field(default=0, ge=0)
    patient_30day_cancel: int = Field(default=0, ge=0)
    patient_provider_history: int = Field(default=0, ge=0)
    model_config = {"extra": "allow"}


class ProviderData(BaseModel):
    provider_encoded: Optional[int] = None
    provider_total_appts: int = Field(default=1, ge=0)
    provider_avg_duration: int = Field(default=30, ge=5)
    provider_cancel_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    provider_7day_util: float = Field(default=0.5, ge=0.0, le=1.0)
    provider_30day_util: float = Field(default=0.5, ge=0.0, le=1.0)
    working_days: List[int] = Field(default=[0, 1, 2, 3, 4])
    hours: Dict[str, int] = Field(default={"start": 8, "end": 17})
    model_config = {"extra": "allow"}


class RecommendSlotsRequest(BaseModel):
    text: str = Field(..., min_length=3, description="Natural language appointment request")
    patient_data: PatientData = Field(default_factory=PatientData)
    provider_data: ProviderData = Field(default_factory=ProviderData)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text cannot be blank")
        return v.strip()


class RecommendSlotsResponse(BaseModel):
    recommended_slots: List[SlotItem]
    total: int
    parsed_intent: Optional[Dict[str, Any]] = None


# ── Book Appointment ───────────────────────────────────────────────────────────

class BookAppointmentRequest(BaseModel):
    patient_encoded: int = Field(..., ge=0)
    provider_encoded: int = Field(..., ge=0)
    appt_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    appt_hour: int = Field(..., ge=0, le=23)
    duration_minutes: int = Field(default=60, ge=15)
    visit_reason: str = Field(default="")
    is_telehealth: bool = False
    is_new_patient: bool = False


class BookAppointmentResponse(BaseModel):
    appointment_id: str          # MongoDB ObjectId as string
    status: str
    appt_date: str
    appt_hour: int
    patient_encoded: int
    provider_encoded: int


# ── Provider Slots ─────────────────────────────────────────────────────────────

class ProviderSlotsResponse(BaseModel):
    provider_encoded: int
    date: Optional[str] = None
    appointments: List[Dict[str, Any]]


# ── Patient History ────────────────────────────────────────────────────────────

class PatientHistoryResponse(BaseModel):
    patient_encoded: int
    appointments: List[Dict[str, Any]]
    total: int
