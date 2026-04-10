"""Tests for database layer."""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.db_connection import Base
from src.database.models import Appointment, Patient, Provider
from src.database.queries import (
    get_booked_slots,
    get_patient_data,
    get_provider_data,
    get_provider_schedule,
    insert_appointment,
    update_appointment_status,
)


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_get_patient_data_defaults(db):
    """Unknown patient returns safe defaults."""
    data = get_patient_data(db, patient_encoded=999)
    assert data["patient_encoded"] == 999
    assert data["patient_hist_success_rate"] == 0.5
    assert data["patient_total_appts"] == 1


def test_get_provider_data_defaults(db):
    data = get_provider_data(db, provider_encoded=999)
    assert data["provider_encoded"] == 999
    assert data["working_days"] == [0, 1, 2, 3, 4]


def test_insert_and_retrieve_appointment(db):
    appt = insert_appointment(
        db=db,
        patient_encoded=1,
        provider_encoded=2,
        appt_date="2026-04-01",
        appt_hour=10,
        duration_minutes=60,
        visit_reason="checkup",
    )
    assert appt.id is not None
    assert appt.appt_date == "2026-04-01"
    assert appt.appt_hour == 10
    assert appt.status == "Confirmation Pending"


def test_update_appointment_status(db):
    appt = insert_appointment(db, 1, 2, "2026-04-01", 10)
    updated = update_appointment_status(db, appt.id, "Confirmed")
    assert updated.status == "Confirmed"


def test_get_booked_slots(db):
    insert_appointment(db, 1, 3, "2026-04-01", 9)
    insert_appointment(db, 2, 3, "2026-04-01", 10)
    booked = get_booked_slots(db, provider_encoded=3, date_iso="2026-04-01")
    assert 9 in booked
    assert 10 in booked


def test_patient_data_with_history(db):
    insert_appointment(db, 5, 1, "2026-03-01", 9)
    insert_appointment(db, 5, 1, "2026-03-15", 10)
    data = get_patient_data(db, patient_encoded=5)
    assert data["patient_total_appts"] == 2
    assert data["patient_encoded"] == 5
