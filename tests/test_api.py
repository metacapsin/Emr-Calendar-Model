"""API integration tests."""
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

PATIENT = {
    "patient_encoded": 10,
    "patient_age": 40,
    "sex_encoded": 1,
    "patient_total_appts": 5,
    "patient_cancel_rate": 0.1,
    "patient_7day_appts": 1,
    "patient_30day_appts": 3,
    "patient_7day_cancel": 0,
    "patient_30day_cancel": 0,
}

PROVIDER = {
    "provider_encoded": 3,
    "provider_total_appts": 300,
    "provider_avg_duration": 30,
    "provider_cancel_rate": 0.05,
    "provider_7day_util": 0.6,
    "provider_30day_util": 0.75,
    "working_days": [0, 1, 2, 3, 4],
    "hours": {"start": 8, "end": 17},
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_recommend_slots_success():
    payload = {
        "text": "I want appointment next monday morning with Dr 3",
        "patient_data": PATIENT,
        "provider_data": PROVIDER,
        "top_k": 3,
    }
    r = client.post("/recommend-slots", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "recommended_slots" in data
    assert isinstance(data["recommended_slots"], list)
    assert "parsed_intent" in data


def test_recommend_slots_invalid_text():
    payload = {"text": "  ", "patient_data": PATIENT, "provider_data": PROVIDER}
    r = client.post("/recommend-slots", json=payload)
    assert r.status_code == 422


def test_book_appointment():
    payload = {
        "patient_encoded": 1,
        "provider_encoded": 2,
        "appt_date": "2026-04-01",
        "appt_hour": 10,
        "duration_minutes": 60,
        "visit_reason": "checkup",
        "is_telehealth": False,
        "is_new_patient": True,
    }
    r = client.post("/book-appointment", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["appt_date"] == "2026-04-01"
    assert data["appt_hour"] == 10
    assert "appointment_id" in data


def test_provider_slots():
    r = client.get("/provider-slots?provider_encoded=2")
    assert r.status_code == 200
    data = r.json()
    assert data["provider_encoded"] == 2
    assert "appointments" in data


def test_patient_history():
    r = client.get("/patient-history?patient_encoded=1")
    assert r.status_code == 200
    data = r.json()
    assert data["patient_encoded"] == 1
    assert "appointments" in data


def test_recommend_slots_with_tomorrow():
    payload = {
        "text": "book tomorrow afternoon with Dr 2",
        "patient_data": PATIENT,
        "provider_data": {**PROVIDER, "provider_encoded": 2},
        "top_k": 5,
    }
    r = client.post("/recommend-slots", json=payload)
    assert r.status_code == 200


def test_recommend_slots_returns_prob_range():
    payload = {
        "text": "next friday morning Dr 3",
        "patient_data": PATIENT,
        "provider_data": PROVIDER,
        "top_k": 5,
    }
    r = client.post("/recommend-slots", json=payload)
    assert r.status_code == 200
    for slot in r.json()["recommended_slots"]:
        assert 0.0 <= slot["prob"] <= 1.0
