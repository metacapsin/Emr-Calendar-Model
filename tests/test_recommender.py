"""Tests for the full recommendation pipeline."""
import pytest

from src.api.nlp_parser import parse_appointment_request
from src.recommendation.recommender import AppointmentRecommender
from src.recommendation.slot_ranker import aggregate_recommendations, rank_slots
from src.scheduling.slot_generator import generate_candidate_slots
from datetime import datetime, timedelta


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


# ── NLP Parser ─────────────────────────────────────────────────────────────────

def test_nlp_next_monday_morning_dr3():
    parsed = parse_appointment_request("I want appointment next monday morning with Dr 3")
    assert parsed["weekday"] == 0
    assert parsed["preferred_time"] == "morning"
    assert parsed["provider_encoded"] == 3
    assert parsed["date"] is not None


def test_nlp_tomorrow():
    parsed = parse_appointment_request("book tomorrow afternoon")
    assert parsed["preferred_time"] == "afternoon"
    expected = (datetime.utcnow() + timedelta(days=1)).date().isoformat()
    assert parsed["date"] == expected


def test_nlp_invalid_input():
    with pytest.raises(ValueError):
        parse_appointment_request("")


# ── Slot Generator ─────────────────────────────────────────────────────────────

def test_slot_generator_respects_working_days():
    start = datetime(2026, 4, 6)   # Monday
    end = datetime(2026, 4, 10)    # Friday
    slots = generate_candidate_slots(
        start_date=start,
        end_date=end,
        provider_availability={"provider_encoded": 1, "working_days": [0, 1, 2, 3, 4], "hours": {"start": 9, "end": 12}},
    )
    assert all(s["weekday"] < 5 for s in slots)
    assert all(9 <= s["hour"] < 12 for s in slots)


def test_slot_generator_blocks_dates():
    start = datetime(2026, 4, 6)
    end = datetime(2026, 4, 8)
    slots = generate_candidate_slots(
        start_date=start,
        end_date=end,
        blocked_dates=["2026-04-07"],
    )
    assert all(s["date"] != "2026-04-07" for s in slots)


def test_slot_generator_skips_booked_hours():
    start = datetime(2026, 4, 7)
    end = datetime(2026, 4, 7)
    slots = generate_candidate_slots(
        start_date=start,
        end_date=end,
        booked_slots_by_date={"2026-04-07": [9, 10]},
        provider_availability={"working_days": [0], "hours": {"start": 9, "end": 12}},
    )
    assert all(s["hour"] not in (9, 10) for s in slots)


# ── Ranker ─────────────────────────────────────────────────────────────────────

def test_rank_slots_orders_by_score():
    candidates = [
        {"date": "2026-04-01", "hour": 9, "prob": 0.9, "provider_7day_util": 0.5},
        {"date": "2026-04-01", "hour": 10, "prob": 0.6, "provider_7day_util": 0.5},
        {"date": "2026-04-01", "hour": 11, "prob": 0.75, "provider_7day_util": 0.5},
    ]
    ranked = rank_slots(candidates, top_k=3)
    assert ranked[0]["prob"] >= ranked[1]["prob"]


def test_aggregate_unique_per_day():
    results = [
        {"date": "2026-04-01", "hour": 9, "prob": 0.9, "score": 0.9},
        {"date": "2026-04-01", "hour": 10, "prob": 0.85, "score": 0.85},
        {"date": "2026-04-02", "hour": 9, "prob": 0.8, "score": 0.8},
    ]
    agg = aggregate_recommendations(results, top_n=3, unique_per_day=True)
    dates = [r["date"] for r in agg]
    assert len(dates) == len(set(dates))


# ── Full Recommender ───────────────────────────────────────────────────────────

def test_recommender_end_to_end():
    recommender = AppointmentRecommender(config_path="configs/config.yaml")
    results = recommender.recommend_slots(
        request_text="I want appointment next monday morning with Dr 3",
        patient_data=PATIENT,
        provider_data=PROVIDER,
        top_k=3,
    )
    assert isinstance(results, list)
    assert len(results) <= 3
    for item in results:
        assert "date" in item
        assert "time" in item
        assert "prob" in item
        assert 0.0 <= item["prob"] <= 1.0


def test_recommender_returns_empty_for_impossible_request():
    recommender = AppointmentRecommender(config_path="configs/config.yaml")
    # Provider only works Mon–Fri, request Saturday
    provider = {**PROVIDER, "working_days": [0, 1, 2, 3, 4]}
    results = recommender.recommend_slots(
        request_text="I want appointment next saturday morning with Dr 3",
        patient_data=PATIENT,
        provider_data=provider,
        top_k=3,
    )
    assert results == []
