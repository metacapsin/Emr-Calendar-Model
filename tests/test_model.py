import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.api.nlp_parser import parse_appointment_request
from src.recommendation.recommender import AppointmentRecommender

client = TestClient(app)


def test_nlp_parser_basic():
    parsed = parse_appointment_request('I want appointment next Monday morning with Dr 3')
    assert parsed['weekday'] == 0
    assert parsed['preferred_time'] == 'morning'
    assert parsed['provider_encoded'] == 3


def test_recommendation_end_to_end():
    recommender = AppointmentRecommender(config_path='configs/config.yaml')

    patient_data = {
        'patient_age': 40,
        'sex_encoded': 1,
        'patient_total_appts': 5,
        'patient_cancel_rate': 0.1,
        'patient_7day_appts': 1,
        'patient_30day_appts': 3,
        'patient_7day_cancel': 0,
        'patient_30day_cancel': 0,
        'patient_encoded': 10,
    }

    provider_data = {
        'provider_encoded': 2,
        'provider_total_appts': 300,
        'provider_avg_duration': 30,
        'provider_cancel_rate': 0.05,
        'provider_7day_util': 0.75,
        'provider_30day_util': 0.80,
        'working_days': [0, 1, 2, 3, 4],
        'hours': {'start': 8, 'end': 16},
    }

    results = recommender.recommend_slots(
        request_text='I want appointment tomorrow morning with Dr 2',
        patient_data=patient_data,
        provider_data=provider_data,
        top_k=3,
    )

    assert isinstance(results, list)
    assert len(results) <= 3
    for item in results:
        assert 'date' in item and 'hour' in item and 'prob' in item


def test_api_recommend_slots():
    payload = {
        'text': 'I want appointment next week morning with Dr 2',
        'patient_data': {
            'patient_age': 37,
            'sex_encoded': 0,
            'patient_total_appts': 8,
            'patient_cancel_rate': 0.05,
            'patient_encoded': 11,
        },
        'provider_data': {
            'provider_encoded': 2,
            'provider_total_appts': 350,
            'provider_avg_duration': 30,
            'provider_cancel_rate': 0.04,
            'provider_7day_util': 0.6,
            'provider_30day_util': 0.72,
            'working_days': [0, 1, 2, 3, 4],
            'hours': {'start': 8, 'end': 16},
        },
        'top_k': 3,
    }

    response = client.post('/recommend-slots', json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 'recommended_slots' in data
    assert isinstance(data['recommended_slots'], list)
