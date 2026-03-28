import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.api.nlp_parser import parse_appointment_request
from src.features.slot_feature_builder import build_slots_feature_dataframe
from src.models.inference import SlotInferenceEngine
from src.recommendation.slot_ranker import aggregate_recommendations, rank_slots
from src.scheduling.slot_generator import generate_candidate_slots


logger = logging.getLogger(__name__)


def _load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    cfg_file = config_path or 'configs/config.yaml'
    p = Path(cfg_file)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {cfg_file}")

    with p.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class AppointmentRecommender:
    def __init__(self, config_path: Optional[str] = None):
        self.config = _load_config(config_path)
        self.engine = SlotInferenceEngine(self.config['model_path'])

    def recommend_slots(
        self,
        request_text: str,
        patient_data: Dict[str, Any],
        provider_data: Dict[str, Any],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        params = parse_appointment_request(request_text)
        now = datetime.utcnow().date()
        start_date = now + timedelta(days=self.config['slot_recommendation'].get('search_start_days', 1))
        end_date = start_date + timedelta(days=self.config['slot_recommendation'].get('search_days', 7))

        provider_availability = {
            'provider_encoded': provider_data.get('provider_encoded'),
            'working_days': provider_data.get('working_days', list(range(5))),
            'hours': provider_data.get('hours', self.config['slot_recommendation']['working_hours']),
        }

        slots = generate_candidate_slots(
            start_date=datetime.combine(start_date, datetime.min.time()),
            end_date=datetime.combine(end_date, datetime.min.time()),
            provider_availability=provider_availability,
            preferred_time_of_day=params.get('preferred_time'),
            slot_duration_minutes=self.config['slot_recommendation'].get('slot_duration_minutes', 60),
            slot_step_minutes=self.config['slot_recommendation'].get('slot_step_minutes', 60),
            working_hours=self.config['slot_recommendation'].get('working_hours'),
        )

        if params.get('weekday') is not None:
            slots = [s for s in slots if s['weekday'] == params['weekday']]

        if params.get('date') is not None:
            slots = [s for s in slots if s['date'] == params['date']]

        if not slots:
            logger.warning('No candidate slots generated for request: %s', request_text)
            return []

        feature_df = build_slots_feature_dataframe(slots, patient_data, provider_data, self.engine.feature_columns)

        probabilities = self.engine.predict_proba(feature_df)

        results: List[Dict[str, Any]] = []
        for i, slot in enumerate(slots):
            results.append(
                {
                    'date': slot['date'],
                    'hour': slot['hour'],
                    'prob': float(probabilities[i][1]),
                    'provider_encoded': slot.get('provider_encoded'),
                }
            )

        ranked = rank_slots(
            candidates=results,
            top_k=top_k or self.config['slot_recommendation'].get('top_k', 5),
            cost_fn=self.config['slot_recommendation'].get('cost_fn', 1000),
            cost_fp=self.config['slot_recommendation'].get('cost_fp', 200),
            min_probability=0.0,
        )

        final = aggregate_recommendations(ranked, top_n=top_k or self.config['slot_recommendation'].get('top_k', 5))

        return final
