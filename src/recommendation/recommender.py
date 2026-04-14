from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import re
import yaml
from pymongo.database import Database

from src.api.nlp_parser import parse_appointment_request
from src.database.errors import EntityNotFoundError
from src.database.queries import (
    get_booked_slots,
    get_patient_by_name,
    get_patient_data,
    get_provider_by_name,
    get_provider_data,
    get_provider_schedule,
    get_slot_statistics,
)
from src.features.slot_feature_builder import build_slots_feature_dataframe, get_time_of_day
from src.models.inference import SlotInferenceEngine
from src.recommendation.slot_ranker import aggregate_recommendations, rank_slots
from src.scheduling.slot_generator import generate_candidate_slots
from src.utils.logger import get_logger, log_prediction

logger = get_logger(__name__)


def _load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    p = Path(config_path or "configs/config.yaml")
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class AppointmentRecommender:
    def __init__(self, config_path: Optional[str] = None):
        self.config    = _load_config(config_path)
        self.engine    = SlotInferenceEngine(self.config["model_path"])
        self._slot_cfg = self.config["slot_recommendation"]
        self._rank_cfg = self.config.get("ranking", {})

    def recommend_slots(
        self,
        request_text: str,
        patient_data: Dict[str, Any],
        provider_data: Dict[str, Any],
        top_k: Optional[int] = None,
        db: Optional[Database] = None,
    ) -> List[Dict[str, Any]]:
        params = parse_appointment_request(request_text)
        logger.info("NLP parsed: %s", params)

        # -- Resolve patient MongoDB _id
        patient_id: Optional[str] = patient_data.get("patient_id") or None
        if not patient_id and db is not None:
            patient_name = patient_data.get("patient_name") or params.get("patient_name")
            if patient_name:
                patient_id = get_patient_by_name(db, patient_name)

        # -- Resolve provider MongoDB _id
        provider_id: Optional[str] = provider_data.get("provider_id") or None
        if not provider_id and db is not None:
            nlp_provider_num = params.get("provider_encoded")
            if nlp_provider_num is not None:
                provider_id = _lookup_provider_id_by_encoded(db, int(nlp_provider_num))
            if not provider_id:
                provider_name = provider_data.get("provider_name") or params.get("provider_name")
                if provider_name:
                    provider_id = get_provider_by_name(db, provider_name)

        if db is not None:
            if not patient_id:
                raise EntityNotFoundError("patient", patient_data.get("patient_name") or params.get("patient_name") or "unknown")
            if not provider_id:
                raise EntityNotFoundError("provider", provider_data.get("provider_name") or params.get("provider_name") or "unknown")

        logger.info("Resolved patient_id=%s provider_id=%s", patient_id, provider_id)

        # -- DB enrichment: fetch full feature dicts by _id
        if db is not None:
            if patient_id:
                db_patient = get_patient_data(db, patient_id, provider_id=provider_id)
                patient_data = {**db_patient, **patient_data}
            if provider_id:
                db_provider = get_provider_data(db, provider_id)
                provider_data = {**db_provider, **provider_data}

        if patient_id:
            patient_data["patient_id"] = patient_id
        if provider_id:
            provider_data["provider_id"] = provider_id

        # -- Date window
        now = datetime.utcnow().date()
        if params.get("date"):
            try:
                target     = datetime.fromisoformat(params["date"]).date()
                start_date = target
                end_date   = target + timedelta(days=self._slot_cfg.get("search_days", 14))
            except ValueError:
                start_date = now + timedelta(days=self._slot_cfg.get("search_start_days", 1))
                end_date   = start_date + timedelta(days=self._slot_cfg.get("search_days", 14))
        else:
            start_date = now + timedelta(days=self._slot_cfg.get("search_start_days", 1))
            end_date   = start_date + timedelta(days=self._slot_cfg.get("search_days", 14))

        # -- Blocked dates + booked slots from DB
        blocked_dates: List[str] = []
        booked_by_date: Dict[str, List[int]] = {}

        if db is not None and provider_id:
            blocked_dates = get_provider_schedule(db, provider_id)
            current = start_date
            while current <= end_date:
                date_iso = current.isoformat()
                booked   = get_booked_slots(db, provider_id, date_iso)
                if booked:
                    booked_by_date[date_iso] = booked
                current += timedelta(days=1)

        # -- Generate candidate slots
        provider_availability = {
            "provider_encoded": provider_data.get("provider_encoded"),
            "working_days":     provider_data.get("working_days", list(range(5))),
            "hours":            provider_data.get("hours", self._slot_cfg.get("working_hours", {"start": 8, "end": 17})),
        }

        slots = generate_candidate_slots(
            start_date=datetime.combine(start_date, datetime.min.time()),
            end_date=datetime.combine(end_date, datetime.min.time()),
            provider_availability=provider_availability,
            preferred_time_of_day=params.get("preferred_time"),
            slot_duration_minutes=self._slot_cfg.get("slot_duration_minutes", 60),
            slot_step_minutes=self._slot_cfg.get("slot_step_minutes", 60),
            working_hours=self._slot_cfg.get("working_hours"),
            blocked_dates=blocked_dates,
            booked_slots_by_date=booked_by_date,
        )

        if params.get("weekday") is not None:
            slots = [s for s in slots if s["weekday"] == params["weekday"]]

        if not slots:
            logger.warning("No candidate slots for: %s", request_text)
            return []

        # -- Enrich slots with DB statistics + per-slot signals
        if db is not None and provider_id:
            pv_hourly_curve = provider_data.get("provider_hourly_util_curve", {}) or {}
            pv_peak_set     = (
                provider_data.get("provider_peak_hour_set", [])
                or provider_data.get("provider_peak_hours", [])
            )
            pv_overbook = provider_data.get(
                "provider_overbooking_ratio",
                min(1.0, provider_data.get("provider_total_appts", 0) /
                    max(1.0, provider_data.get("max_daily_slots", 16) * 7.0)),
            )
            for slot in slots:
                stats     = get_slot_statistics(db, provider_id, slot["weekday"], slot["hour"])
                slot_hour = slot["hour"]
                slot["slot_historical_success_rate"] = stats["success_rate"]
                slot["slot_popularity_score"]        = stats["popularity_score"]
                slot["slot_demand_count"]            = stats["total_count"]
                slot["slot_days_ahead"]              = max(
                    0, (datetime.fromisoformat(slot["date"]).date() - now).days
                )
                slot["provider_overbooking_ratio"]   = pv_overbook
                # These three vary per slot hour:
                slot["patient_time_preference_match"] = (
                    1.0 if get_time_of_day(slot_hour) == patient_data.get("patient_preferred_time", "") else 0.0
                )
                slot["provider_peak_hour_score"] = 1.0 if slot_hour in pv_peak_set else 0.0
                slot["provider_hour_load"]       = float(pv_hourly_curve.get(slot_hour, 0.0))

        # -- Build features + predict
        feature_df    = build_slots_feature_dataframe(
            slots, patient_data, provider_data, self.engine.feature_columns
        )
        probabilities = self.engine.predict_proba(feature_df)

        results: List[Dict[str, Any]] = []
        for i, slot in enumerate(slots):
            results.append({
                "date":                       slot["date"],
                "time":                       f"{slot['hour']:02d}:00",
                "hour":                       slot["hour"],
                "weekday":                    slot["weekday"],
                "prob":                       round(float(probabilities[i][1]), 4),
                "patient_id":                 patient_id,
                "provider_id":                provider_id,
                "provider_encoded":           provider_data.get("provider_encoded"),
                "provider_7day_util":         provider_data.get("provider_7day_util", 0.5),
                "provider_avg_daily_appointments": provider_data.get("provider_avg_daily_appointments", 3.0),
                "slot_popularity_score":      slot.get("slot_popularity_score", 0.0),
                "slot_demand_count":          slot.get("slot_demand_count", 0.0),
                "patient_preference_match":   slot.get("patient_time_preference_match", 0.0),
                "provider_overbooking_ratio": slot.get("provider_overbooking_ratio", 0.0),
                "slot_quality_score":         slot.get("slot_quality_score", 0.0),
                "provider_peak_hour_score":   slot.get("provider_peak_hour_score", 0.0),
                "provider_hour_load":         slot.get("provider_hour_load", 0.0),
            })

        # -- Rank
        top_k_val = top_k or self._slot_cfg.get("top_k", 5)
        ranked = rank_slots(
            candidates=results,
            top_k=top_k_val * 3,
            cost_fn=self._slot_cfg.get("cost_fn", 1000),
            cost_fp=self._slot_cfg.get("cost_fp", 200),
            min_probability=self._slot_cfg.get("min_probability", 0.0),
            preferred_time=params.get("preferred_time"),
            ranking_weights=self._rank_cfg.get("weights"),
        )

        unique_per_day = self._rank_cfg.get("unique_per_day", False)
        final = aggregate_recommendations(ranked, top_n=top_k_val, unique_per_day=unique_per_day)

        log_prediction(logger, patient_id, provider_id, len(final))
        return final


# -- Lookup helpers

def _lookup_provider_id_by_encoded(db: Database, provider_encoded: int) -> Optional[str]:
    """Find provider _id by numeric provider_encoded field."""
    from src.database.db_connection import get_collection_names
    coll = db[get_collection_names()["providers"]]
    doc  = coll.find_one({"$or": [
        {"provider_encoded": provider_encoded},
        {"providerId":       provider_encoded},
    ]})
    if doc:
        _id = str(doc["_id"])
        logger.info("[provider_lookup] by_encoded=%s  _id=%s", provider_encoded, _id)
        return _id
    return None


def _best_provider_match(candidates: list, tokens: List[str]) -> Optional[dict]:
    """Pick candidate with most token hits across firstName + lastName."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    def hit_count(doc: dict) -> int:
        combined = f"{doc.get('firstName', '')} {doc.get('lastName', '')}".lower()
        return sum(1 for t in tokens if t.lower() in combined)

    best  = max(candidates, key=hit_count)
    score = hit_count(best)
    return best if score > 0 else None
