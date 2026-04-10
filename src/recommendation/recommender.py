from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from sqlalchemy.orm import Session

from src.api.nlp_parser import parse_appointment_request
from src.database.queries import (
    get_booked_slots,
    get_patient_data,
    get_provider_data,
    get_provider_schedule,
    get_slot_statistics,
)
from src.features.slot_feature_builder import build_slots_feature_dataframe
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
        self.config = _load_config(config_path)
        self.engine = SlotInferenceEngine(self.config["model_path"])
        self._slot_cfg = self.config["slot_recommendation"]
        self._rank_cfg = self.config.get("ranking", {})

    # ── Public API ─────────────────────────────────────────────────────────────

    def recommend_slots(
        self,
        request_text: str,
        patient_data: Dict[str, Any],
        provider_data: Dict[str, Any],
        top_k: Optional[int] = None,
        db: Optional[Session] = None,
    ) -> List[Dict[str, Any]]:
        """Full recommendation pipeline.

        If `db` is provided, patient/provider data is enriched from the database.
        Otherwise, the passed dicts are used directly (useful for testing).
        """
        params = parse_appointment_request(request_text)
        logger.info("NLP parsed: %s", params)

        # ── Resolve provider_encoded from NLP if not in provider_data ──────────
        provider_encoded = (
            params.get("provider_encoded")
            or provider_data.get("provider_encoded")
        )
        patient_encoded = patient_data.get("patient_encoded")

        # ── DB enrichment ──────────────────────────────────────────────────────
        if db is not None:
            if patient_encoded:
                db_patient = get_patient_data(db, patient_encoded)
                patient_data = {**db_patient, **patient_data}  # caller overrides DB
            if provider_encoded:
                db_provider = get_provider_data(db, provider_encoded)
                provider_data = {**db_provider, **provider_data}

        # Ensure provider_encoded is set
        if provider_encoded:
            provider_data["provider_encoded"] = provider_encoded

        # ── Date window ────────────────────────────────────────────────────────
        now = datetime.utcnow().date()
        if params.get("date"):
            try:
                target = datetime.fromisoformat(params["date"]).date()
                start_date = target
                end_date = target + timedelta(days=self._slot_cfg.get("search_days", 14))
            except ValueError:
                start_date = now + timedelta(days=self._slot_cfg.get("search_start_days", 1))
                end_date = start_date + timedelta(days=self._slot_cfg.get("search_days", 14))
        else:
            start_date = now + timedelta(days=self._slot_cfg.get("search_start_days", 1))
            end_date = start_date + timedelta(days=self._slot_cfg.get("search_days", 14))

        # ── Blocked dates + booked slots from DB ───────────────────────────────
        blocked_dates: List[str] = []
        booked_by_date: Dict[str, List[int]] = {}

        if db is not None and provider_encoded:
            blocked_dates = get_provider_schedule(db, provider_encoded)
            current = start_date
            while current <= end_date:
                date_iso = current.isoformat()
                booked = get_booked_slots(db, provider_encoded, date_iso)
                if booked:
                    booked_by_date[date_iso] = booked
                current += timedelta(days=1)

        # ── Generate candidate slots ───────────────────────────────────────────
        provider_availability = {
            "provider_encoded": provider_encoded,
            "working_days": provider_data.get("working_days", list(range(5))),
            "hours": provider_data.get("hours", self._slot_cfg.get("working_hours", {"start": 8, "end": 17})),
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

        # ── Weekday filter from NLP ────────────────────────────────────────────
        if params.get("weekday") is not None:
            slots = [s for s in slots if s["weekday"] == params["weekday"]]

        if not slots:
            logger.warning("No candidate slots for: %s", request_text)
            return []

        # ── Enrich slots with DB statistics ───────────────────────────────────
        if db is not None and provider_encoded:
            for slot in slots:
                stats = get_slot_statistics(db, provider_encoded, slot["weekday"], slot["hour"])
                slot["slot_historical_success_rate"] = stats["success_rate"]
                slot["slot_popularity_score"] = stats["popularity_score"]
                slot["slot_demand_count"] = stats["total_count"]

        # ── Build features + predict ───────────────────────────────────────────
        feature_df = build_slots_feature_dataframe(
            slots, patient_data, provider_data, self.engine.feature_columns
        )
        probabilities = self.engine.predict_proba(feature_df)

        # ── Assemble results ───────────────────────────────────────────────────
        results: List[Dict[str, Any]] = []
        for i, slot in enumerate(slots):
            results.append(
                {
                    "date": slot["date"],
                    "time": f"{slot['hour']:02d}:00",
                    "hour": slot["hour"],
                    "weekday": slot["weekday"],
                    "prob": round(float(probabilities[i][1]), 4),
                    "provider_encoded": slot.get("provider_encoded"),
                    "provider_7day_util": provider_data.get("provider_7day_util", 0.5),
                    "slot_popularity_score": slot.get("slot_popularity_score", 0.0),
                }
            )

        # ── Rank ───────────────────────────────────────────────────────────────
        top_k_val = top_k or self._slot_cfg.get("top_k", 5)
        ranked = rank_slots(
            candidates=results,
            top_k=top_k_val * 3,  # over-fetch before dedup
            cost_fn=self._slot_cfg.get("cost_fn", 1000),
            cost_fp=self._slot_cfg.get("cost_fp", 200),
            min_probability=self._slot_cfg.get("min_probability", 0.0),
            preferred_time=params.get("preferred_time"),
            ranking_weights=self._rank_cfg.get("weights"),
        )

        unique_per_day = self._rank_cfg.get("unique_per_day", False)
        final = aggregate_recommendations(ranked, top_n=top_k_val, unique_per_day=unique_per_day)

        log_prediction(logger, patient_encoded, provider_encoded, len(final))
        return final
