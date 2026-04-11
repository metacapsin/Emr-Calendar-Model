from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import re
import yaml
from pymongo.database import Database

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

        # ── Resolve patient MongoDB _id ────────────────────────────────────────
        # Priority: payload patient_id > DB name lookup
        patient_id: Optional[str] = patient_data.get("patient_id") or None
        if not patient_id and patient_data.get("patient_name") and db is not None:
            patient_id = _lookup_patient_id(db, patient_data["patient_name"])

        # ── Resolve provider MongoDB _id ───────────────────────────────────────
        # Priority: payload provider_id > NLP numeric > DB name lookup
        provider_id: Optional[str] = provider_data.get("provider_id") or None
        if not provider_id and db is not None:
            nlp_provider_name = params.get("provider_name")
            nlp_provider_num  = params.get("provider_encoded")
            if nlp_provider_num:
                provider_id = _lookup_provider_id_by_encoded(db, int(nlp_provider_num))
            if not provider_id and nlp_provider_name:
                provider_id = _lookup_provider_id(db, nlp_provider_name)

        logger.info("Resolved  patient_id=%s  provider_id=%s", patient_id, provider_id)

        # ── DB enrichment — fetch full feature dicts by _id ───────────────────
        if db is not None:
            if patient_id:
                db_patient  = get_patient_data(db, patient_id)
                patient_data = {**db_patient, **patient_data}
            if provider_id:
                db_provider  = get_provider_data(db, provider_id)
                provider_data = {**db_provider, **provider_data}

        # Ensure _id fields are set in dicts for downstream use
        if patient_id:
            patient_data["patient_id"] = patient_id
        if provider_id:
            provider_data["provider_id"] = provider_id

        # ── Date window ────────────────────────────────────────────────────────
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

        # ── Blocked dates + booked slots from DB ───────────────────────────────
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

        # ── Generate candidate slots ───────────────────────────────────────────
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

        # ── Enrich slots with DB statistics ───────────────────────────────────
        if db is not None and provider_id:
            for slot in slots:
                stats = get_slot_statistics(db, provider_id, slot["weekday"], slot["hour"])
                slot["slot_historical_success_rate"] = stats["success_rate"]
                slot["slot_popularity_score"]        = stats["popularity_score"]
                slot["slot_demand_count"]            = stats["total_count"]

        # ── Build features + predict ───────────────────────────────────────────
        feature_df    = build_slots_feature_dataframe(
            slots, patient_data, provider_data, self.engine.feature_columns
        )
        probabilities = self.engine.predict_proba(feature_df)

        results: List[Dict[str, Any]] = []
        for i, slot in enumerate(slots):
            results.append({
                "date":                  slot["date"],
                "time":                  f"{slot['hour']:02d}:00",
                "hour":                  slot["hour"],
                "weekday":               slot["weekday"],
                "prob":                  round(float(probabilities[i][1]), 4),
                "patient_id":            patient_id,
                "provider_id":           provider_id,
                "provider_encoded":      provider_data.get("provider_encoded"),
                "provider_7day_util":    provider_data.get("provider_7day_util", 0.5),
                "slot_popularity_score": slot.get("slot_popularity_score", 0.0),
            })

        # ── Rank ───────────────────────────────────────────────────────────────
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


# ── Name → MongoDB _id lookup helpers ─────────────────────────────────────────

def _normalize(name: str) -> str:
    return " ".join(name.strip().upper().split())


def _lookup_patient_id(db: Database, patient_name: str) -> Optional[str]:
    """
    Find patient MongoDB _id by name.
    DB: { _id: ObjectId, fullName: "EMMA GAMEZ" }
        OR nested: { _id: ObjectId, data: { fullName: "EMMA GAMEZ" } }

    Steps:
      1. Exact case-insensitive on data.fullName, then fullName
      2. All tokens AND on same fields
      3. First OR last token on same fields
    """
    from src.database.db_connection import get_collection_names
    coll   = db[get_collection_names()["patients"]]
    norm   = _normalize(patient_name)
    tokens = norm.split()

    logger.info("[patient_lookup] input='%s'  normalized='%s'  tokens=%s",
                patient_name, norm, tokens)

    doc: Optional[dict] = None

    # Step 1 — exact match on fullName (top-level)
    q   = {"fullName": {"$regex": f"^{re.escape(norm)}$", "$options": "i"}}
    doc = coll.find_one(q)
    logger.info("[patient_lookup] step1 exact fullName  found=%s", bool(doc))

    # Step 2 — all tokens AND on fullName
    if not doc and len(tokens) >= 2:
        q   = {"$and": [{"fullName": {"$regex": re.escape(t), "$options": "i"}} for t in tokens]}
        doc = coll.find_one(q)
        logger.info("[patient_lookup] step2 all-tokens fullName  found=%s", bool(doc))

    # Step 3 — firstName + lastName separate fields (patient-details schema)
    if not doc and len(tokens) >= 2:
        q   = {
            "firstName": {"$regex": re.escape(tokens[0]),  "$options": "i"},
            "lastName":  {"$regex": re.escape(tokens[-1]), "$options": "i"},
        }
        doc = coll.find_one(q)
        logger.info("[patient_lookup] step3 firstName+lastName  found=%s", bool(doc))

    # Step 4 — patientFirstName + patientLastName (patient-register schema fallback)
    if not doc and len(tokens) >= 2:
        q   = {
            "patientFirstName": {"$regex": re.escape(tokens[0]),  "$options": "i"},
            "patientLastName":  {"$regex": re.escape(tokens[-1]), "$options": "i"},
        }
        doc = coll.find_one(q)
        logger.info("[patient_lookup] step4 patientFirstName+patientLastName  found=%s", bool(doc))

    # Step 5 — any token on fullName OR firstName OR lastName
    if not doc:
        q   = {"$or": [
            {"fullName":  {"$regex": re.escape(tokens[0]),  "$options": "i"}},
            {"fullName":  {"$regex": re.escape(tokens[-1]), "$options": "i"}},
            {"firstName": {"$regex": re.escape(tokens[0]),  "$options": "i"}},
            {"lastName":  {"$regex": re.escape(tokens[-1]), "$options": "i"}},
        ]}
        doc = coll.find_one(q)
        logger.info("[patient_lookup] step5 partial  found=%s", bool(doc))

    if not doc:
        logger.warning("[patient_lookup] NO MATCH for '%s'", patient_name)
        return None

    _id           = str(doc["_id"])
    resolved_name = (
        doc.get("fullName")
        or f"{doc.get('firstName', '')} {doc.get('lastName', '')}".strip()
        or doc.get("data", {}).get("fullName", "?")
    )
    logger.info("[patient_lookup] MATCHED  name='%s'  _id=%s", resolved_name, _id)
    return _id


def _lookup_provider_id(db: Database, provider_name: str) -> Optional[str]:
    """
    Find provider MongoDB _id by name.
    DB: { _id: ObjectId, firstName: "Tam", lastName: "Bui, FNP BC" }

    Steps:
      1. firstName exact + lastName contains last token
      2. Any token in firstName OR lastName → best hit-count match
      3. First token only
      4. Last token only
    """
    from src.database.db_connection import get_collection_names
    coll    = db[get_collection_names()["providers"]]
    cleaned = re.sub(r"^(dr\.?|doctor)\s+", "", provider_name.strip(), flags=re.IGNORECASE)
    tokens  = [t for t in cleaned.split() if len(t) > 1]

    logger.info("[provider_lookup] input='%s'  cleaned='%s'  tokens=%s",
                provider_name, cleaned, tokens)

    if not tokens:
        logger.warning("[provider_lookup] no usable tokens from '%s'", provider_name)
        return None

    # Base filter: only match users with provider role
    role_filter = {"role": {"$in": ["provider"]}}
    doc: Optional[dict] = None

    # Step 1 — firstName exact + lastName contains last token (provider role only)
    if len(tokens) >= 2:
        q   = {
            **role_filter,
            "firstName": {"$regex": f"^{re.escape(tokens[0])}$", "$options": "i"},
            "lastName":  {"$regex": re.escape(tokens[-1]),        "$options": "i"},
        }
        doc = coll.find_one(q)
        logger.info("[provider_lookup] step1 firstName+lastName  found=%s", bool(doc))

    # Step 2 — any token in firstName OR lastName (provider role only) → best match
    if not doc:
        or_clauses = []
        for t in tokens:
            e = re.escape(t)
            or_clauses.append({"firstName": {"$regex": e, "$options": "i"}})
            or_clauses.append({"lastName":  {"$regex": e, "$options": "i"}})
        candidates = list(coll.find({"$and": [role_filter, {"$or": or_clauses}]}))
        logger.info("[provider_lookup] step2 any-token  candidates=%d", len(candidates))
        doc = _best_provider_match(candidates, tokens)
        logger.info("[provider_lookup] step2 best-match  found=%s", bool(doc))

    # Step 3 — first token only (provider role)
    if not doc:
        q   = {"$and": [role_filter, {"$or": [
            {"firstName": {"$regex": re.escape(tokens[0]), "$options": "i"}},
            {"lastName":  {"$regex": re.escape(tokens[0]), "$options": "i"}},
        ]}]}
        doc = coll.find_one(q)
        logger.info("[provider_lookup] step3 first-token  found=%s", bool(doc))

    # Step 4 — last token only (provider role)
    if not doc and len(tokens) > 1:
        q   = {"$and": [role_filter, {"$or": [
            {"firstName": {"$regex": re.escape(tokens[-1]), "$options": "i"}},
            {"lastName":  {"$regex": re.escape(tokens[-1]), "$options": "i"}},
        ]}]}
        doc = coll.find_one(q)
        logger.info("[provider_lookup] step4 last-token  found=%s", bool(doc))

    if not doc:
        logger.warning("[provider_lookup] NO MATCH for '%s'", provider_name)
        return None

    _id = str(doc["_id"])
    logger.info("[provider_lookup] MATCHED  firstName='%s'  lastName='%s'  _id=%s",
                doc.get("firstName"), doc.get("lastName"), _id)
    return _id


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
    logger.info("[provider_lookup] best-match  firstName='%s'  lastName='%s'  score=%d/%d",
                best.get("firstName"), best.get("lastName"), score, len(tokens))
    return best if score > 0 else None
