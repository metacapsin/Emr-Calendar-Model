from typing import Any, Dict, List, Optional


def _preference_score(slot: Dict[str, Any], preferred_time: Optional[str]) -> float:
    """+1.0 if slot matches preferred time of day, else 0."""
    if not preferred_time:
        return 0.0
    hour = slot.get("hour", 0)
    mapping = {"morning": range(8, 11), "midday": range(11, 13), "afternoon": range(13, 16), "evening": range(16, 19)}
    return 1.0 if hour in mapping.get(preferred_time, range(0)) else 0.0


def _utilization_penalty(slot: Dict[str, Any]) -> float:
    """Penalize over-utilized providers (0–1 scale, higher = more penalty)."""
    util = slot.get("provider_7day_util", slot.get("provider_utilization", 0.5))
    return max(0.0, float(util) - 0.7)


def _overbooking_risk(slot: Dict[str, Any]) -> float:
    risk = float(slot.get("provider_overbooking_ratio", 0.0))
    if "slot_demand_count" in slot and slot["slot_demand_count"] > 0:
        demand = float(slot.get("slot_demand_count", 0))
        risk = min(1.0, risk + min(demand / 20.0, 0.3))
    return min(1.0, max(0.0, risk))


def rank_slots(
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
    cost_fn: float = 1000.0,
    cost_fp: float = 200.0,
    min_probability: float = 0.0,
    preferred_time: Optional[str] = None,
    ranking_weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Multi-factor slot ranking.

    final_score = probability * w_prob
                  + preference_score * w_pref
                  + patient_preference_match * w_patient_pref
                  + slot_popularity_score * w_popularity
                  - utilization_penalty * w_util
                  - fn_cost_scale * (1 - probability)
                  - fp_cost_scale * probability * overbooking_risk
    """
    weights = {
        "probability": 0.6,
        "preference_score": 0.2,
        "patient_preference_match": 0.2,
        "slot_popularity_score": 0.1,
        "utilization_penalty": 0.1,
        "fn_cost_scale": 0.001,
        "fp_cost_scale": 0.001,
    }
    if ranking_weights:
        weights.update(ranking_weights)

    scored: List[Dict[str, Any]] = []
    for candidate in candidates:
        prob = float(candidate.get("prob", 0.0))
        if prob < min_probability:
            continue

        pref = _preference_score(candidate, preferred_time)
        patient_pref = float(candidate.get("patient_preference_match", 0.0))
        provider_util = float(candidate.get("provider_7day_util", candidate.get("provider_utilization", 0.0)))
        popularity = float(candidate.get("slot_popularity_score", 0.0))
        util_penalty = _utilization_penalty(candidate)
        overbooking_risk = _overbooking_risk(candidate)

        score = (
            prob * float(weights["probability"])
            + pref * float(weights["preference_score"])
            + patient_pref * float(weights["patient_preference_match"])
            + popularity * float(weights["slot_popularity_score"])
            - util_penalty * float(weights["utilization_penalty"])
            - float(weights["fn_cost_scale"]) * (1.0 - prob)
            - float(weights["fp_cost_scale"]) * prob * overbooking_risk
        )

        entry = candidate.copy()
        entry["score"] = round(score, 6)
        entry["preference_score"] = pref
        entry["patient_preference_match"] = patient_pref
        entry["provider_utilization"] = provider_util
        entry["slot_popularity_score"] = popularity
        entry["utilization_penalty"] = round(util_penalty, 6)
        entry["overbooking_risk"] = round(overbooking_risk, 6)
        scored.append(entry)

    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]


def aggregate_recommendations(
    results: List[Dict[str, Any]],
    top_n: int = 3,
    unique_per_day: bool = False,
) -> List[Dict[str, Any]]:
    """Return top-N slots, optionally enforcing one slot per day."""
    if not unique_per_day:
        return results[:top_n]

    output: List[Dict[str, Any]] = []
    seen: set = set()
    for item in results:
        date = item.get("date")
        if date in seen:
            continue
        output.append(item)
        seen.add(date)
        if len(output) >= top_n:
            break
    return output
