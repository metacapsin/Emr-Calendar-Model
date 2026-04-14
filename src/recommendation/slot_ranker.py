"""
Slot Ranker — Cost-Aware, Sensitivity-Preserving Ranking
=========================================================
Ranking formula:
    utility = prob * w_prob
              + preference_score * w_pref
              + patient_preference_match * w_patient_pref
              + slot_popularity_score * w_popularity
              - utilization_penalty * w_util
              - fn_cost_term          (missed patient cost)
              - fp_cost_term          (overbooking cost)

FN/FP costs are normalized to [0,1] scale so they meaningfully
compete with probability in the final score.
"""
from typing import Any, Dict, List, Optional


def _preference_score(slot: Dict[str, Any], preferred_time: Optional[str]) -> float:
    if not preferred_time:
        return 0.0
    hour = slot.get("hour", 0)
    mapping = {
        "morning":   range(8, 11),
        "midday":    range(11, 13),
        "afternoon": range(13, 16),
        "evening":   range(16, 19),
    }
    return 1.0 if hour in mapping.get(preferred_time, range(0)) else 0.0


def _utilization_penalty(slot: Dict[str, Any]) -> float:
    """Penalize over-utilized providers. Penalty starts at 70% utilization."""
    util = float(slot.get("provider_7day_util", slot.get("provider_utilization", 0.5)))
    return max(0.0, util - 0.7) / 0.3  # normalized: 0 at 70%, 1.0 at 100%


def _overbooking_risk(slot: Dict[str, Any]) -> float:
    risk = float(slot.get("provider_overbooking_ratio", 0.0))
    demand = float(slot.get("slot_demand_count", 0.0))
    avg_daily = max(1.0, float(slot.get("provider_avg_daily_appointments", 3.0)))
    demand_pressure = min(1.0, demand / (avg_daily * 5.0))
    return min(1.0, max(0.0, risk * 0.7 + demand_pressure * 0.3))


def _cost_adjusted_utility(
    prob: float,
    overbooking_risk: float,
    cost_fn: float,
    cost_fp: float,
) -> float:
    """
    Expected cost-adjusted utility per slot.
    Normalized so costs are on the same scale as probability (0–1).

    Expected FN cost (missed patient): cost_fn * (1 - prob)
    Expected FP cost (overbooking):    cost_fp * prob * overbooking_risk

    We convert to a utility gain by subtracting normalized costs from 1.0.
    """
    total_cost_scale = cost_fn + cost_fp  # normalization denominator
    fn_term = (cost_fn / total_cost_scale) * (1.0 - prob)
    fp_term = (cost_fp / total_cost_scale) * prob * overbooking_risk
    return max(0.0, 1.0 - fn_term - fp_term)


def rank_slots(
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
    cost_fn: float = 1000.0,
    cost_fp: float = 200.0,
    min_probability: float = 0.0,
    preferred_time: Optional[str] = None,
    ranking_weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Multi-factor slot ranking with meaningful cost integration.

    final_score = w_prob     * prob
                + w_cost     * cost_adjusted_utility
                + w_pref     * preference_score
                + w_pat_pref * patient_preference_match
                + w_pop      * slot_popularity_score
                - w_util     * utilization_penalty
    """
    weights = {
        "probability":             0.40,
        "cost_utility":            0.30,  # FN/FP cost function — now meaningful
        "preference_score":        0.10,
        "patient_preference_match": 0.10,
        "slot_popularity_score":   0.05,
        "utilization_penalty":     0.05,
    }
    if ranking_weights:
        weights.update(ranking_weights)

    scored: List[Dict[str, Any]] = []
    for candidate in candidates:
        prob = float(candidate.get("prob", 0.0))
        if prob < min_probability:
            continue

        pref          = _preference_score(candidate, preferred_time)
        patient_pref  = float(candidate.get("patient_preference_match", 0.0))
        popularity    = float(candidate.get("slot_popularity_score", 0.0))
        util_penalty  = _utilization_penalty(candidate)
        ob_risk       = _overbooking_risk(candidate)
        cost_utility  = _cost_adjusted_utility(prob, ob_risk, cost_fn, cost_fp)

        score = (
            prob         * float(weights["probability"])
            + cost_utility * float(weights["cost_utility"])
            + pref         * float(weights["preference_score"])
            + patient_pref * float(weights["patient_preference_match"])
            + popularity   * float(weights["slot_popularity_score"])
            - util_penalty * float(weights["utilization_penalty"])
        )

        entry = candidate.copy()
        entry["score"]                = round(score, 6)
        entry["cost_utility"]         = round(cost_utility, 4)
        entry["preference_score"]     = pref
        entry["patient_preference_match"] = patient_pref
        entry["utilization_penalty"]  = round(util_penalty, 4)
        entry["overbooking_risk"]     = round(ob_risk, 4)
        scored.append(entry)

    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]


def aggregate_recommendations(
    results: List[Dict[str, Any]],
    top_n: int = 3,
    unique_per_day: bool = False,
) -> List[Dict[str, Any]]:
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
