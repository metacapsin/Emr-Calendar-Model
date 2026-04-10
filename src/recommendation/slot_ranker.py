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
    util = slot.get("provider_7day_util", 0.5)
    return max(0.0, float(util) - 0.8)   # penalty only above 80% utilization


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

    score = w_prob * probability
            + w_pref * preference_score
            - w_util * utilization_penalty
            - cost_adjusted_penalty
    """
    weights = ranking_weights or {"probability": 0.6, "preference_score": 0.25, "utilization_penalty": 0.15}

    scored: List[Dict[str, Any]] = []
    for candidate in candidates:
        prob = float(candidate.get("prob", 0.0))
        if prob < min_probability:
            continue

        pref = _preference_score(candidate, preferred_time)
        util_pen = _utilization_penalty(candidate)

        # Cost-adjusted component (normalized to 0–1)
        fn_cost = (1.0 - prob) * cost_fn
        fp_cost = prob * cost_fp
        cost_component = (fn_cost + fp_cost) / (cost_fn + cost_fp)

        score = (
            weights.get("probability", 0.6) * prob
            + weights.get("preference_score", 0.25) * pref
            - weights.get("utilization_penalty", 0.15) * util_pen
            - 0.1 * cost_component
        )

        entry = candidate.copy()
        entry["score"] = round(score, 6)
        entry["preference_score"] = pref
        entry["utilization_penalty"] = util_pen
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
