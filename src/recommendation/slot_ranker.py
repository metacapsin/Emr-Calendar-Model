from typing import Any, Dict, List, Optional


def rank_slots(
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
    cost_fn: float = 1000.0,
    cost_fp: float = 200.0,
    min_probability: float = 0.0,
) -> List[Dict[str, Any]]:
    """Sort candidate slots by model probability and apply cost adjustment.

    Each candidate entry must include 'prob' (success probability) and slot metadata.
    """
    scored: List[Dict[str, Any]] = []

    for candidate in candidates:
        prob = float(candidate.get('prob', 0.0))
        fn_cost = (1.0 - prob) * cost_fn
        fp_cost = prob * cost_fp
        candidate_score = prob - (fn_cost + fp_cost) / (cost_fn + cost_fp)

        candidate_entry = candidate.copy()
        candidate_entry['sorted_score'] = candidate_score
        candidate_entry['cost_fn'] = fn_cost
        candidate_entry['cost_fp'] = fp_cost

        scored.append(candidate_entry)

    filtered = [c for c in scored if c.get('prob', 0.0) >= min_probability]
    ranked = sorted(filtered, key=lambda item: (item['sorted_score'], item['prob']), reverse=True)
    return ranked[:top_k]


def aggregate_recommendations(
    results: List[Dict[str, Any]],
    top_n: int = 3,
    unique_per_day: bool = True,
) -> List[Dict[str, Any]]:
    """Return top N final recommended slots with business constraints."""
    if not unique_per_day:
        return results[:top_n]

    output: List[Dict[str, Any]] = []
    seen_dates = set()
    for item in results:
        date = item.get('date')
        if date in seen_dates:
            continue
        output.append(item)
        seen_dates.add(date)
        if len(output) >= top_n:
            break

    return output
