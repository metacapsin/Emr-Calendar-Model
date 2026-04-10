"""CLI pipeline: generate candidate slots for a provider and print them."""
import argparse
import json
from datetime import datetime, timedelta

from src.scheduling.slot_generator import generate_candidate_slots


def main():
    parser = argparse.ArgumentParser(description="Generate candidate slots for a provider")
    parser.add_argument("--provider", type=int, default=1)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--time", type=str, default=None, help="morning/afternoon/evening")
    args = parser.parse_args()

    start = datetime.utcnow()
    end = start + timedelta(days=args.days)

    slots = generate_candidate_slots(
        start_date=start,
        end_date=end,
        provider_availability={"provider_encoded": args.provider, "working_days": list(range(5)), "hours": {"start": 8, "end": 17}},
        preferred_time_of_day=args.time,
    )
    print(json.dumps(slots, indent=2))
    print(f"\nTotal slots: {len(slots)}")


if __name__ == "__main__":
    main()
