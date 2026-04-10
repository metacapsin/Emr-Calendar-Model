"""CLI: build feature DataFrame from raw slot + patient + provider data."""
import argparse
import json

import joblib
import pandas as pd

from src.features.slot_feature_builder import build_slots_feature_dataframe


def main():
    parser = argparse.ArgumentParser(description="Build feature DataFrame for slots")
    parser.add_argument("--model", default="models/slot_prediction_model.pkl")
    parser.add_argument("--slots", required=True, help="JSON file with slot list")
    parser.add_argument("--patient", required=True, help="JSON file with patient dict")
    parser.add_argument("--provider", required=True, help="JSON file with provider dict")
    parser.add_argument("--output", default="features.csv")
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    feature_columns = bundle["feature_columns"]

    with open(args.slots) as f:
        slots = json.load(f)
    with open(args.patient) as f:
        patient = json.load(f)
    with open(args.provider) as f:
        provider = json.load(f)

    df = build_slots_feature_dataframe(slots, patient, provider, feature_columns)
    df.to_csv(args.output, index=False)
    print(f"Features written to {args.output} — shape: {df.shape}")


if __name__ == "__main__":
    main()
