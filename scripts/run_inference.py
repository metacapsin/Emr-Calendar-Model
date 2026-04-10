"""CLI: run batch inference on a CSV of feature rows."""
import argparse
import json

import pandas as pd

from src.models.inference import SlotInferenceEngine


def main():
    parser = argparse.ArgumentParser(description="Run batch slot inference")
    parser.add_argument("--model", default="models/slot_prediction_model.pkl")
    parser.add_argument("--input", required=True, help="CSV file with feature rows")
    parser.add_argument("--output", default="predictions.json")
    args = parser.parse_args()

    engine = SlotInferenceEngine(args.model)
    df = pd.read_csv(args.input)
    results = engine.batch_predict(df.to_dict(orient="records"))

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Predictions written to {args.output}")


if __name__ == "__main__":
    main()
