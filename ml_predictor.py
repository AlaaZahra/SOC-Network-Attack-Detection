from pathlib import Path
import argparse

import joblib
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_DIR / "Models" / "network_detector.joblib"


class NetworkPredictor:
    def __init__(self, model_path=DEFAULT_MODEL):
        artifact = joblib.load(model_path)

        self.pipeline = artifact["pipeline"]
        self.feature_columns = artifact["feature_columns"]
        self.threshold = artifact["decision_threshold"]

        classes = list(
            self.pipeline.named_steps["classifier"].classes_
        )
        self.attack_index = classes.index(1)

    def predict(self, data):
        data = data.copy()
        data.columns = data.columns.str.strip()

        if data.columns.duplicated().any():
            raise ValueError("Duplicate column names found.")

        missing = [
            column for column in self.feature_columns
            if column not in data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required network features: {missing}"
            )

        features = data[self.feature_columns].apply(
            pd.to_numeric, errors="raise"
        )
        features = features.replace([np.inf, -np.inf], np.nan)

        scores = self.pipeline.predict_proba(features)[
            :, self.attack_index
        ]

        # Preserve input fields for dashboard display.
        result = data.copy()
        result["ml_prediction"] = np.where(
            scores >= self.threshold, "ATTACK", "BENIGN"
        )

        # This model score has not been calibrated.
        result["ml_attack_score"] = scores
        result["ml_threshold"] = self.threshold

        return result


def main():
    parser = argparse.ArgumentParser(
        description="Classify network flows using the saved model."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different.")

    predictor = NetworkPredictor(args.model)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    attacks = 0
    first_chunk = True

    # Process large inputs without loading the entire file.
    for chunk in pd.read_csv(
        input_path,
        chunksize=10_000,
        encoding="utf-8-sig",
    ):
        if chunk.empty:
            continue

        result = predictor.predict(chunk)

        result.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )

        total += len(result)
        attacks += int(result["ml_prediction"].eq("ATTACK").sum())
        first_chunk = False

    if first_chunk:
        raise ValueError("The input contains no data rows.")

    print(f"Processed flows: {total:,}")
    print(f"Predicted ATTACK: {attacks:,}")
    print(f"Predicted BENIGN: {total - attacks:,}")
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()