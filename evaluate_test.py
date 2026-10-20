from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "Data" / "ml"
MODEL_PATH = PROJECT_DIR / "Models" / "network_detector.joblib"

print("Loading saved model and test data...", flush=True)

artifact = joblib.load(MODEL_PATH)
pipeline = artifact["pipeline"]
feature_columns = artifact["feature_columns"]
threshold = artifact["decision_threshold"]

test = pd.read_csv(DATA_DIR / "test.csv")

X_test = test[feature_columns].replace(
    [np.inf, -np.inf], np.nan
)
y_test = test["target"].astype("int8")

# Use the saved preprocessing without fitting it again.
attack_index = list(
    pipeline.named_steps["classifier"].classes_
).index(1)

probabilities = pipeline.predict_proba(X_test)[:, attack_index]
predictions = (probabilities >= threshold).astype("int8")

print("\n========== TEST RESULTS ==========")
print(f"Decision threshold: {threshold}")

print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["BENIGN", "ATTACK"],
    digits=4,
    zero_division=0,
))

tn, fp, fn, tp = confusion_matrix(
    y_test, predictions, labels=[0, 1]
).ravel()

print(f"True Negatives:  {tn:,}")
print(f"False Positives: {fp:,}")
print(f"False Negatives: {fn:,}")
print(f"True Positives:  {tp:,}")

fpr = fp / (fp + tn) if fp + tn else 0
recall = tp / (tp + fn) if tp + fn else 0

print(f"\nFalse Positive Rate: {fpr:.4%}")
print(f"Attack Recall: {recall:.4%}")
print(
    "Average Precision: "
    f"{average_precision_score(y_test, probabilities):.4f}"
)

attack_mask = y_test.to_numpy() == 1
coverage = test.loc[attack_mask, ["attack_type"]].copy()
coverage["detected"] = predictions[attack_mask]

summary = coverage.groupby("attack_type").agg(
    examples=("detected", "size"),
    detected=("detected", "sum"),
)
summary["recall"] = summary["detected"] / summary["examples"]

print("\nTest recall by attack type:")
print(summary.to_string())

error_mask = predictions != y_test.to_numpy()
errors = test.loc[error_mask].copy()
errors["predicted_target"] = predictions[error_mask]
errors["attack_probability"] = probabilities[error_mask]

errors.to_csv(DATA_DIR / "test_errors.csv", index=False)

print("\nEvaluation completed. No training was performed.")