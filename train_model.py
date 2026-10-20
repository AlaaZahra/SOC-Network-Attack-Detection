from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "Data" / "ml"
MODEL_DIR = PROJECT_DIR / "Models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

METADATA_COLUMNS = ["target", "attack_type", "source_file"]
RANDOM_STATE = 42

print("Loading training and validation data...", flush=True)

train = pd.read_csv(DATA_DIR / "train.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")

feature_columns = [
    column for column in train.columns
    if column not in METADATA_COLUMNS
]

X_train = train[feature_columns].replace(
    [np.inf, -np.inf], np.nan
)
y_train = train["target"].astype("int8")

X_validation = validation[feature_columns].replace(
    [np.inf, -np.inf], np.nan
)
y_validation = validation["target"].astype("int8")

# Identify unusable columns using training data only.
usable_features = [
    column for column in feature_columns
    if X_train[column].nunique(dropna=True) > 1
]

print(f"Original features: {len(feature_columns)}")
print(f"Usable features: {len(usable_features)}")

X_train = X_train[usable_features]
X_validation = X_validation[usable_features]

# Fit the imputer only on the training split.
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=2,
            max_samples=0.8,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=2,
        ),
    ),
])

print("\nTraining Random Forest...", flush=True)
pipeline.fit(X_train, y_train)

print("Evaluating validation data...", flush=True)

predictions = pipeline.predict(X_validation)
probabilities = pipeline.predict_proba(X_validation)[:, 1]

print("\n========== VALIDATION RESULTS ==========")

print(classification_report(
    y_validation,
    predictions,
    labels=[0, 1],
    target_names=["BENIGN", "ATTACK"],
    digits=4,
    zero_division=0,
))

tn, fp, fn, tp = confusion_matrix(
    y_validation, predictions, labels=[0, 1]
).ravel()

print(f"True Negatives:  {tn:,}")
print(f"False Positives: {fp:,}")
print(f"False Negatives: {fn:,}")
print(f"True Positives:  {tp:,}")

false_positive_rate = fp / (fp + tn) if fp + tn else 0
attack_recall = tp / (tp + fn) if tp + fn else 0

print(f"\nFalse Positive Rate: {false_positive_rate:.4%}")
print(f"Attack Recall: {attack_recall:.4%}")
print(
    "Average Precision: "
    f"{average_precision_score(y_validation, probabilities):.4f}"
)

# Report attack detection coverage without using attack names as inputs.
attack_results = validation.loc[
    y_validation.eq(1), ["attack_type"]
].copy()

attack_results["detected"] = predictions[
    y_validation.to_numpy() == 1
]

coverage = attack_results.groupby("attack_type").agg(
    examples=("detected", "size"),
    detected=("detected", "sum"),
)

coverage["recall"] = (
    coverage["detected"] / coverage["examples"]
)

print("\nValidation recall by attack type:")
print(coverage.to_string())

# Store preprocessing and the feature schema with the classifier.
model_path = MODEL_DIR / "network_detector.joblib"

joblib.dump(
    {
        "pipeline": pipeline,
        "feature_columns": usable_features,
        "class_mapping": {0: "BENIGN", 1: "ATTACK"},
        "decision_threshold": 0.5,
    },
    model_path,
)

print(f"\nModel saved to: {model_path}")
print("The test split has not been evaluated.")

# Export validation errors for analysis.
error_mask = predictions != y_validation.to_numpy()

errors = validation.loc[error_mask].copy()
errors["predicted_target"] = predictions[error_mask]
errors["attack_probability"] = probabilities[error_mask]

errors["error_type"] = np.where(
    errors["target"].eq(0),
    "False Positive",
    "False Negative",
)

error_path = DATA_DIR / "validation_errors.csv"
errors.to_csv(error_path, index=False)

print("\nValidation errors by type and original label:")
print(
    errors.groupby(["error_type", "attack_type"])
    .size()
    .rename("count")
    .to_string()
)

print(f"\nErrors saved to: {error_path}")