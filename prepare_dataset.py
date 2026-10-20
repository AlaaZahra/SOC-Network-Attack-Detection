from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(
    r"D:\digilians\cyberSecurity\ACUD Train\Project"
    r"\MachineLearningCSV\MachineLearningCVE"
)

OUTPUT_DIR = Path(__file__).resolve().parent / "Data" / "ml"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
CHUNK_SIZE = 50_000
SAMPLE_FRACTION = 0.106
RARE_CLASS_LIMIT = 2_000

rng = np.random.default_rng(RANDOM_STATE)
files = sorted(DATA_DIR.glob("*.csv"))

if not files:
    raise SystemExit(f"No CSV files found in: {DATA_DIR}")

# First pass: count original labels across all files.
label_counts = {}

print("Counting labels...", flush=True)

for file in files:
    for chunk in pd.read_csv(
        file,
        usecols=lambda name: name.strip() == "Label",
        chunksize=CHUNK_SIZE,
        encoding="utf-8-sig",
    ):
        labels = chunk.iloc[:, 0].astype("string").str.strip()

        for label, count in labels.value_counts().items():
            label_counts[label] = label_counts.get(label, 0) + int(count)

rare_labels = {
    label
    for label, count in label_counts.items()
    if count <= RARE_CLASS_LIMIT
}

# Second pass: sample common classes and retain rare classes.
parts = []
feature_columns = None
invalid_labels = 0

print("Sampling data...", flush=True)

for file in files:
    print(f"Reading: {file.name}", flush=True)

    for chunk in pd.read_csv(
        file,
        chunksize=CHUNK_SIZE,
        encoding="utf-8-sig",
        low_memory=False,
    ):
        chunk.columns = chunk.columns.str.strip()

        labels = chunk["Label"].astype("string").str.strip()
        valid = labels.notna() & labels.ne("").fillna(False)
        invalid_labels += int((~valid).sum())

        chunk = chunk.loc[valid].copy()
        labels = labels.loc[valid]

        keep = (
            labels.isin(rare_labels).to_numpy()
            | (rng.random(len(chunk)) < SAMPLE_FRACTION)
        )

        chunk = chunk.loc[keep].copy()
        labels = labels.loc[keep]

        if chunk.empty:
            continue

        features = chunk.drop(columns="Label").apply(
            pd.to_numeric, errors="raise"
        )

        # Standardize non-finite values before duplicate checks.
        features = features.replace([np.inf, -np.inf], np.nan)

        if feature_columns is None:
            feature_columns = features.columns.tolist()
        elif features.columns.tolist() != feature_columns:
            raise ValueError(f"Feature schema mismatch: {file.name}")

        features["attack_type"] = labels.to_numpy()
        features["target"] = (
            labels.ne("BENIGN").astype("int8").to_numpy()
        )

        # Retain provenance for analysis, never as model input.
        features["source_file"] = file.name
        parts.append(features)

if not parts:
    raise SystemExit("No rows were selected.")

data = pd.concat(parts, ignore_index=True)
del parts

print(f"\nSampled rows: {len(data):,}")
print(f"Invalid labels skipped: {invalid_labels:,}")

# Identify identical features with conflicting binary targets.
duplicate_mask = data.duplicated(
    subset=feature_columns, keep=False
)

duplicate_rows = data.loc[duplicate_mask]
conflicting_indices = []

if not duplicate_rows.empty:
    conflicting_mask = (
        duplicate_rows.groupby(
            feature_columns,
            dropna=False,
            sort=False,
        )["target"]
        .transform("nunique")
        .gt(1)
    )

    conflicting_indices = duplicate_rows.index[
        conflicting_mask
    ].tolist()

print(f"Conflicting rows excluded: {len(conflicting_indices):,}")
data = data.drop(index=conflicting_indices)

# Remove exact feature duplicates before creating the splits.
before = len(data)
data = data.drop_duplicates(
    subset=feature_columns, keep="first"
).reset_index(drop=True)

print(f"Duplicate rows removed: {before - len(data):,}")
print(f"Remaining rows: {len(data):,}")

# Stratify by the binary target, not the rare attack families.
train, remaining = train_test_split(
    data,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=data["target"],
)

validation, test = train_test_split(
    remaining,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=remaining["target"],
)

for name, split in [
    ("train", train),
    ("validation", validation),
    ("test", test),
]:
    output_path = OUTPUT_DIR / f"{name}.csv"
    split.to_csv(output_path, index=False)

    print(f"\n{name.upper()}: {len(split):,} rows")
    print(
        split["target"]
        .map({0: "BENIGN", 1: "ATTACK"})
        .value_counts()
        .to_string()
    )

print("\nAttack coverage by split:")
coverage = pd.DataFrame({
    "train": train["attack_type"].value_counts(),
    "validation": validation["attack_type"].value_counts(),
    "test": test["attack_type"].value_counts(),
}).fillna(0).astype(int)

print(coverage.to_string())

print(f"\nFiles saved to: {OUTPUT_DIR}")
print("Preparation completed.")