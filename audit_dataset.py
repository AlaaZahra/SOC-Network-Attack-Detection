from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

DATA_DIR = Path(
    r"D:\digilians\cyberSecurity\ACUD Train\Project"
    r"\MachineLearningCSV\MachineLearningCVE"
)

CHUNK_SIZE = 100_000
files = sorted(DATA_DIR.glob("*.csv"))

if not files:
    raise SystemExit(f"No CSV files found in: {DATA_DIR}")

label_counts = Counter()
missing_counts = Counter()
infinite_counts = Counter()

total_rows = 0
missing_labels = 0

for file in files:
    file_rows = 0
    print(f"\nReading: {file.name}", flush=True)

    # Read each file in chunks to limit memory usage.
    for chunk in pd.read_csv(
        file,
        chunksize=CHUNK_SIZE,
        encoding="utf-8-sig",
        low_memory=False,
    ):
        chunk.columns = chunk.columns.str.strip()

        if "Label" not in chunk.columns:
            raise ValueError(f"Label column missing in: {file.name}")

        labels = chunk["Label"].astype("string").str.strip()
        invalid_labels = labels.isna() | labels.eq("").fillna(False)

        missing_labels += int(invalid_labels.sum())
        label_counts.update(
            labels[~invalid_labels].value_counts().to_dict()
        )

        features = chunk.drop(columns="Label")

        # Count missing values separately from infinite values.
        missing = features.isna().sum()
        missing_counts.update(
            {name: int(count) for name, count in missing.items() if count}
        )

        numeric = features.select_dtypes(include="number")
        infinite = np.isinf(numeric).sum()
        infinite_counts.update(
            {name: int(count) for name, count in infinite.items() if count}
        )

        file_rows += len(chunk)

    total_rows += file_rows
    print(f"Rows: {file_rows:,}", flush=True)

print("\n========== DATASET SUMMARY ==========")
print(f"Total rows: {total_rows:,}")
print(f"Missing or empty labels: {missing_labels:,}")

print("\nLabel distribution:")
for label, count in label_counts.most_common():
    percentage = count / total_rows * 100 if total_rows else 0
    print(f"{label}: {count:,} ({percentage:.2f}%)")

print("\nMissing feature values:")
if missing_counts:
    for column, count in missing_counts.most_common():
        print(f"{column}: {count:,}")
else:
    print("None")

print("\nInfinite feature values:")
if infinite_counts:
    for column, count in infinite_counts.most_common():
        print(f"{column}: {count:,}")
else:
    print("None")

print("\nAudit completed. Original files were not modified.")