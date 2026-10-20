from pathlib import Path
import pandas as pd

DATA_DIR = Path(
    r"D:\digilians\cyberSecurity\ACUD Train\Project"
    r"\MachineLearningCSV\MachineLearningCVE"
)

files = sorted(DATA_DIR.glob("*.csv"))

if not files:
    raise SystemExit(
        f"No CSV files found. Check the folder path:\n{DATA_DIR}"
    )

print(f"CSV files found: {len(files)}")

for file in files:
    print(f"- {file.name}")

# نقرأ أول 5 صفوف فقط للفحص
first_file = files[0]
df = pd.read_csv(first_file, nrows=5)
df.columns = df.columns.str.strip()

print(f"\nInspecting: {first_file.name}")
print(f"Number of columns: {len(df.columns)}")

print("\nColumns:")
print(df.columns.tolist())

print("\nSample:")
print(df.head().to_string(index=False))

if "Label" in df.columns:
    print("\nLabels in this sample only:")
    print(df["Label"].unique())