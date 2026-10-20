# SOC Network Attack Detection

A machine learning component for an Enterprise SOC Platform that classifies network flows as **BENIGN** or **ATTACK** using Random Forest and CICIDS2017.

## Features

- Dataset inspection and quality auditing.
- Chunk-based sampling and duplicate filtering.
- Training-only preprocessing and binary classification.
- Validation and test evaluation with error analysis.
- Batch CSV predictions for future dashboard integration.

## Dataset

Source: [CICIDS2017 — Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html).

Download and extract `MachineLearningCSV.zip`. Refer to the source for usage and citation requirements.

The inspected dataset contains 8 CSV files, 2,830,743 flows, 78 features, and a label column. Labels cover benign traffic and 14 attack types, including brute force, PortScan, DoS, DDoS, Bot, web attacks, Infiltration, and Heartbleed.

## Data Preparation

Common classes are randomly sampled at approximately 10.6%. Classes with at most 2,000 original examples are retained before filtering. Infinite values are converted to missing values. Identical feature rows with conflicting binary targets are excluded from the sample, and exact feature duplicates are removed before splitting.

| Preparation step | Rows |
|---|---:|
| Initial sample | 303,050 |
| Conflicting rows excluded | 29 |
| Duplicate rows removed | 14,976 |
| Final sample | 288,045 |

| Split | Rows | BENIGN | ATTACK |
|---|---:|---:|---:|
| Training (70%) | 201,631 | 163,044 | 38,587 |
| Validation (15%) | 43,207 | 34,938 | 8,269 |
| Test (15%) | 43,207 | 34,938 | 8,269 |

Splits are stratified by binary target. Original attack names and source filenames are metadata, never model inputs.

## Model

A scikit-learn Pipeline combines median imputation with Random Forest:

- 100 trees; maximum depth 20; minimum leaf size 2.
- Balanced subsample class weights and bootstrap sample fraction 0.8.
- Random seed 42 and decision threshold 0.5.
- Training-only filtering retains 70 nonconstant features.
- Imputation is fitted on training data only.

The saved artifact includes the fitted pipeline, feature schema, class mapping, and threshold.

## Results

| Metric | Validation | Test |
|---|---:|---:|
| Attack Precision | 99.42% | 99.40% |
| Attack Recall | 99.60% | 99.55% |
| Attack F1-score | 99.51% | 99.47% |
| False Positive Rate | 0.1374% | 0.1431% |
| Average Precision | 0.9993 | 0.9994 |

### Test Confusion Matrix

| Actual / Predicted | BENIGN | ATTACK |
|---|---:|---:|
| BENIGN | 34,888 | 50 |
| ATTACK | 37 | 8,232 |

### Detection Recall by Attack Type

These are binary detection results grouped by original labels; the model does not predict attack names.

| Attack type | Test examples | Detected | Recall |
|---|---:|---:|---:|
| Bot | 287 | 272 | 94.77% |
| DDoS | 2,077 | 2,074 | 99.86% |
| DoS GoldenEye | 149 | 147 | 98.66% |
| DoS Hulk | 2,783 | 2,779 | 99.86% |
| DoS Slowhttptest | 76 | 76 | 100.00% |
| DoS slowloris | 75 | 73 | 97.33% |
| FTP-Patator | 104 | 104 | 100.00% |
| Infiltration | 5 | 4 | 80.00% |
| PortScan | 2,331 | 2,331 | 100.00% |
| SSH-Patator | 57 | 56 | 98.25% |
| Web Brute Force | 212 | 210 | 99.06% |
| SQL Injection | 6 | 4 | 66.67% |
| XSS | 107 | 102 | 95.33% |

Heartbleed has no test examples and is not evaluated in this split.

## Repository Files

| File | Purpose |
|---|---|
| `inspect_dataset.py` | Inspect files, columns, and sample rows |
| `audit_dataset.py` | Count labels, missing values, and infinities |
| `prepare_dataset.py` | Sample, filter duplicates, and create splits |
| `train_model.py` | Train, validate, and save the model |
| `evaluate_test.py` | Evaluate the saved baseline |
| `ml_predictor.py` | Run batch predictions |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Exclude datasets, models, and local files |

## Setup

Developed using Python 3.13.

```bash
git clone https://github.com/AlaaZahra/SOC-Network-Attack-Detection.git
cd SOC-Network-Attack-Detection
python -m pip install -r requirements.txt
```

Update `DATA_DIR` in `inspect_dataset.py`, `audit_dataset.py`, and `prepare_dataset.py` to the local folder containing the eight CSV files. The scripts currently contain a developer-specific Windows path.

## Workflow

### Inspect and audit

```bash
python inspect_dataset.py
python audit_dataset.py
```

### Prepare splits

```bash
python prepare_dataset.py
```

Creates split files under `Data/ml/`.

### Train and validate

```bash
python train_model.py
```

Saves `Models/network_detector.joblib` and exports `Data/ml/validation_errors.csv`.

### Evaluate the fixed baseline

```bash
python evaluate_test.py
```

Exports `Data/ml/test_errors.csv`. Use validation data for model selection and reserve test data for final evaluation.

### Run predictions

```bash
python ml_predictor.py --input "Data/ml/test.csv" --output "Data/ml/predictions.csv"
```

This example demonstrates inference on the previously evaluated test set; it is not an independent evaluation.

For another compatible CSV:

```bash
python ml_predictor.py --input "path/to/flows.csv" --output "path/to/predictions.csv"
```

Inputs must contain all required features with compatible definitions and units. Labels are not required.

| Output column | Meaning |
|---|---|
| `ml_prediction` | BENIGN or ATTACK |
| `ml_attack_score` | Uncalibrated attack-class score |
| `ml_threshold` | Decision threshold |

The score is not a calibrated probability, severity, or risk score. Input fields are preserved in the output.

Datasets, predictions, and model artifacts are excluded from Git. Train locally to generate the model.

## Limitations

- Random splits can share similar traffic and attack scenarios even after exact duplicate removal.
- These results do not establish generalization to new networks or live traffic.
- Rare-class retention changes the sampled distribution; metrics may differ under deployment conditions.
- Estimates for attack types with very few examples are uncertain.
- The model performs binary network-flow classification, not attack-family classification or alert auto-triage.
- Raw AD and VPN authentication logs cannot be passed directly to this model.
- Dashboard integration and live traffic feature extraction are not implemented here.

## Future Work

- Integrate predictions with the Enterprise SOC dashboard.
- Evaluate on independent captures and network environments.
- Compare models and thresholds using validation data.
- Investigate rare-attack detection and score calibration.
