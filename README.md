\# SOC Network Attack Detection



A machine learning component for an Enterprise SOC Platform that classifies network flows as \*\*BENIGN\*\* or \*\*ATTACK\*\* using Random Forest and CICIDS2017.



The project covers dataset inspection, quality auditing, sampling, preprocessing, training, evaluation, and batch prediction.



\## Scope



\- Binary classification of network flows.

\- Validation and test evaluation with false positive and false negative analysis.

\- Per-attack detection recall for analysis.

\- CSV prediction output for future SOC dashboard integration.



The model does not predict attack names or perform alert auto-triage. It requires network flow features and cannot directly classify raw AD or VPN authentication logs.



\## Dataset



\*\*Source:\*\* \[CICIDS2017 — Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)



Download and extract `MachineLearningCSV.zip`.



The inspected dataset contains:



\- 8 CSV files.

\- 2,830,743 network flows.

\- 78 input features and one label column.

\- Benign traffic and 14 attack labels.



Attack labels include FTP/SSH brute force, PortScan, DDoS, several DoS variants, Bot, web brute force, XSS, SQL injection, Infiltration, and Heartbleed.



Dataset files are excluded from this repository. Refer to the source website for usage and citation requirements.



\## Data Preparation



Files are processed in chunks to limit memory usage.



The preparation script:



1\. Counts original attack labels.

2\. Randomly samples common classes at approximately 10.6%.

3\. Retains all examples of classes with at most 2,000 original rows before duplicate filtering.

4\. Converts infinite feature values to missing values.

5\. Excludes sampled rows with identical features but conflicting binary targets.

6\. Removes exact feature duplicates from the sample.

7\. Creates binary-stratified training, validation, and test splits.



| Preparation step | Rows |

|---|---:|

| Initial sample | 303,050 |

| Conflicting rows excluded | 29 |

| Duplicate rows removed | 14,976 |

| Final sample | 288,045 |



| Split | Rows | BENIGN | ATTACK |

|---|---:|---:|---:|

| Training — 70% | 201,631 | 163,044 | 38,587 |

| Validation — 15% | 43,207 | 34,938 | 8,269 |

| Test — 15% | 43,207 | 34,938 | 8,269 |



Original attack labels and source filenames are retained as metadata and excluded from model inputs.



\## Model



The baseline uses a scikit-learn Pipeline containing:



\- Median imputation fitted on training data only.

\- A Random Forest classifier with 100 trees.

\- Maximum tree depth of 20.

\- Minimum leaf size of 2.

\- Balanced subsample class weights.

\- A decision threshold of 0.5.

\- Random seed of 42.



Training-only filtering reduces the input from 78 to 70 nonconstant features.



The saved artifact includes the fitted pipeline, required feature names, class mapping, and decision threshold.



\## Results



Results below are from the fixed baseline and the prepared CICIDS2017 splits.



| Metric | Validation | Test |

|---|---:|---:|

| Attack Precision | 99.42% | 99.40% |

| Attack Recall | 99.60% | 99.55% |

| Attack F1-score | 99.51% | 99.47% |

| False Positive Rate | 0.1374% | 0.1431% |

| Average Precision | 0.9993 | 0.9994 |



\### Test Confusion Matrix



| Actual / Predicted | BENIGN | ATTACK |

|---|---:|---:|

| BENIGN | 34,888 | 50 |

| ATTACK | 37 | 8,232 |



\### Test Detection Recall by Attack Type



The model predicts only BENIGN or ATTACK. This table measures how many examples of each original attack type were classified as ATTACK.



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



Heartbleed has no examples in the test split and is therefore not evaluated there.



\## Repository Files



| File | Purpose |

|---|---|

| `inspect\_dataset.py` | Inspect input files, columns, and sample rows |

| `audit\_dataset.py` | Count labels, missing values, and infinite values |

| `prepare\_dataset.py` | Sample, filter duplicates, and create splits |

| `train\_model.py` | Train, validate, and save the model |

| `evaluate\_test.py` | Evaluate the saved baseline on the test split |

| `ml\_predictor.py` | Run batch predictions using the saved model |

| `requirements.txt` | Python dependencies |

| `.gitignore` | Exclude datasets, model artifacts, and local files |



\## Setup



Developed using Python 3.13.



```bash

git clone https://github.com/AlaaZahra/SOC-Network-Attack-Detection.git

cd SOC-Network-Attack-Detection

python -m pip install -r requirements.txt

```



Extract the dataset locally and update `DATA\_DIR` in:



\- `inspect\_dataset.py`

\- `audit\_dataset.py`

\- `prepare\_dataset.py`



Set it to the folder containing the eight CSV files. The current scripts contain a developer-specific Windows path that must be changed on another machine.



\## Workflow



\### 1. Inspect and audit



```bash

python inspect\_dataset.py

python audit\_dataset.py

```



\### 2. Prepare the dataset



```bash

python prepare\_dataset.py

```



Creates training, validation, and test CSV files under `Data/ml/`.



\### 3. Train and validate



```bash

python train\_model.py

```



Saves the fitted artifact to:



```text

Models/network\_detector.joblib

```



Also exports validation errors to `Data/ml/validation\_errors.csv`.



\### 4. Evaluate the fixed baseline



```bash

python evaluate\_test.py

```



Reports test metrics and exports `Data/ml/test\_errors.csv`.



Use validation data for model selection. Keep the test split reserved for final evaluation rather than repeatedly tuning against it.



\### 5. Run predictions



```bash

python ml\_predictor.py --input "Data/ml/test.csv" --output "Data/ml/predictions.csv"

```



This example reuses the evaluated test set to demonstrate inference; it is not an independent evaluation.



For another compatible flow CSV:



```bash

python ml\_predictor.py --input "path/to/flows.csv" --output "path/to/predictions.csv"

```



The input must contain all required feature columns with compatible meanings and units. Labels are not required.



\### Prediction Output



Input fields are preserved, with these additional columns:



| Column | Meaning |

|---|---|

| `ml\_prediction` | BENIGN or ATTACK |

| `ml\_attack\_score` | Uncalibrated model score for the attack class |

| `ml\_threshold` | Decision threshold used |



The attack score is not a calibrated probability or a severity/risk score.



Datasets, predictions, and saved models are excluded from Git. Train locally to generate the model artifact.



\## Limitations



\- Random splits from the same dataset can share similar traffic patterns and attack scenarios even after exact duplicate removal.

\- Reported metrics do not establish performance on a new network or live traffic.

\- Rare-class retention changes the sampled class distribution.

\- Results for attack types with very few test examples are uncertain.

\- This is a binary classifier, not a multiclass attack classifier.

\- Dashboard integration and live traffic feature extraction are not implemented in this repository.

\- Feature extraction for new traffic must match the training feature definitions.



\## Future Work



\- Integrate prediction output with the Enterprise SOC dashboard.

\- Evaluate on separate captures or network environments.

\- Compare models and thresholds using validation data.

\- Investigate rare-attack detection and score calibration.

