# TASK 02 COMPLETED REPORT — Data Quality Audit & Anomaly Detection

---

## 1. Task Metadata

| Attribute | Details |
|---|---|
| **Task ID** | `TASK_02` |
| **Task Name** | Data Quality Audit & Anomaly Detection |
| **Execution Status** | **COMPLETED** |
| **Completion Date** | 2026-10-06 |
| **Target Folder** | `docs/task_reports/` |
| **Lead Developer** | Antigravity AI |

---

## 2. Objective & Scope

The objective of TASK 02 was to perform an automated data quality audit on the raw dataset (`data/raw/data_23.csv`), identifying missing values, empty whitespace strings, duplicate records, target class distributions, and data type anomalies.

All audit results were exported to `reports/data_quality_audit.json`, documented in `notebooks/01_data_understanding_and_cleaning.ipynb`, verified with 8 unit tests in `tests/test_quality_audit.py`, and recorded in this standalone report inside `docs/task_reports/`.

---

## 3. Work Performed

### 3.1 Automated Audit Script (`src/quality_audit.py`)
Developed `src/quality_audit.py` containing:
- `run_data_quality_audit()`: Analyzes missing value counts/percentages per feature, locates whitespace characters in string columns, quantifies duplicate rows and duplicate IDs, breaks down target class counts, and checks feature data types.
- `save_audit_report()`: Exports structured audit findings to JSON format in `reports/data_quality_audit.json`.

### 3.2 Audit Report Export (`reports/data_quality_audit.json`)
Exported structured JSON artifact cataloging 100% of data quality metrics.

### 3.3 Data Understanding Notebook (`notebooks/01_data_understanding_and_cleaning.ipynb`)
Generated Jupyter Notebook containing code blocks for dataset loading, automated quality auditing, and markdown analysis cells documenting data quality insights.

### 3.4 Automated Unit Tests (`tests/test_quality_audit.py`)
Implemented unit test cases validating:
1. Presence of top-level audit dictionary keys.
2. Dataset metrics (7,111 rows, 21 columns, 20 exact duplicates).
3. Whitespace anomalies (11 blank `TotalCharges` records corresponding to `tenure == 0.0`).
4. Target class counts (5,166 `No`, 1,873 `Yes`, 72 NaNs).
5. JSON report file serialization and cleanup.

---

## 4. Empirical Data Quality Findings Summary

### 4.1 Missing Values Breakdown (17 Features)

| Feature Name | Storage Type | Missing Count | Missing Percentage (%) | Cleaning Action Planned (TASK 03) |
|---|---|---|---|---|
| `Churn` (Target) | `object` | 72 | 1.013% | Drop 72 rows missing target |
| `TotalCharges` | `object` | 72 | 1.013% | Impute missing values with Median |
| `Contract` | `object` | 29 | 0.408% | Impute missing categories with Mode |
| `TechSupport` | `object` | 27 | 0.380% | Impute missing categories with Mode |
| `StreamingTV` | `object` | 27 | 0.380% | Impute missing categories with Mode |
| `StreamingMovies` | `object` | 27 | 0.380% | Impute missing categories with Mode |
| `DeviceProtection` | `object` | 20 | 0.281% | Impute missing categories with Mode |
| `MonthlyCharges` | `float64` | 18 | 0.253% | Impute missing values with Median |
| `InternetService` | `object` | 17 | 0.239% | Impute missing categories with Mode |
| `OnlineSecurity` | `object` | 16 | 0.225% | Impute missing categories with Mode |
| `PaperlessBilling` | `object` | 16 | 0.225% | Impute missing categories with Mode |
| `PaymentMethod` | `object` | 16 | 0.225% | Impute missing categories with Mode |
| `PhoneService` | `object` | 13 | 0.183% | Impute missing categories with Mode |
| `MultipleLines` | `object` | 13 | 0.183% | Impute missing categories with Mode |
| `OnlineBackup` | `object` | 9 | 0.127% | Impute missing categories with Mode |
| `Dependents` | `object` | 7 | 0.098% | Impute missing categories with Mode |
| `tenure` | `float64` | 7 | 0.098% | Impute missing values with Median |

### 4.2 Whitespace & Type Anomalies
- **`TotalCharges` Column**: Loaded as `object` data type due to **11 empty space strings (`' '`)**.
- **Tenure Cross-Reference**: All 11 whitespace rows correspond to customers with `tenure == 0.0` (new accounts with $0 accrued total bill).
- **Required Action**: Convert `' '` to `0.0` and coerce column to `float64`.

### 4.3 Duplicate Records
- **Exact Duplicate Rows**: 20 identical row entries detected across all features.
- **Duplicate Customer IDs**: 68 duplicate customer ID records detected.
- **Required Action**: Remove 20 exact duplicate rows during TASK 03 data cleaning.

### 4.4 Target Variable Class Imbalance (`Churn`)
- `No` (Retained Customers): **5,166 records** (72.65% of non-null)
- `Yes` (Churned Customers): **1,873 records** (26.34% of non-null)
- `NaN` (Missing Target): **72 records** (1.01%)
- **Imbalance Rationale**: Class ratio is approximately 2.76 : 1. Accuracy alone will be misleading; Recall, F1-Score, and ROC-AUC will serve as primary evaluation metrics.

---

## 5. Verification & Test Suite Logs

Command executed:
```bash
python -m unittest discover -s tests
```

Output:
```text
Ran 8 tests in 0.958s

OK
```

---

## 6. Created & Modified Artifacts

- [src/quality_audit.py](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/src/quality_audit.py)
- [reports/data_quality_audit.json](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/reports/data_quality_audit.json)
- [notebooks/01_data_understanding_and_cleaning.ipynb](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/notebooks/01_data_understanding_and_cleaning.ipynb)
- [tests/test_quality_audit.py](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/tests/test_quality_audit.py)
- [docs/task_reports/TASK_02_COMPLETED_REPORT.md](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/docs/task_reports/TASK_02_COMPLETED_REPORT.md)

---

## 7. Status & Next Task

**TASK 02 is 100% Complete.**

Next planned task: **TASK 03 — Data Cleaning & Preprocessing Pipeline Design**.
Waiting for user explicit instruction to proceed with TASK 03.
