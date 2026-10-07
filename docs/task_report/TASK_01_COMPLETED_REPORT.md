# TASK 01 COMPLETED REPORT — Project Setup & Dataset Ingestion

---

## 1. Task Metadata

| Attribute | Details |
|---|---|
| **Task ID** | `TASK_01` |
| **Task Name** | Project Setup & Dataset Ingestion |
| **Execution Status** | **COMPLETED** |
| **Completion Date** | 2026-10-06 |
| **Target Folder** | `docs/task_reports/` |
| **Lead Developer** | Antigravity AI |

---

## 2. Objective & Scope

The objective of TASK 01 was to establish a modular, scalable project workspace, configure environment logging and path utilities, safely ingest the raw telecom dataset into `data/raw/data_23.csv` without modifying original files, and verify the ingestion pipeline using automated unit tests.

Additionally, as per user requirement, this dedicated task completion documentation folder (`docs/task_reports/`) was added to store standalone `.md` completion reports for every project task.

---

## 3. Work Performed

### 3.1 Directory Structure Initialization
The complete directory hierarchy was generated across the workspace:
- `data/raw/` — Dedicated storage for original raw datasets (immutable).
- `data/processed/` — Storage for cleaned, encoded, and engineered datasets.
- `src/` — Modular Python source package.
- `notebooks/` — Dedicated directory for exploratory & analytical Jupyter Notebooks.
- `models/` — Storage for serialized `.joblib` model artifacts & metadata.
- `api/` — FastAPI REST backend application source.
- `app/` — Streamlit Web UI application source.
- `tests/` — Automated pytest / unittest suite.
- `docs/task_reports/` — **Centralized task completion reports folder**.
- `logs/` — Application execution logs (`app.log`).

### 3.2 Safe Raw Dataset Ingestion
- `data_23.csv` was safely copied to `data/raw/data_23.csv`.
- The original root file `data_23.csv` was left completely untouched.

### 3.3 Core Codebase Implementation
1. **`src/__init__.py`**: Initialized Python package.
2. **`src/utils.py`**:
   - `get_project_root()`: Dynamic path resolution ensuring cross-platform OS compatibility (Windows/Linux/macOS).
   - `get_logger()`: Configured dual logging outputting structured log events simultaneously to `sys.stdout` console and `logs/app.log`.
3. **`src/data_loader.py`**:
   - `load_raw_data()`: Safely loads `data/raw/data_23.csv` into a Pandas DataFrame with error handling for missing files.
   - `get_dataset_summary()`: Computes dataset metadata including shape, column dtypes, missing value counts, and duplicate row detection.

### 3.4 Automated Unit Testing
- Developed `tests/test_data_loader.py` containing 3 automated unit test cases:
  1. `test_load_raw_data_shape`: Verifies exact 7,111 rows and 21 columns loaded.
  2. `test_raw_data_columns`: Verifies exact matching sequence of all 21 column names.
  3. `test_dataset_summary`: Validates duplicate counts (20) and missing column counts (17).

---

## 4. Ingestion Baseline Results

| Metric | Result Value |
|---|---|
| **Total Ingested Rows** | 7,111 records |
| **Total Features / Columns** | 21 features |
| **Target Variable** | `Churn` |
| **Duplicate Rows Detected** | 20 exact duplicate rows |
| **Columns with Missing Values** | 17 columns |
| **Data Ingestion Latency** | < 0.25 seconds |

---

## 5. Verification & Test Execution Logs

Command executed:
```bash
python -m unittest discover -s tests
```

Output:
```text
[2026-10-06 20:21:28] [INFO] [data_loader] Loading raw data from: C:\Users\mihir\Desktop\Telecom Customer Churn Prediction & Retention System\data\raw\data_23.csv
[2026-10-06 20:21:28] [INFO] [data_loader] Successfully loaded dataset with shape: (7111, 21) (rows=7111, cols=21)
[2026-10-06 20:21:28] [INFO] [data_loader] Dataset Summary: Rows=7111, Cols=21, Duplicates=20, Cols with NaNs=17
.[2026-10-06 20:21:28] [INFO] [data_loader] Loading raw data from: C:\Users\mihir\Desktop\Telecom Customer Churn Prediction & Retention System\data\raw\data_23.csv
.[2026-10-06 20:21:28] [INFO] [data_loader] Loading raw data from: C:\Users\mihir\Desktop\Telecom Customer Churn Prediction & Retention System\data\raw\data_23.csv
.
----------------------------------------------------------------------
Ran 3 tests in 0.246s

OK
```

---

## 6. Created / Affected Files Summary

- [src/__init__.py](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/src/__init__.py)
- [src/utils.py](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/src/utils.py)
- [src/data_loader.py](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/src/data_loader.py)
- [data/raw/data_23.csv](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/data/raw/data_23.csv)
- [tests/test_data_loader.py](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/tests/test_data_loader.py)
- [docs/task_reports/TASK_01_COMPLETED_REPORT.md](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/docs/task_reports/TASK_01_COMPLETED_REPORT.md)

---

## 7. Status & Next Task

**TASK 01 is 100% Complete.**

Next planned task: **TASK 02 — Data Quality Audit & Anomaly Detection**.
Waiting for user explicit instruction to proceed with TASK 02.
