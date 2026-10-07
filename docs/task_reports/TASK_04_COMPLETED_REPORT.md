# TASK 04 COMPLETED REPORT — Exploratory Data Analysis (EDA) & Domain Insights

---

## 1. Task Metadata

| Attribute | Details |
|---|---|
| **Task ID** | `TASK_04` |
| **Task Name** | Exploratory Data Analysis (EDA) & Domain Insights |
| **Execution Status** | **COMPLETED** |
| **Completion Date** | 2026-10-06 |
| **Target Folder** | `docs/task_reports/` |
| **Lead Developer** | Antigravity AI |

---

## 2. Objective & Scope

The objective of TASK 04 was to execute statistical and visual Exploratory Data Analysis (EDA) on the cleaned dataset (`data/processed/cleaned_churn_data.csv`). High-resolution visualization figures were exported to `reports/figures/`, an interactive notebook was generated at `notebooks/02_exploratory_data_analysis.ipynb` with explicit business observations under every visualization, and automated unit tests were implemented in `tests/test_eda.py`.

---

## 3. Work Performed

### 3.1 EDA Pipeline Module (`src/eda.py`)
Developed `src/eda.py` containing:
- `generate_eda_figures()`: Generates 4 high-resolution publication-quality PNG charts (300 DPI):
  1. `01_target_distribution.png`: Bar chart of `Churn` class balance with count and percentage annotations.
  2. `02_numerical_distributions.png`: Histograms with KDE curves and box plots for `tenure`, `MonthlyCharges`, and `TotalCharges`.
  3. `03_categorical_vs_churn.png`: Grouped count plots showing churn breakdown across `Contract`, `InternetService`, `PaymentMethod`, `TechSupport`, `OnlineSecurity`, and `SeniorCitizen`.
  4. `04_multivariate_tenure_vs_charges.png`: Scatter plot mapping `tenure` vs `MonthlyCharges` colored by `Churn` status.
- `compute_eda_summary_statistics()`: Computes mean, median, standard deviation metrics by Churn group and cross-tabulation churn percentages across categorical features.

### 3.2 Visual Figures Export (`reports/figures/`)
Saved 4 standalone PNG charts:
- `reports/figures/01_target_distribution.png`
- `reports/figures/02_numerical_distributions.png`
- `reports/figures/03_categorical_vs_churn.png`
- `reports/figures/04_multivariate_tenure_vs_charges.png`

### 3.3 Exploratory Analysis Notebook (`notebooks/02_exploratory_data_analysis.ipynb`)
Generated complete EDA Jupyter notebook embedding visual charts, summary dataframes, and detailed markdown sections for business insights.

### 3.4 Unit Test Suite Expansion (`tests/test_eda.py`)
Implemented automated unit tests asserting:
1. All 4 EDA figure files are created and exist in `reports/figures/`.
2. Summary statistics calculation returns expected numerical feature keys (`tenure`, `MonthlyCharges`, `TotalCharges`) and categorical cross-tabulations.

---

## 4. Key EDA & Domain Insights Summary

### 4.1 Target Class Balance (`Churn`)
- **Retained (`No`)**: **5,163 customers (73.56%)**
- **Churned (`Yes`)**: **1,856 customers (26.44%)**
- **Implication**: Severe class imbalance. Accuracy is misleading (73.56% baseline predicting all `No`). **Recall** and **F1-score** must be prioritized during model evaluation.

### 4.2 Numerical Driver Analysis

| Feature | Retained (`No`) Median | Churned (`Yes`) Median | Key Insight & Business Impact |
|---|---|---|---|
| `tenure` | **38.0 months** | **10.0 months** | Churn is heavily concentrated in the first **0 to 12 months** of customer lifecycle. |
| `MonthlyCharges` | **$64.43** | **$79.65** | High monthly bill amounts significantly increase price sensitivity and churn likelihood. |
| `TotalCharges` | **$1,679.33** | **$703.55** | Retained customers generate > 2.3x higher cumulative lifetime value than churned customers. |

### 4.3 Categorical Key Driver Analysis

| Feature / Category | Category Level | Churn Rate (%) | Risk Level & Retention Recommendation |
|---|---|---|---|
| **Contract** | Month-to-Month | **42.71%** | **CRITICAL RISK**: Offer transition discounts to 1-Year or 2-Year contracts. |
| **Contract** | One-Year | **11.27%** | **LOW RISK**: High loyalty retention. |
| **Contract** | Two-Year | **2.83%** | **MINIMAL RISK**: Locked-in retention anchor. |
| **InternetService** | Fiber Optic | **41.89%** | **HIGH RISK**: High monthly cost and potential service friction. Target for support. |
| **InternetService** | DSL | **18.96%** | **MODERATE RISK**: Stable retention profile. |
| **TechSupport** | No Tech Support | **41.64%** | **HIGH RISK**: Customers lacking support defect easily. Offer free 3-month trial. |
| **TechSupport** | Has Tech Support | **15.17%** | **LOW RISK**: Tech support serves as a core ecosystem anchor. |

### 4.4 Multivariate Interaction Insight
- Customers with **Short Tenure ($< 15$ months)** combined with **High Monthly Charges ($> \$70/month$)** and **Month-to-Month Contracts** represent the single highest churn risk cohort (> 55% churn rate).

---

## 5. Verification & Test Suite Logs

Command executed:
```bash
python -m unittest discover -s tests
```

Output:
```text
Ran 16 tests in 11.419s

OK
```

---

## 6. Created & Modified Artifacts

- [src/eda.py](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/src/eda.py)
- [reports/figures/01_target_distribution.png](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/reports/figures/01_target_distribution.png)
- [reports/figures/02_numerical_distributions.png](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/reports/figures/02_numerical_distributions.png)
- [reports/figures/03_categorical_vs_churn.png](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/reports/figures/03_categorical_vs_churn.png)
- [reports/figures/04_multivariate_tenure_vs_charges.png](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/reports/figures/04_multivariate_tenure_vs_charges.png)
- [notebooks/02_exploratory_data_analysis.ipynb](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/notebooks/02_exploratory_data_analysis.ipynb)
- [tests/test_eda.py](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/tests/test_eda.py)
- [docs/task_reports/TASK_04_COMPLETED_REPORT.md](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/docs/task_reports/TASK_04_COMPLETED_REPORT.md)

---

## 7. Status & Next Task

**TASK 04 is 100% Complete.**

Next planned task: **TASK 05 — Correlation & Multicollinearity Analysis**.
Waiting for user explicit instruction to proceed with TASK 05.
