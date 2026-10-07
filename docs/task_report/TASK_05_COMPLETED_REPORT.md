# TASK 05 COMPLETED REPORT — Correlation & Multicollinearity Analysis

---

## 1. Task Metadata

| Attribute | Details |
|---|---|
| **Task ID** | `TASK_05` |
| **Task Name** | Correlation & Multicollinearity Analysis |
| **Execution Status** | **COMPLETED** |
| **Completion Date** | 2026-10-06 |
| **Target Folder** | `docs/task_reports/` |
| **Lead Developer** | Antigravity AI |

---

## 2. Objective & Scope

The objective of TASK 05 was to calculate Pearson and Spearman correlation matrices across numerical features and the target variable (`Churn`), compute Variance Inflation Factors (VIF) to diagnose multicollinearity risks, export an annotated heatmap figure (`reports/figures/05_correlation_heatmap.png`), export a structured JSON report (`reports/correlation_analysis.json`), update `notebooks/02_exploratory_data_analysis.ipynb`, implement automated unit tests in `tests/test_correlation.py`, and establish feature retention recommendations.

---

## 3. Work Performed

### 3.1 Correlation & Multicollinearity Module (`src/correlation.py`)
Developed `src/correlation.py` containing:
- `compute_vif()`: Computes Variance Inflation Factor (VIF) for numerical predictor features using `statsmodels`.
- `run_correlation_analysis()`:
  - Encodes binary target `Churn` into `Churn_Encoded` (1 = Yes, 0 = No).
  - Computes Pearson and Spearman correlation matrices.
  - Calculates VIF scores for `tenure`, `MonthlyCharges`, and `TotalCharges`.
  - Generates and exports 300 DPI annotated correlation heatmap `reports/figures/05_correlation_heatmap.png`.
  - Exports structured correlation JSON summary to `reports/correlation_analysis.json`.

### 3.2 Artifact Exports
- **Heatmap Image**: [reports/figures/05_correlation_heatmap.png](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/reports/figures/05_correlation_heatmap.png)
- **JSON Metrics**: [reports/correlation_analysis.json](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/reports/correlation_analysis.json)

### 3.3 Exploratory Notebook Update (`notebooks/02_exploratory_data_analysis.ipynb`)
Updated Jupyter notebook with Section 6 (Correlation & Multicollinearity Analysis), rendering the heatmap figure, printing VIF scores, and documenting modeling feature retention strategies.

### 3.4 Unit Test Suite Expansion (`tests/test_correlation.py`)
Implemented automated unit tests asserting:
1. `compute_vif()` returns exact VIF data structure and confirms `TotalCharges` VIF $> 8.0$.
2. `run_correlation_analysis()` returns expected dictionary keys, generates `05_correlation_heatmap.png`, and exports `correlation_analysis.json`.

---

## 4. Key Empirical Findings

### 4.1 Pearson & Spearman Correlation Matrix

| Feature Pair | Pearson Correlation ($r$) | Spearman Correlation ($\rho$) | Relationship & Interpretation |
|---|---|---|---|
| `tenure` vs `TotalCharges` | **+0.8258** | **+0.8893** | **Strong Positive Collinearity**: Cumulative charges naturally accumulate over tenure duration. |
| `MonthlyCharges` vs `TotalCharges` | **+0.6515** | **+0.6385** | **Moderate-High Positive Correlation**: Higher monthly bills scale total charges up. |
| `tenure` vs `MonthlyCharges` | **+0.2485** | **+0.2771** | **Weak Positive Correlation**: Longer tenure customers have slightly higher monthly plans. |

### 4.2 Feature Correlations with Target (`Churn`)

| Numerical Feature | Pearson $r$ with `Churn` | Direction & Business Meaning |
|---|---|---|
| `tenure` | **-0.3527** | **Strongest Negative Predictor**: Longer customer tenure significantly reduces churn risk. |
| `TotalCharges` | **-0.1987** | **Moderate Negative Predictor**: Higher lifetime spending reflects customer loyalty retention. |
| `MonthlyCharges` | **+0.1937** | **Moderate Positive Predictor**: Higher monthly bills increase customer price sensitivity and churn. |
| `SeniorCitizen` | **+0.1493** | **Weak Positive Predictor**: Senior citizens exhibit slightly higher churn rates. |

### 4.3 Multicollinearity Diagnostic (Variance Inflation Factor - VIF)

| Feature Name | VIF Score | Multicollinearity Risk Level |
|---|---|---|
| `TotalCharges` | **9.450** | **HIGH MULTICOLLINEARITY** ($\text{VIF} > 8.0$) |
| `tenure` | **5.797** | **MODERATE MULTICOLLINEARITY** ($5.0 < \text{VIF} < 8.0$) |
| `MonthlyCharges` | **3.204** | **LOW MULTICOLLINEARITY** ($\text{VIF} < 5.0$) |

---

## 5. Modeling Feature Retention Strategy

1. **Tree-Based Ensemble Models (Random Forest, XGBoost, LightGBM)**:
   - Tree models naturally partition feature space and are immune to multicollinearity.
   - **Decision**: Retain all numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) for tree-based classifiers.

2. **Linear Models (Logistic Regression, SVM)**:
   - High VIF (9.450) between `tenure` and `TotalCharges` can destabilize coefficient estimation in unregularized linear models.
   - **Decision**: Evaluate $L1 / L2$ regularization penalty or feature ratio engineering (`AvgMonthlyChargeRatio = TotalCharges / (tenure + 1)`) during Feature Engineering (TASK 06) and Selection (TASK 07).

---

## 6. Verification & Test Suite Logs

Command executed:
```bash
python -m unittest discover -s tests
```

Output:
```text
Ran 18 tests in 12.235s

OK
```

---

## 7. Created & Modified Artifacts

- [src/correlation.py](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/src/correlation.py)
- [reports/figures/05_correlation_heatmap.png](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/reports/figures/05_correlation_heatmap.png)
- [reports/correlation_analysis.json](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/reports/correlation_analysis.json)
- [notebooks/02_exploratory_data_analysis.ipynb](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/notebooks/02_exploratory_data_analysis.ipynb)
- [tests/test_correlation.py](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/tests/test_correlation.py)
- [docs/task_reports/TASK_05_COMPLETED_REPORT.md](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/docs/task_reports/TASK_05_COMPLETED_REPORT.md)

---

## 8. Status & Next Task

**TASK 05 is 100% Complete.**

Next planned task: **TASK 06 — Feature Engineering & Domain Transformation**.
Waiting for user explicit instruction to proceed with TASK 06.
