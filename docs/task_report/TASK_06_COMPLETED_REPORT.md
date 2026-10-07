# TASK 06 — Feature Engineering & Domain Transformation
## Completion Report

**Project**: Telecom Customer Churn Prediction & Retention System
**Task ID**: TASK 06
**Status**: COMPLETED
**Executed On**: 2026-10-06

---

## 1. Objective

Construct domain-specific attributes, ratio features, interaction features, and categorical encodings to enhance classifier predictive power.

---

## 2. Input / Output

| Item | Path | Description |
|------|------|-------------|
| Input | data/processed/cleaned_churn_data.csv | 7,019 rows x 21 columns (from TASK 03) |
| Output | data/processed/engineered_churn_data.csv | 7,019 rows x 40 columns |
| Module | src/features.py | FeatureEngineeringTransformer class |
| Notebook | notebooks/03_feature_engineering_and_selection.ipynb | Interactive walkthrough |
| Tests | tests/test_features.py | 8 unit tests, all passing |

---

## 3. Features Engineered

### 3A. Domain Ratio Features (Continuous, Standardized)

| Feature | Formula | Rationale |
|---------|---------|-----------|
| AvgMonthlyChargeRatio | TotalCharges / (tenure + 1) | Historical vs current billing |
| MonthlyToTotalRatio | MonthlyCharges / (TotalCharges + 1) | Short-tenure proportional spend |
| TotalServicesCount | Sum of 6 service flags | Ecosystem bundling index |

### 3B. Interaction / Behavioral Flags (Binary)

| Feature | Logic | Insight |
|---------|-------|---------|
| HasTechSupportSecurity | TechSupport==Yes AND OnlineSecurity==Yes | Premium add-on: <15% churn |
| IsAutomaticPayment | PaymentMethod in {Credit card, Bank transfer} | Auto-pay lowers churn risk |

### 3C. Tenure Cohort Binning (OHE Lifecycle Segmentation)

Bins: 0-12m (reference), 12-24m, 24-48m, 48-72m
OHE columns: TenureCohort_12-24m, TenureCohort_24-48m, TenureCohort_48-72m

### 3D. Categorical Encodings

- Binary map: gender, Partner, Dependents, PhoneService, PaperlessBilling
- One-Hot (drop_first): Contract, InternetService, PaymentMethod, MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- Target: Churn -> 0/1

### 3E. Numerical Scaling

StandardScaler on: tenure, MonthlyCharges, TotalCharges, AvgMonthlyChargeRatio, MonthlyToTotalRatio, TotalServicesCount

---

## 4. Final Feature Inventory (40 columns total)

38 predictor features + customerID + Churn:
customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
PaperlessBilling, MonthlyCharges, TotalCharges, Churn, AvgMonthlyChargeRatio,
MonthlyToTotalRatio, TotalServicesCount, HasTechSupportSecurity, IsAutomaticPayment,
Contract_One year, Contract_Two year, InternetService_Fiber optic, InternetService_No,
PaymentMethod_Credit card (automatic), PaymentMethod_Electronic check, PaymentMethod_Mailed check,
MultipleLines_No phone service, MultipleLines_Yes, OnlineSecurity_No internet service,
OnlineSecurity_Yes, OnlineBackup_No internet service, OnlineBackup_Yes,
DeviceProtection_No internet service, DeviceProtection_Yes, TechSupport_No internet service,
TechSupport_Yes, StreamingTV_No internet service, StreamingTV_Yes,
StreamingMovies_No internet service, StreamingMovies_Yes,
TenureCohort_12-24m, TenureCohort_24-48m, TenureCohort_48-72m

---

## 5. Validation Results

| Check | Result |
|-------|--------|
| Output shape | PASS: 7,019 x 40 |
| Zero null values | PASS: 0 nulls |
| Domain features present | PASS: All 5 confirmed |
| Tenure cohort OHE cols | PASS: All 3 confirmed |
| Binary columns in {0,1} | PASS: Validated |
| Churn encoded as {0,1} | PASS: Confirmed |
| Engineered CSV saved | PASS: Verified |

Unit Tests: tests/test_features.py -- 8 tests | 26 total suite tests | ALL PASSED

---

## 6. Implementation Notes

FeatureEngineeringTransformer uses a stateful fit/transform design consistent with TelecomDataCleaner.
StandardScaler is fitted strictly on training data to prevent leakage during cross-validation.

---

## 7. Files Created / Modified

| File | Type | Description |
|------|------|-------------|
| src/features.py | Created | Feature engineering transformer |
| data/processed/engineered_churn_data.csv | Generated | Model-ready feature matrix |
| notebooks/03_feature_engineering_and_selection.ipynb | Created | Engineering notebook |
| tests/test_features.py | Created | 8 unit tests |
| docs/task_reports/TASK_06_COMPLETED_REPORT.md | Created | This report |

---

## 8. Next Task

TASK 07 - Baseline Model Evaluation and Benchmarking:
- Train Logistic Regression, Decision Tree, KNN, Naive Bayes baselines
- Evaluate with Accuracy, Precision, Recall, F1, AUC-ROC
- Log metrics to reports/baseline_metrics.json
