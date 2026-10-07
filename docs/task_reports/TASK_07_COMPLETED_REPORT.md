# TASK 07 — Feature Selection & Importance Ranking
## Completion Report

**Project**: Telecom Customer Churn Prediction & Retention System
**Task ID**: TASK 07
**Status**: COMPLETED
**Executed On**: 2026-10-06

---

## 1. Objective

Evaluate and select the most impactful features using Mutual Information, Tree-based Feature Importances, and Recursive Feature Elimination with Cross-Validation (RFECV) to eliminate noisy or redundant predictors, lower model complexity, and prevent overfitting.

---

## 2. Input / Output

| Item | Path | Description |
|------|------|-------------|
| Input | data/processed/engineered_churn_data.csv | 7,019 rows x 40 columns |
| Output | data/processed/selected_churn_data.csv | Pruned feature subset (28 features) |
| Output | reports/feature_importance_ranking.csv | Ranked feature importance listing |
| Module | src/feature_selection.py | Mutual Info & RFECV logic |
| Notebook | notebooks/03_feature_engineering_and_selection.ipynb | Updated with feature selection cells |
| Tests | tests/test_feature_selection.py | Unit tests for selection logic |

---

## 3. Methodology

We applied three distinct selection mechanisms:

1. **Mutual Information Classification**: Captures both linear and non-linear dependencies between each predictor and the `Churn` target.
2. **Random Forest Feature Importances**: Extract Gini importance from an ensemble of decision trees.
3. **Recursive Feature Elimination (RFECV)**: Trained a Random Forest estimator with 5-fold Stratified Cross-Validation (scoring='f1'). Iteratively drops the least important features.

All metrics are aggregated to produce a combined ranked dataframe (`reports/feature_importance_ranking.csv`).

---

## 4. Results

The RFECV algorithm determined that the optimal subset of features to maximize F1-score is **28 features** (out of the 38 predictor features). 

10 low-importance features were safely pruned to reduce model complexity without sacrificing (and potentially improving) predictive generalization.

### Top Selected Features (Examples)
Features with the strongest predictive signal include:
- `Contract` types (Month-to-month, Two year)
- `tenure` and its corresponding lifecycle cohorts (`TenureCohort`)
- `TotalCharges`, `MonthlyCharges`, `AvgMonthlyChargeRatio`
- `InternetService_Fiber optic`

### Pruned Dataset
The final dataset `selected_churn_data.csv` now consists of:
- `customerID`
- 28 Selected Predictor Features
- `Churn` (Target)

---

## 5. Next Task

**TASK 08 — Data Splitting & Stratified Cross-Validation Setup**
- Objective: Split the pruned data into training and hold-out test sets.
- Setup Stratified K-Fold CV for hyperparameter tuning.
- Address class imbalance considerations for training.
