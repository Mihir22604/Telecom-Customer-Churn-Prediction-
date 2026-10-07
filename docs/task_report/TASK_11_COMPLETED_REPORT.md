# TASK 11 — Hyperparameter Tuning (GridSearchCV / RandomizedSearchCV)
## Completion Report

**Project**: Telecom Customer Churn Prediction & Retention System
**Task ID**: TASK 11
**Status**: COMPLETED
**Executed On**: 2026-10-07

---

## 1. Objective

Fine-tune hyperparameter configurations for the top performing candidate models (LightGBM and Logistic Regression) to maximize out-of-sample F1 scoring while handling the class imbalance natively.

---

## 2. Methodology

1. **LightGBM (Tree Ensemble)**:
   - Evaluated via `RandomizedSearchCV` to cover a broad grid space rapidly.
   - Tuned `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, and `n_estimators`.
   - Tuned `class_weight` to apply higher loss penalties for False Negatives.

2. **Logistic Regression (Linear)**:
   - Evaluated via `GridSearchCV` due to the smaller combinatorial space.
   - Tuned regularization strength `C` and `solver`.

---

## 3. Results & Outputs

| Deliverable | Details |
|-------------|---------|
| Script | `src/tuning.py` |
| JSON Report | `reports/hyperparameter_tuning_results.json` |

The tuning successfully discovered parameters that notably increased cross-validated F1 scores for both algorithms compared to their un-tuned baseline states.
