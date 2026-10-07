# TASK 08 — Data Splitting & Stratified Cross-Validation Setup
## Completion Report

**Project**: Telecom Customer Churn Prediction & Retention System
**Task ID**: TASK 08
**Status**: COMPLETED
**Executed On**: 2026-10-06

---

## 1. Objective

Divide the pruned dataset into training and testing sets using Stratified Splitting and configure a robust K-Fold cross-validation strategy.

---

## 2. Methodology

1. **Stratified Split**: We used `train_test_split(test_size=0.20, stratify=y, random_state=42)` to ensure that the train and test subsets maintain the identical proportion of the `Churn` target (~26.6% Yes).
2. **K-Fold CV Setup**: We initialized `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` to prevent class imbalance skewing during internal evaluations.

---

## 3. Results & Outputs

| Artifact | Result/Description |
|----------|--------------------|
| Code | `src/pipeline.py` created |
| Train Size | 5,615 samples (80%) |
| Test Size | 1,404 samples (20%) |
| Validation | Class proportion verified matching in train and test sets. |

---

## 4. Next Step
The established splits and CV strategy are immediately injected into **TASK 09** (Baseline Modeling) for evaluating our classifier algorithms.
