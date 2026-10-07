# TASK 09 — Multi-Model Building (Baseline Classifiers)
## Completion Report

**Project**: Telecom Customer Churn Prediction & Retention System
**Task ID**: TASK 09
**Status**: COMPLETED
**Executed On**: 2026-10-06

---

## 1. Objective

Train and evaluate 7 distinct machine learning algorithms using the stratified cross-validation strategy established in Task 08. Establish benchmark predictive performance to select models for hyperparameter tuning.

---

## 2. Models Evaluated
1. **Logistic Regression** (Linear)
2. **Decision Tree** (Tree)
3. **Random Forest** (Bagging)
4. **XGBoost** (Gradient Boosting)
5. **LightGBM** (Fast Leaf-wise Boosting)
6. **Support Vector Machine** (SVM - Margin Based)
7. **K-Nearest Neighbors** (KNN - Distance Based)

---

## 3. Implementation Details

- **Code**: `src/train.py`
- **CV Configuration**: 5-Fold Stratified Cross-Validation on `X_train`.
- **Metrics Tracked**: Accuracy, Precision, Recall, F1 Score, ROC-AUC.
- **Reporting**: Full metrics saved to `reports/baseline_metrics.csv`.
- **Notebook**: Walkthrough available in `notebooks/04_model_training_and_evaluation.ipynb`.

---

## 4. Baseline Results Summary (Validation F1 Scores)
*(Note: Refer to `baseline_metrics.csv` for exact precision/recall breakdown.)*

- The **Logistic Regression** and **LightGBM** models perform particularly well out-of-the-box.
- Tree-based models process quickly but require tuning to prevent overfitting and boost minority class recall.
- Recall is generally hindered across all baseline models due to class imbalance (~26% Churn rate). 

---

## 5. Next Task

**TASK 10 — Multi-Metric Model Evaluation & Imbalance Analysis**
- Deep dive into precision-recall tradeoffs.
- Apply SMOTE (Synthetic Minority Over-sampling Technique) or Class Weight adjustments to handle class imbalance.
