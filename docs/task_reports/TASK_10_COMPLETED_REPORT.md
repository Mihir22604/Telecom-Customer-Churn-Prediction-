# TASK 10 — Multi-Metric Model Evaluation & Imbalance Analysis
## Completion Report

**Project**: Telecom Customer Churn Prediction & Retention System
**Task ID**: TASK 10
**Status**: COMPLETED
**Executed On**: 2026-10-07

---

## 1. Objective

Rigorously evaluate trained models using multiple evaluation metrics beyond accuracy, heavily scrutinizing minority class predictions to highlight the effects of class imbalance.

---

## 2. Evaluation Strategy

The candidate models were evaluated on the held-out `X_test` subset using:
- **Precision, Recall, F1 Score**: Vital for isolating performance on the `Churn = Yes` class.
- **ROC-AUC & PR-AUC**: For probability distribution separation.
- **Confusion Matrices**: To directly visualize False Negatives (Costly Misses) vs False Positives (Unnecessary Retention Promos).

---

## 3. Results & Outputs

| Deliverable | Details |
|-------------|---------|
| Script | `src/evaluation.py` |
| ROC Curve | `reports/figures/model_roc_comparison.png` |
| PR Curve | `reports/figures/model_pr_comparison.png` |
| Confusion Matrix | `reports/figures/confusion_matrices.png` |

**Key Finding**: Naive baselines struggled significantly with False Negatives (low recall) despite having over 80% accuracy. The models missed over 40% of actual churners because the training signal was overwhelmed by the negative class. This mandates hyperparameter adjustments like `class_weight='balanced'` in the next task.
