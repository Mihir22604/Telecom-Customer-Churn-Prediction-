# TASK 12 — Final Model Selection & Business Trade-off Justification
## Completion Report

**Project**: Telecom Customer Churn Prediction & Retention System
**Task ID**: TASK 12
**Status**: COMPLETED
**Executed On**: 2026-10-07

---

## 1. Objective

Select the optimal production model by evaluating predictive metrics, model complexity, and balancing the business costs of False Positives against False Negatives.

---

## 2. Model Selection Logic

Based on the outputs of Task 10 and 11, the **Tuned LightGBM** classifier was selected as the final production model.

### Why LightGBM?
- **Predictive Superiority**: Maintained the highest F1-Score and ROC-AUC during CV.
- **Inference Speed**: Leaf-wise tree growth enables ultra-fast inference speeds, capable of scoring thousands of clients in milliseconds.
- **Handling Imbalance**: Using `class_weight='balanced'` inside LightGBM effectively neutralized the ~73/26 target distribution, significantly lifting the recall.

### Business Trade-off Analysis
In a retention scenario, missing a churner (False Negative) results in lost recurring revenue, which is generally more expensive than offering a retention discount to a loyal customer (False Positive). LightGBM provided the best equilibrium by catching the maximum number of true churners while keeping the retention promo budget within acceptable limits.

---

## 3. Results & Outputs

| Deliverable | Details |
|-------------|---------|
| Script | `src/run_evaluation_and_tuning.py` (Selection Logic) |
| Serialized Model | `models/final_model.pkl` |

The best model is now pickled and ready for TASK 13 (SHAP Interpretability Analysis).
