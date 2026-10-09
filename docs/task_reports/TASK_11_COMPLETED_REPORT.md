# TASK 11 COMPLETED REPORT — Hyperparameter Tuning (GridSearchCV / RandomizedSearchCV)

---

## 1. Task Metadata

| Attribute | Details |
|---|---|
| **Task ID** | `TASK_11` |
| **Task Name** | Hyperparameter Tuning (GridSearchCV / RandomizedSearchCV) |
| **Execution Status** | **COMPLETED** |
| **Completion Date** | 2026-10-09 |
| **Target Folder** | `docs/task_reports/` |
| **Execution Script** | `scripts/run_model_evaluation.py` |
| **Primary Scoring Metric** | `roc_auc` (5-Fold Stratified Cross-Validation) |
| **Tuned Models** | Logistic Regression, LightGBM |
| **Lead Developer** | Antigravity AI |

---

## 2. Objective & Scope

The objective of TASK 11 was to conduct disciplined hyperparameter tuning on the top two candidate architectures:
1. **Logistic Regression**: Linear regularized classifier optimized via exhaustive **`GridSearchCV`**.
2. **LightGBM**: Gradient-boosted decision tree ensemble optimized via **`RandomizedSearchCV`** (60 iterations).

Both tuning searches targeted **`roc_auc`** as the primary cross-validation optimization metric to preserve global ranking discrimination while resolving class imbalance.

---

## 3. Tuning Methodology & Search Configurations

### 3.1 Logistic Regression (`GridSearchCV`)
- **Search Space**: 54 candidate hyperparameter combinations.
  - `C`: `[0.01, 0.1, 1.0]`
  - `penalty`: `['l1', 'l2', 'elasticnet']`
  - `solver`: `['saga', 'liblinear', 'lbfgs']`
  - `l1_ratio`: `[0.3, 0.5, 0.7]` (for ElasticNet)
  - `class_weight`: `[None, 'balanced']`
- **CV Budget**: 54 candidates $\times$ 5 folds = 270 fits.
- **Tuning Time**: 310.26 seconds.
- **Best Cross-Validation ROC-AUC**: **0.8503** (0.850280).
- **Optimal Hyperparameters**:
  ```python
  {
      "C": 1.0,
      "penalty": "elasticnet",
      "solver": "saga",
      "l1_ratio": 0.3,
      "class_weight": None,
      "max_iter": 2000
  }
  ```

### 3.2 LightGBM (`RandomizedSearchCV`)
- **Search Space**: 60 randomized iterations over broad distributions.
  - `n_estimators`: `[100, 200, 300]`
  - `learning_rate`: `[0.01, 0.05, 0.1]`
  - `max_depth`: `[3, 5, 7, -1]`
  - `num_leaves`: `[15, 31, 63, 64]`
  - `subsample`: `[0.7, 0.8, 0.9, 1.0]`
  - `colsample_bytree`: `[0.7, 0.8, 0.9, 1.0]`
  - `reg_alpha`: `[0.0, 0.1, 0.5]`
  - `reg_lambda`: `[0.0, 0.1, 0.5]`
  - `min_child_samples`: `[20, 30, 50]`
  - `class_weight`: `[None, 'balanced']`
- **CV Budget**: 60 iterations $\times$ 5 folds = 300 fits (`random_state=42`).
- **Tuning Time**: 122.47 seconds.
- **Best Cross-Validation ROC-AUC**: **0.8472** (0.847196).
- **Optimal Hyperparameters**:
  ```python
  {
      "n_estimators": 300,
      "num_leaves": 64,
      "max_depth": 3,
      "learning_rate": 0.01,
      "subsample": 1.0,
      "colsample_bytree": 0.9,
      "reg_lambda": 0.5,
      "reg_alpha": 0.0,
      "min_child_samples": 50,
      "class_weight": "balanced"
  }
  ```

---

## 4. Before vs. After Tuning Performance (Held-Out Test Set)

Evaluated on `X_test` ($N = 1,404$):

| Model & State | Accuracy | ROC-AUC | Churn Recall | Churn Precision | Churn F1 | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|---|---|
| **Logistic Regression (Baseline)** | 0.8013 | 0.8434 | 0.5147 | 0.6621 | 0.5792 | 0.7246 | 0.7927 |
| **Logistic Regression (Tuned)** | 0.8013 | 0.8435 | 0.5147 | 0.6621 | 0.5792 | 0.7246 | 0.7927 |
| *Logistic Regression Delta* | *+0.0000* | *+0.0001* | *+0.0000* | *+0.0000* | *+0.0000* | *+0.0000* | *+0.0000* |
|---|---|---|---|---|---|---|---|
| **LightGBM (Baseline)** | 0.7963 | 0.8344 | 0.5147 | 0.6465 | 0.5731 | 0.7197 | 0.7884 |
| **LightGBM (Tuned)** | 0.7393 | **0.8424** | **0.8177** | 0.5058 | **0.6250** | 0.7126 | 0.7537 |
| *LightGBM Delta* | *-0.0570* | *+0.0080* | ***+0.3030*** | *-0.1407* | ***+0.0519*** | *-0.0071* | *-0.0347* |

---

## 5. Tuned LightGBM Detailed Classification Report

```
              precision    recall  f1-score   support

    No-Churn       0.92      0.71      0.80      1031
       Churn       0.51      0.82      0.62       373

    accuracy                           0.74      1404
   macro avg       0.71      0.76      0.71      1404
weighted avg       0.81      0.74      0.75      1404
```

### Confusion Matrix Comparison (Baseline vs Tuned LightGBM)
- **Baseline LightGBM**:
  - TN = 926, FP = 105, FN = 181, TP = 192
  - Churn Miss Rate: **48.5%** (181 missed customers)
- **Tuned LightGBM (`class_weight='balanced'`)**:
  - TN = 733, FP = 298, FN = 68, TP = **305**
  - Churn Miss Rate: **18.2%** (only 68 missed customers)
  - **Net Gain**: **113 additional churning customers successfully intercepted!**

---

## 6. Key Empirical Takeaways

1. **LightGBM Churn Recall Surged by +30.3 Percentage Points**:
   - By tuning with `class_weight='balanced'` and shallow tree depth (`max_depth=3`, `learning_rate=0.01`), LightGBM boosted churn recall from **51.47% to 81.77%**.
   - Churn F1-Score increased from **0.5731 to 0.6250**.
   - ROC-AUC improved from **0.8344 to 0.8424**, demonstrating genuine distributional ranking improvement.
2. **Logistic Regression Saturation**:
   - Logistic Regression was already well-calibrated in its baseline configuration; ElasticNet penalty with `l1_ratio=0.3` produced marginal ROC-AUC gain (+0.0001) but remained locked at 51.47% recall under default thresholding.

---

## 7. Generated Visualizations & Artifacts

| Deliverable | Description | File Path |
|---|---|---|
| LR Before/After Tuning PNG | 4-panel visual comparison | [reports/model_evaluation/logistic_regression_before_after_tuning.png](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/reports/model_evaluation/logistic_regression_before_after_tuning.png) |
| LightGBM Before/After Tuning PNG | 4-panel visual comparison | [reports/model_evaluation/lightgbm_before_after_tuning.png](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/reports/model_evaluation/lightgbm_before_after_tuning.png) |
| Side-by-Side Model Comparison PNG | LR vs LightGBM multi-metric bars | [reports/model_evaluation/logistic_vs_lightgbm_before_after_tuning.png](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/reports/model_evaluation/logistic_vs_lightgbm_before_after_tuning.png) |
| Hyperparameter Search Results CSV | Full iterations log | [reports/model_evaluation/hyperparameter_search_results.csv](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/reports/model_evaluation/hyperparameter_search_results.csv) |
| Tuned Metrics CSV | Held-out test performance table | [reports/model_evaluation/tuned_model_metrics.csv](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/reports/model_evaluation/tuned_model_metrics.csv) |

---

## 8. Next Task

The project proceeds to **TASK 12 — Final Model Selection & Business Trade-off Justification** for evidence-based selection and cost-benefit justification.
