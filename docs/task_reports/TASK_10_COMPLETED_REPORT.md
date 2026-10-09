# TASK 10 COMPLETED REPORT — Multi-Metric Model Evaluation & Imbalance Analysis

---

## 1. Task Metadata

| Attribute | Details |
|---|---|
| **Task ID** | `TASK_10` |
| **Task Name** | Multi-Metric Model Evaluation & Imbalance Analysis |
| **Execution Status** | **COMPLETED** |
| **Completion Date** | 2026-10-09 |
| **Target Folder** | `docs/task_reports/` |
| **Execution Script** | `scripts/run_model_evaluation.py` |
| **Evaluated Models** | 5 Candidate Classifiers |
| **Lead Developer** | Antigravity AI |

---

## 2. Objective & Scope

The objective of TASK 10 was to conduct an in-depth, multi-metric performance evaluation across all 5 candidate models on the held-out test set (`X_test`, $N = 1,404$), with a specific focus on:
1. Detailed per-class Classification Reports (Precision, Recall, F1-Score, Support for `No-Churn` and `Churn`).
2. Exact Confusion Matrix breakdowns (True Negatives, False Positives, False Negatives, True Positives).
3. The empirical impact of class imbalance (26.57% Churn minority class vs 73.43% majority class).
4. Threshold sensitivity and precision-recall trade-offs.

---

## 3. Baseline Classification Reports (Held-Out Test Set)

Evaluated on `X_test` with 1,031 `No-Churn` (0) and 373 `Churn` (1) instances:

### 3.1 Logistic Regression
```
              precision    recall  f1-score   support

    No-Churn       0.84      0.90      0.87      1031
       Churn       0.66      0.51      0.58       373

    accuracy                           0.80      1404
   macro avg       0.75      0.71      0.72      1404
weighted avg       0.79      0.80      0.79      1404
```
- **Confusion Matrix**: TN = 933, FP = 98, FN = 181, TP = 192
- **Metrics**: Accuracy = 80.13%, ROC-AUC = 0.8434, Churn Recall = 51.47%, Churn Precision = 66.21%

### 3.2 Decision Tree
```
              precision    recall  f1-score   support

    No-Churn       0.82      0.80      0.81      1031
       Churn       0.49      0.52      0.51       373

    accuracy                           0.73      1404
   macro avg       0.66      0.66      0.66      1404
weighted avg       0.73      0.73      0.73      1404
```
- **Confusion Matrix**: TN = 827, FP = 204, FN = 178, TP = 195
- **Metrics**: Accuracy = 72.79%, ROC-AUC = 0.6628, Churn Recall = 52.28%, Churn Precision = 48.87%

### 3.3 Random Forest
```
              precision    recall  f1-score   support

    No-Churn       0.83      0.91      0.87      1031
       Churn       0.66      0.50      0.57       373

    accuracy                           0.80      1404
   macro avg       0.75      0.70      0.72      1404
weighted avg       0.79      0.80      0.79      1404
```
- **Confusion Matrix**: TN = 935, FP = 96, FN = 188, TP = 185
- **Metrics**: Accuracy = 79.77%, ROC-AUC = 0.8286, Churn Recall = 49.60%, Churn Precision = 65.84%

### 3.4 XGBoost
```
              precision    recall  f1-score   support

    No-Churn       0.84      0.88      0.86      1031
       Churn       0.62      0.53      0.57       373

    accuracy                           0.79      1404
   macro avg       0.73      0.70      0.71      1404
weighted avg       0.78      0.79      0.78      1404
```
- **Confusion Matrix**: TN = 911, FP = 120, FN = 177, TP = 196
- **Metrics**: Accuracy = 78.85%, ROC-AUC = 0.8247, Churn Recall = 52.55%, Churn Precision = 62.03%

### 3.5 LightGBM
```
              precision    recall  f1-score   support

    No-Churn       0.84      0.90      0.87      1031
       Churn       0.65      0.51      0.57       373

    accuracy                           0.80      1404
   macro avg       0.74      0.71      0.72      1404
weighted avg       0.79      0.80      0.79      1404
```
- **Confusion Matrix**: TN = 926, FP = 105, FN = 181, TP = 192
- **Metrics**: Accuracy = 79.63%, ROC-AUC = 0.8344, Churn Recall = 51.47%, Churn Precision = 64.65%

---

## 4. Confusion Matrix Comparison Summary

| Model | TN | FP | FN | TP | False Negative Rate | Churn Detection Rate (Recall) |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 933 | 98 | 181 | 192 | 48.53% | 51.47% |
| **LightGBM** | 926 | 105 | 181 | 192 | 48.53% | 51.47% |
| **Random Forest** | 935 | 96 | 188 | 185 | 50.40% | 49.60% |
| **XGBoost** | 911 | 120 | 177 | 196 | 47.45% | 52.55% |
| **Decision Tree** | 827 | 204 | 178 | 195 | 47.72% | 52.28% |

---

## 5. Class Imbalance & Business Trade-off Analysis

### 5.1 The Imbalance Dilemma
In this dataset, only **26.57%** of customers churn. Standard loss functions optimize overall accuracy, which incentivizes the model to default toward the majority class (`No-Churn`). Consequently:
- **Accuracy is deceptively high** (~80%).
- **Minority recall is unacceptably low** (~51%), meaning nearly 1 out of every 2 churning customers is missed.

### 5.2 Asymmetric Business Costs
- **False Negative Cost**: High. Losing an actual churner results in lost Customer Lifetime Value (CLV, typically $70–$100/month for telecom subscribers).
- **False Positive Cost**: Low. Offering an unnecessary retention discount or outreach call costs a fraction of lost CLV (typically $10–$15 per promotion).
- **Conclusion**: The business requires models that maximize **Churn Recall** while preserving high discriminatory ability (**ROC-AUC**), which mandates cost-sensitive tuning (`class_weight='balanced'`) in TASK 11.

---

## 6. Generated Artifacts

| Artifact | File Path |
|---|---|
| Classification Reports JSON | [reports/model_evaluation/classification_reports.json](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/reports/model_evaluation/classification_reports.json) |
| Visual Classification Reports PNG | [reports/model_evaluation/baseline_classification_reports_all_models.png](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/reports/model_evaluation/baseline_classification_reports_all_models.png) |
| Baseline Confusion Matrices PNG | [reports/model_evaluation/baseline_confusion_matrices.png](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/reports/model_evaluation/baseline_confusion_matrices.png) |
| LR Threshold Analysis CSV | [reports/model_evaluation/lr_threshold_analysis.csv](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/reports/model_evaluation/lr_threshold_analysis.csv) |
| LightGBM Threshold Analysis CSV | [reports/model_evaluation/lgbm_threshold_analysis.csv](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/reports/model_evaluation/lgbm_threshold_analysis.csv) |

---

## 7. Next Task

The project advances to **TASK 11 — Hyperparameter Tuning (GridSearchCV / RandomizedSearchCV)**, where Logistic Regression and LightGBM undergo exhaustive and randomized searches with `roc_auc` scoring to overcome the minority class recall bottleneck.
