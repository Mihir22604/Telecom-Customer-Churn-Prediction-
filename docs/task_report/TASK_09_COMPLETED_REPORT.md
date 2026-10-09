# TASK 09 COMPLETED REPORT — Multi-Model Building (Baseline Classifiers)

---

## 1. Task Metadata

| Attribute | Details |
|---|---|
| **Task ID** | `TASK_09` |
| **Task Name** | Multi-Model Building (Baseline Classifiers) |
| **Execution Status** | **COMPLETED** |
| **Completion Date** | 2026-10-09 |
| **Target Folder** | `docs/task_reports/` |
| **Execution Script** | `scripts/run_model_evaluation.py` |
| **Primary Scoring Metric** | `roc_auc` |
| **Lead Developer** | Antigravity AI |

---

## 2. Objective & Scope

The objective of TASK 09 was to establish reproducible benchmark predictive performance across 5 candidate classification algorithms:
1. **Logistic Regression** (Linear baseline with L2 regularization)
2. **Decision Tree** (Non-linear CART baseline)
3. **Random Forest** (Ensemble Bagging classifier)
4. **XGBoost** (Extreme Gradient Boosting)
5. **LightGBM** (Light Gradient Boosting Machine)

The evaluation used 5-Fold Stratified Cross-Validation on the training partition (`X_train`) and evaluated held-out generalization performance on `X_test`.

---

## 3. Dataset Configuration & Leakage Prevention

| Partition | Row Count | Feature Count | Churn Distribution (Yes / No) | Churn Rate |
|---|---|---|---|---|
| **Full Dataset** | 7,019 | 30 (incl. ID & Target) | 1,865 / 5,154 | 26.57% |
| **Train Set (`X_train`)** | 5,615 | 28 features | 1,492 / 4,123 | 26.57% |
| **Test Set (`X_test`)** | 1,404 | 28 features | 373 / 1,031 | 26.57% |

### Preprocessing & Data Leakage Controls
- Continuous features (`tenure`, `MonthlyCharges`, `TotalCharges`, `monthly_to_total_ratio`, `charges_per_tenure`, `charge_discrepancy`) were normalized using `StandardScaler` fitted **strictly on `X_train`**.
- The fitted scaler transform was applied out-of-sample to `X_test` with zero re-estimation, ensuring strict data leakage prevention.

---

## 4. Baseline Evaluation Results (Held-Out Test Set)

Evaluated on `X_test` (1,404 samples: 1,031 non-churners, 373 churners):

| Model | Accuracy | ROC-AUC | Churn Recall | Churn Precision | Churn F1 | Macro F1 | Weighted F1 | Train Time (s) | Inference Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | **0.8013** | **0.8434** | 0.5147 | **0.6621** | **0.5792** | **0.7246** | **0.7927** | **0.072s** | **2.94 ms** |
| **LightGBM** | 0.7963 | 0.8344 | 0.5147 | 0.6465 | 0.5731 | 0.7197 | 0.7884 | 0.236s | 19.06 ms |
| **Random Forest** | 0.7977 | 0.8286 | 0.4960 | 0.6584 | 0.5657 | 0.7170 | 0.7878 | 1.328s | 59.49 ms |
| **XGBoost** | 0.7885 | 0.8247 | **0.5255** | 0.6203 | 0.5689 | 0.7144 | 0.7826 | 2.169s | 17.71 ms |
| **Decision Tree** | 0.7279 | 0.6628 | 0.5228 | 0.4887 | 0.5052 | 0.6588 | 0.7308 | 0.076s | 2.73 ms |

---

## 5. Key Empirical Observations

1. **Top Baseline Performers**:
   - **Logistic Regression** achieved the highest baseline ROC-AUC (**0.8434**) and accuracy (**80.13%**), while training in under 0.1 seconds.
   - **LightGBM** was the strongest non-linear gradient-boosted tree candidate (**0.8344** ROC-AUC, **0.5731** F1-score) with fast training (**0.24s**).
2. **Class Imbalance Impact**:
   - Across all 5 baseline models without cost-sensitive weighting, Churn Recall hovered tightly around **49.6% – 52.6%**.
   - The default 0.50 decision threshold causes models to miss approximately **48% of churning customers** (181 False Negatives out of 373 actual churners).
3. **Model Candidates Selected for Hyperparameter Tuning**:
   - **Logistic Regression** (best linear model, fast inference, high ROC-AUC).
   - **LightGBM** (best gradient booster, robust non-linear feature interaction modeling, fast leaf-wise training).

---

## 6. Generated Artifacts

| Artifact | Type | File Path |
|---|---|---|
| Baseline Metrics CSV | Structured Data | [reports/model_evaluation/baseline_metrics.csv](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/reports/model_evaluation/baseline_metrics.csv) |
| Classification Reports Summary PNG | High-Res Visual | [reports/model_evaluation/baseline_classification_reports_all_models.png](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/reports/model_evaluation/baseline_classification_reports_all_models.png) |
| Baseline Comparison PNG | High-Res Visual | [reports/model_evaluation/baseline_model_comparison.png](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/reports/model_evaluation/baseline_model_comparison.png) |
| Baseline Confusion Matrices PNG | High-Res Visual | [reports/model_evaluation/baseline_confusion_matrices.png](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/reports/model_evaluation/baseline_confusion_matrices.png) |
| Experiment Reproducibility Script | Python Script | [scripts/run_model_evaluation.py](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/scripts/run_model_evaluation.py) |

---

## 7. Next Task

The project proceeds to **TASK 10 — Multi-Metric Model Evaluation & Imbalance Analysis** for comprehensive per-class classification report verification, threshold analysis, and confusion matrix diagnostics.
