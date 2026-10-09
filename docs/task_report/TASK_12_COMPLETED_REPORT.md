# TASK 12 COMPLETED REPORT — Final Model Selection & Business Trade-off Justification

---

## 1. Task Metadata

| Attribute | Details |
|---|---|
| **Task ID** | `TASK_12` |
| **Task Name** | Final Model Selection & Business Trade-off Justification |
| **Execution Status** | **COMPLETED** |
| **Completion Date** | 2026-10-09 |
| **Target Folder** | `docs/task_reports/` |
| **Execution Script** | `scripts/run_model_evaluation.py` |
| **Final Selected Model** | **LightGBM (Tuned)** |
| **Selection Criterion** | **Higher Churn Recall with comparable AUC** |
| **Lead Developer** | Antigravity AI |

---

## 2. Objective & Scope

The objective of TASK 12 was to finalize the enterprise production model by comparing the two tuned finalist architectures—**Logistic Regression (Tuned)** and **LightGBM (Tuned)**—using empirical validation metrics, latency benchmarks, and business ROI trade-off analysis.

---

## 3. Head-to-Head Empirical Comparison (Held-Out Test Set)

Evaluated on `X_test` ($N = 1,404$; 1,031 non-churners, 373 churners):

| Metric | Logistic Regression (Tuned) | LightGBM (Tuned) | Absolute Difference | Winner |
|---|---|---|---|---|
| **ROC-AUC** | **0.8435** | 0.8424 | -0.0011 (negligible) | *Tie / LR marginal* |
| **Churn Recall** | 0.5147 | **0.8177** | **+0.3030 (+30.30 pp)** | **LightGBM** |
| **Churn F1-Score** | 0.5792 | **0.6250** | **+0.0458 (+4.58 pp)** | **LightGBM** |
| **Accuracy** | **0.8013** | 0.7393 | -0.0620 | Logistic Regression |
| **Churn Precision** | **0.6621** | 0.5058 | -0.1563 | Logistic Regression |
| **Macro F1** | **0.7246** | 0.7126 | -0.0120 | Logistic Regression |
| **Weighted F1** | **0.7927** | 0.7537 | -0.0390 | Logistic Regression |
| **True Positives (Rescued)** | 192 | **305** | **+113 churners intercepted** | **LightGBM** |
| **False Negatives (Missed)** | 181 | **68** | **-113 missed customers** | **LightGBM** |
| **False Positives** | **98** | 298 | +200 promotions sent | Logistic Regression |
| **Inference Latency (Batch)** | 9.69 ms | 47.08 ms | +37.39 ms (both $\ll 100$ ms SLA) | *Passes SLA* |

---

## 4. Selection Rationale: Evidence-Based Analysis

### 4.1 Criterion: Higher Churn Recall with Comparable AUC
- **ROC-AUC Equivalence**: The ROC-AUC difference between Logistic Regression (0.8435) and LightGBM (0.8424) is **0.0011** (less than 0.13%). Both models possess virtually identical global class-ranking separation capacity.
- **Massive Recall Superiority**: LightGBM captures **81.77%** of actual churners compared to only **51.47%** for Logistic Regression.
  - Miss rate for Logistic Regression: **48.53%** (almost half of churners walk away undetected).
  - Miss rate for LightGBM: **18.23%** (more than 4 out of 5 churners are flagged).

### 4.2 Economic Cost-Benefit Simulation
Assuming industry-standard telecom parameters:
- **Cost of a False Negative ($C_{\text{FN}}$)**: Lost Customer Lifetime Value (CLV) $\approx \$1,200/\text{year}$ ($\$100/\text{month}$).
- **Cost of a False Positive ($C_{\text{FP}}$)**: Cost of proactive retention incentive or discount $\approx \$15$.
- **Cost of True Positive ($C_{\text{TP}}$)**: Retention incentive cost ($\$15$) with successful retention rate $\eta = 50\%$: Net value saved $\approx \$585$.

#### Performance on 1,404 Test Customers (373 Churners):
| Metric | Logistic Regression (Tuned) | LightGBM (Tuned) | Advantage |
|---|---|---|---|
| **Churners Rescued (TP)** | 192 | **305** | **+113 customers** |
| **Gross Value Rescued** | $\$112,320$ | **$\$178,425$** | **+$\$66,105$** |
| **Promo Incentive Budget Spent** | $\$4,350$ | $\$9,045$ | $-\$4,695$ |
| **Lost CLV from Misses (FN)** | $\$217,200$ | **$\$81,600$** | **$\$135,600$ less loss** |
| **Net Financial Bottom Line** | **$-\$109,230$** | **$+\$87,780$** | **+$\$197,010$ Net Benefit** |

The additional $\$4,695$ spent on retention outreach for False Positives is dwarfed by the **$\$135,600$ reduction in lost revenue**. LightGBM delivers vastly superior enterprise ROI.

---

## 5. Production Viability & Latency SLA

- **Single Record Latency**: $\approx 0.8\text{ ms} - 1.5\text{ ms}$, exceeding production SLA requirements ($< 100\text{ ms}$).
- **Batch Evaluation Latency**: $47.08\text{ ms}$ for 1,404 records ($\approx 33.5\ \mu\text{s}$ per record).
- **Scalability**: Gradient-boosted decision trees execute rapidly via C++ compiled backend in LightGBM, with minimal RAM overhead ($\sim 15\text{ MB}$).

---

## 6. Generated Artifacts & Model Deliverables

| Deliverable | Description | File Path |
|---|---|---|
| Final ROC Comparison PNG | ROC Curves (Baseline vs Tuned) | [reports/model_evaluation/roc_curves_final_comparison.png](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/reports/model_evaluation/roc_curves_final_comparison.png) |
| Selected Models Confusion Matrices PNG | 2x2 side-by-side matrices | [reports/model_evaluation/selected_models_confusion_matrices.png](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/reports/model_evaluation/selected_models_confusion_matrices.png) |
| Experiment Metadata JSON | Full configuration and outcome audit | [reports/model_evaluation/experiment_metadata.json](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/reports/model_evaluation/experiment_metadata.json) |
| Serialized Final Model | Production pipeline bundle | [models/final_model.pkl](file:///c:/Users/mihir/Desktop/Telecom%20Customer%20Churn%20Prediction%20&%20Retention%20System/models/final_model.pkl) |

---

## 7. Next Task

With **LightGBM (Tuned)** established as the champion production model, the project proceeds to **TASK 13 — Model Explainability (SHAP & Feature Importance Analysis)** to compute global and local feature attributions.
