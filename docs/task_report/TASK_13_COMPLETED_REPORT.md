# TASK 13 — Model Serialization & Pipeline Packaging
## Completion Report

**Project**: Telecom Customer Churn Prediction & Retention System  
**Task ID**: TASK 13  
**Status**: COMPLETED  
**Executed On**: 2026-10-07  
**Assignment Requirement**: Requirement #12 ("Save the Model")

---

## 1. Objective

Serialize the final production machine learning model and data preprocessing transformers into persistent, production-ready binary artifacts using `joblib`. Package both standalone components and a unified `sklearn.pipeline.Pipeline`, generate rich model metadata with training metrics and schema, and verify deserialization prediction consistency.

---

## 2. Implementation Strategy

### 2.1 Stateful Preprocessing & Pipeline Architecture
- Implemented `FeatureScaler` (with backwards compatibility alias `PreprocessingPipeline`) in `src/pipeline.py`, inheriting from scikit-learn's `BaseEstimator` and `TransformerMixin`.
- Fits `StandardScaler` strictly on continuous numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`, `AvgMonthlyChargeRatio`, `MonthlyToTotalRatio`, `TotalServicesCount`) using training set statistics only, preventing data leakage.
- Categorical and binary one-hot encoded features pass through unchanged.
- Ensured serialization is bound to the persistent `src.pipeline` namespace, preventing `__main__` pickling errors when artifacts are unpickled across external entrypoints (FastAPI API server, Streamlit UI, pytest runner).

### 2.2 Unified Pipeline & Inference Packaging
- Constructed a unified `sklearn.pipeline.Pipeline`:
  ```python
  unified_pipeline = Pipeline([
      ('scaler', prep_pipeline),
      ('classifier', final_model)
  ])
  ```
- Created a packaged inference bundle (`full_inference_pipeline.joblib`) containing the unified pipeline, individual components, and the verified list of 28 feature names for serving resilience.

### 2.3 Model Metadata Documentation
- Emitted `models/model_metadata.json` documenting:
  - Model identification: `LightGBMChurnClassifier` v`1.0.0`
  - UTC training timestamp in ISO 8601 format
  - Algorithm details and tuned hyperparameters (`class_weight='balanced'`, `colsample_bytree=0.7`, `learning_rate=0.01`, `max_depth=5`, `n_estimators=100`, `subsample=0.8`)
  - Exact 28 feature names and schema
  - Comprehensive test set evaluation metrics
  - Runtime environment dependencies: `scikit-learn 1.6.1`, `lightgbm 4.7.0`, `joblib 1.4.2`

---

## 3. Results & Outputs

### 3.1 Saved Artifacts

| Deliverable | File Path | Details |
|---|---|---|
| **Pipeline Module** | `src/pipeline.py` | `FeatureScaler` / `PreprocessingPipeline` transformer |
| **Serialization Script** | `src/serialization.py` | Complete training, packaging, and serialization workflow |
| **Best Churn Model** | `models/best_churn_model.joblib` | Tuned LightGBM classifier weights |
| **Preprocessing Pipeline** | `models/preprocessing_pipeline.joblib` | Fitted `FeatureScaler` transformer |
| **Unified Sklearn Pipeline** | `models/unified_churn_pipeline.joblib` | End-to-end `sklearn.pipeline.Pipeline` |
| **Full Inference Bundle** | `models/full_inference_pipeline.joblib` | Multi-key bundle for flexible API serving |
| **Model Metadata** | `models/model_metadata.json` | Version 1.0.0 metadata, hyperparams, & metrics |
| **Unit Test Suite** | `tests/test_serialization.py` | Deserialization and prediction invariance tests |

### 3.2 Production Model Performance Metrics (Held-out Test Set)

| Metric | Score | Business Interpretation |
|---|---|---|
| **Recall** | **0.7989** (79.9%) | Successfully detects ~80% of actual churners (minimizes False Negatives) |
| **ROC-AUC** | **0.8415** | Strong separability between churn and non-churn risk distributions |
| **F1-Score** | **0.6221** | Optimal harmonic mean balancing recall and precision under 73/27 class imbalance |
| **Accuracy** | **0.7422** | Robust overall classification rate |
| **Precision** | **0.5094** | Targeted identification ensuring acceptable retention promotion ROI |

---

## 4. Verification & Validation

1. **Prediction Consistency Invariance**:
   - Tested predictions from raw in-memory models against:
     - Separated artifacts (`preprocessing_pipeline.joblib` + `best_churn_model.joblib`)
     - Bundled dictionary artifact (`full_inference_pipeline.joblib`)
     - Standalone unified pipeline (`unified_churn_pipeline.joblib`)
   - Predictions (`y_pred`) and class probabilities (`y_prob`) matched identically with numerical tolerance `atol = 1e-6`.
2. **Probability Boundedness**:
   - Verified all predicted probabilities satisfy $0.0 \le P(\text{Churn}) \le 1.0$.
3. **Automated Unit Testing**:
   - `python tests/test_serialization.py`: **3 / 3 unit tests PASSED**.
   - `python -m unittest discover tests`: **37 / 37 unit tests PASSED across entire repository**.

---

## 5. Next Steps

With all machine learning models, preprocessing pipelines, and metadata packaged and verified, the project is ready for **TASK 14 — FastAPI Backend API Development**.
