# How to Run the Telecom Customer Churn Prediction & Retention System

This guide provides end-to-end instructions for running the complete system locally, including the Machine Learning pipeline, the FastAPI REST microservice, the interactive Streamlit Web UI, and the automated test suite.

---

## 1. System Architecture Overview

```
                      ┌──────────────────────────────────────┐
                      │    Streamlit Web User Interface      │
                      │       (http://localhost:8501)        │
                      └──────────────────┬───────────────────┘
                                         │ HTTP REST (JSON)
                                         ▼
                      ┌──────────────────────────────────────┐
                      │        FastAPI REST Service          │
                      │       (http://localhost:8000)        │
                      │         Swagger: /docs               │
                      └──────────────────┬───────────────────┘
                                         │ In-Memory Pipeline
                                         ▼
                      ┌──────────────────────────────────────┐
                      │      LightGBM Churn Classifier      │
                      │        (models/*.joblib)             │
                      └──────────────────────────────────────┘
```

---

## 2. Prerequisites & Environment Setup

### 2.1 Requirements
- **Python**: Version 3.10, 3.11, or 3.12
- **Operating System**: Windows, macOS, or Linux

### 2.2 Clone or Navigate to the Project Root
Open PowerShell (or your terminal) and ensure you are in the project root directory:
```powershell
cd "c:\Users\mihir\Desktop\Telecom Customer Churn Prediction & Retention System"
```

### 2.3 Verify / Activate Python Environment
If using Anaconda / Conda:
```powershell
conda activate base
```
Or if using a standard Python virtual environment:
```powershell
# Create venv (if not already created)
python -m venv venv

# Activate on Windows PowerShell
.\venv\Scripts\Activate.ps1

# Activate on Linux / macOS
source venv/bin/activate
```

---

## 3. Step-by-Step Execution Guide

### Option A: Running the Entire System (Recommended)

To run the complete interactive stack, start both the **FastAPI Backend** and the **Streamlit Frontend** in separate terminal windows.

---

### Step 1: Start the FastAPI Backend REST Service

Open Terminal Window 1:

```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Verification:
1. **Interactive Swagger Documentation**: Open your browser and navigate to:
   [http://localhost:8000/docs](http://localhost:8000/docs)
2. **ReDoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
3. **Health Check Endpoint**:
   ```powershell
   curl http://localhost:8000/health
   ```
   *Expected Response:*
   ```json
   {
     "status": "healthy",
     "model_loaded": true,
     "model_version": "1.0.0",
     "uptime_seconds": 5.2,
     "timestamp": "2026-10-07T..."
   }
   ```

---

### Step 2: Start the Streamlit Web User Interface

Open Terminal Window 2:

```powershell
streamlit run app/streamlit_app.py
```

#### Accessing the Web Dashboard:
- The browser will automatically open at:
  [http://localhost:8501](http://localhost:8501)
- If the browser does not open automatically, copy and paste the URL into your browser.

#### Features Available in the Web UI:
1. **Sidebar Connection Monitor**: Displays real-time API connection status (green dot when FastAPI is running on port 8000).
2. **Preset Customer Profiles**: Use the dropdown in the sidebar to instantly load:
   - *High Churn Risk Customer* (Month-to-month, Fiber optic, No Tech Support)
   - *Low Churn Risk Customer* (2-Year contract, DSL, Full Security Bundle)
   - *Moderate Churn Risk Customer* (1-Year contract, Partial Services)
3. **Tab 1: Single Customer Risk Assessment**:
   - Customize demographics, services, and billing sliders.
   - Click **"🚀 Predict Customer Churn Risk"**.
   - View probability gauges, decision badges, inference latencies, and actionable retention recommendations.
4. **Tab 2: Batch Customer Prediction**:
   - Upload any custom customer CSV or click **"Load Cleaned Sample Batch (25 records)"**.
   - Click **"⚡ Score All Customers"** to evaluate the cohort.
   - Download the scored results as `telecom_scored_customers.csv`.
5. **Tab 3: Model Performance & Architecture**:
   - Inspect production model metrics (Recall, ROC-AUC, F1-Score, Accuracy, Precision).
   - Review tuned LightGBM hyperparameters and top predictive features.

> **Note on Hybrid Resilience**: If the FastAPI server is not running, the Streamlit app automatically falls back to the local in-memory inference engine (`src/inference.py`), ensuring uninterrupted operation.

---

### Step 3: Run the Automated Test Suite

To verify system integrity, data quality, and API contract compliance, run the complete unit and integration test suite:

```powershell
python -m unittest discover tests
```

*Expected Result:*
```text
Ran 53 tests in 26.848s

OK
```

#### Running Specific Test Suites:
- **API Integration Tests only:**
  ```powershell
  python tests/test_api.py
  ```
- **Data Quality & Invariants Tests only:**
  ```powershell
  python tests/test_data_quality.py
  ```
- **Model Serialization & Pipeline Packaging Tests only:**
  ```powershell
  python tests/test_serialization.py
  ```

---

## 4. Testing the API Directly via CLI (cURL / PowerShell)

### Example 1: Predict Churn for a High-Risk Customer

#### Using cURL:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "customerID": "CUST-HIGH-RISK",
       "gender": "Female",
       "SeniorCitizen": 0,
       "Partner": "No",
       "Dependents": "No",
       "tenure": 2.0,
       "PhoneService": "Yes",
       "MultipleLines": "No",
       "InternetService": "Fiber optic",
       "OnlineSecurity": "No",
       "OnlineBackup": "No",
       "DeviceProtection": "No",
       "TechSupport": "No",
       "StreamingTV": "Yes",
       "StreamingMovies": "Yes",
       "Contract": "Month-to-month",
       "PaperlessBilling": "Yes",
       "PaymentMethod": "Electronic check",
       "MonthlyCharges": 95.50,
       "TotalCharges": 191.00
     }'
```

#### Response:
```json
{
  "customerID": "CUST-HIGH-RISK",
  "churn_prediction": 1,
  "churn_label": "Yes",
  "churn_probability": 0.7252,
  "risk_score_percent": 72.52,
  "risk_level": "High",
  "retention_recommendations": [
    "Contract Risk: Offer a 1-Year or 2-Year contract with an introductory 15% billing discount.",
    "Payment Friction Risk: Encourage switching to Automatic Bank Transfer or Credit Card with a one-time $10 bill credit.",
    "Anchor Service Opportunity: Offer a complimentary 3-month trial of TechSupport and OnlineSecurity to boost retention lock-in.",
    "Early Lifecycle Alert: Assign to proactive customer success onboarding to resolve early experience friction.",
    "Price Sensitivity Risk: Conduct plan audit to recommend an optimized service bundle tailored to actual usage.",
    "Fiber Optic Service Check: Schedule proactive line quality and speed verification to prevent tech dissatisfaction churn."
  ],
  "model_version": "1.0.0",
  "latency_ms": 32.4
}
```

---

### Example 2: Inspect Model Metadata

```bash
curl http://localhost:8000/metadata
```

#### Response:
```json
{
  "model_name": "LightGBMChurnClassifier",
  "model_version": "1.0.0",
  "training_timestamp": "2026-10-07T06:20:19Z",
  "algorithm": "LightGBM",
  "feature_count": 28,
  "feature_names": [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", 
    "PaperlessBilling", "MonthlyCharges", "TotalCharges", ...
  ],
  "test_set_metrics": {
    "Accuracy": 0.742165,
    "Precision": 0.509402,
    "Recall": 0.798928,
    "F1_Score": 0.622129,
    "ROC_AUC": 0.841495
  }
}
```

---

## 5. Re-training or Re-serializing Models (Optional)

If you modify feature engineering or model hyperparameters, you can regenerate all serialized model artifacts with a single command:

```powershell
python -m src.serialization
```

This will automatically:
1. Load and split the dataset (`80% Train / 20% Test`).
2. Fit the continuous feature scaler (`FeatureScaler`).
3. Re-train the tuned LightGBM classifier.
4. Evaluate test metrics (`Recall: 79.9%`, `ROC-AUC: 0.8415`).
5. Export artifacts to `models/`:
   - `models/best_churn_model.joblib`
   - `models/preprocessing_pipeline.joblib`
   - `models/unified_churn_pipeline.joblib`
   - `models/full_inference_pipeline.joblib`
   - `models/model_metadata.json`
6. Validate deserialization prediction invariance.

---

## 6. Troubleshooting & FAQs

### Q1: `ModuleNotFoundError: No module named 'src'`
**Solution**: Always run commands using the `-m` flag from the repository root directory:
```powershell
python -m src.serialization
python -m uvicorn api.main:app
python -m unittest discover tests
```

### Q2: Port 8000 or 8501 is already in use
**Solution**: Specify alternative ports:
- For FastAPI: `python -m uvicorn api.main:app --port 8080`
- For Streamlit: `streamlit run app/streamlit_app.py --server.port 8502`
- In Streamlit, enter the updated FastAPI URL (e.g. `http://localhost:8080`) in the sidebar input box.

### Q3: Streamlit shows "🟠 API Offline (Using In-Memory Fallback)"
**Explanation**: This is normal if you haven't started the FastAPI server. The Streamlit app has a built-in hybrid fallback that executes predictions directly using the in-memory ML model. To connect to the REST API, simply start FastAPI on port 8000 in a separate terminal.
