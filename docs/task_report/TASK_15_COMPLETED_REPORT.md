# TASK 15 — Streamlit Web UI Development
## Completion Report

**Project**: Telecom Customer Churn Prediction & Retention System  
**Task ID**: TASK 15  
**Status**: COMPLETED  
**Executed On**: 2026-10-07  
**Assignment Requirement**: Requirement #14 ("Build a Web UI")

---

## 1. Objective

Build an interactive, modern web dashboard using Streamlit that enables customer success managers, retention teams, and executive stakeholders to evaluate customer churn risks in real-time, inspect driving risk factors, generate tailored retention strategies, and execute batch customer scoring.

---

## 2. Implementation Strategy & Architecture

### 2.1 Web Application Structure (`app/streamlit_app.py`)
- Configured wide-layout Streamlit application with custom CSS styling, metric cards, shadows, and color-coded risk status badges:
  - **High Risk**: Crimson Red (`badge-high`, $\ge 65\%$)
  - **Moderate Risk**: Amber/Yellow (`badge-medium`, $35\% - 65\%$)
  - **Low Risk**: Emerald Green (`badge-low`, $< 35\%$)

### 2.2 System Sidebar & Health Integration
- **Live Connection Monitor**: Probes FastAPI `/health` endpoint to show real-time service status, model version, and uptime.
- **Fail-safe Hybrid Engine**: Automatically queries the FastAPI backend via HTTP; if the backend service is offline, it seamlessly fails over to the local in-memory `ChurnPredictor` so user workflows are never interrupted.
- **Direct Custom Attribute Controls**: Comprehensive sidebar and panel input controls allow users to enter full custom customer profiles directly with validation bounds and immediate reactive updates.

### 2.3 Dashboard Modules

#### Tab 1 — Single Customer Risk Assessment
- Organized into 3 logical columns:
  1. **Demographics**: Gender, Senior Citizen status, Partner, Dependents.
  2. **Subscribed Services**: Phone Service, Multiple Lines, Internet Tier, Online Security, Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies.
  3. **Account & Billing Terms**: Tenure slider (0–72m), Contract type, Paperless Billing, Payment Method, Monthly Charges ($18–$125), Total Charges input.
- **Results Panel**:
  - 4 KPI metric cards: Risk Classification, Churn Probability Score, Model Decision, Inference Latency.
  - Interactive risk progress bar.
  - Actionable Retention Action Plan with targeted mitigation steps.
  - Risk Drivers Deep Dive explaining specific customer churn hazards.
  - Raw JSON API response viewer.

#### Tab 2 — Batch Customer Risk Scoring
- File uploader supporting customer CSV uploads.
- Quick-demo button loading 25 real records from `data/processed/cleaned_churn_data.csv`.
- High-speed scoring displaying total latency and average latency per customer.
- Summary metrics showing counts and percentages of High, Medium, and Low risk cohorts.
- Interactive table and one-click downloadable `telecom_scored_customers.csv`.

#### Tab 3 — Model Performance & Architecture
- Live KPIs displaying production model metrics:
  - **Recall**: `79.89%`
  - **ROC-AUC**: `0.8415`
  - **F1-Score**: `0.6221`
  - **Accuracy**: `74.22%`
  - **Precision**: `50.94%`
- Formatted view of optimal tuned hyperparameters (`class_weight='balanced'`, `n_estimators=100`, etc.).
- Feature importance ranking table.

---

## 3. Results & Deliverables

| Deliverable | File Path | Description |
|---|---|---|
| **Streamlit Web Application** | `app/streamlit_app.py` | Complete responsive frontend interface |
| **Inference Integration** | `src/inference.py` | Unified prediction & retention recommendations layer |
| **Custom Input Controls** | Embedded in `streamlit_app.py` | Full demographic, service, and billing parameter controls |

---

## 4. Verification & Validation

1. **Compilation & Syntax**:
   - `python -m py_compile app/streamlit_app.py` executed cleanly with exit code 0.
2. **API & Engine Connectivity**:
   - Verified seamless communication with FastAPI `/predict` and `/health` endpoints.
   - Verified in-memory fallback execution when external HTTP services are uninitialized.
3. **End-to-End Prediction Verification**:
   - High-risk profile correctly triggers High Risk alert (`72.5%` probability) and offers contract discount recommendations.
   - Low-risk profile correctly triggers Low Risk badge (`20.6%` probability).

---

## 5. Next Steps

With the UI operational and integrated with the backend API, the project advances to **TASK 16 — Integration & End-to-End System Testing**.
