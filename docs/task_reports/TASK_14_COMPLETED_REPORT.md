# TASK 14 — FastAPI Backend API Development
## Completion Report

**Project**: Telecom Customer Churn Prediction & Retention System  
**Task ID**: TASK 14  
**Status**: COMPLETED  
**Executed On**: 2026-10-07  
**Assignment Requirement**: Requirement #13 ("Build an ML API")

---

## 1. Objective

Develop a production-ready RESTful API microservice using FastAPI to expose churn prediction endpoints, handle input validation, provide system health monitoring, and generate real-time customer retention recommendations.

---

## 2. Implementation Strategy & Architecture

### 2.1 Pydantic Request & Response Validation (`api/schemas.py`)
- Created `ChurnPredictionRequest` validating 19 telecom customer fields:
  - Categorical enums: Contract, InternetService, PaymentMethod, service add-ons.
  - Numerical constraints: `tenure` ($0 \le \text{tenure} \le 120$), `MonthlyCharges` ($\ge 0$), `TotalCharges` ($\ge 0$).
  - Automatic calculation of `TotalCharges` if omitted by caller.
- Created `ChurnPredictionResponse` returning:
  - `churn_prediction` (0 or 1), `churn_label` ("Yes" or "No")
  - `churn_probability` (float $0.0 \dots 1.0$), `risk_score_percent` (0.0% to 100.0%)
  - `risk_level` ("Low", "Medium", "High")
  - `retention_recommendations` (Actionable, domain-specific mitigation steps)
  - `model_version` ("1.0.0"), `latency_ms` (Inference timing)
- Created `BatchPredictionRequest` / `BatchPredictionResponse` for multi-record scoring.
- Created `HealthResponse` and `MetadataResponse` for observability.

### 2.2 Endpoint Routers (`api/endpoints.py`)
- `GET /health`: Returns service health, uptime seconds, model version, and load state.
- `POST /predict`: Single customer churn risk prediction and retention strategies.
- `POST /predict/batch`: High-throughput batch inference with average latency metrics.
- `GET /metadata`: Exposes model metadata, hyperparameters, and test metrics.
- `GET /sample-profiles`: Delivers pre-configured high, moderate, and low-risk test payloads.

### 2.3 Application Configuration & Error Handling (`api/main.py`)
- Configured FastAPI `lifespan` context manager pre-loading model weights and transformers into memory upon startup.
- Configured `CORSMiddleware` with open policy for seamless Web UI and external integration.
- Custom exception handler for `RequestValidationError` converting schema failures into clean HTTP 422 JSON error messages.
- Global exception handler catching unhandled exceptions and returning structured HTTP 500 error envelopes.
- Mounted interactive Swagger documentation at `/docs` and ReDoc at `/redoc`.

---

## 3. Results & Outputs

### 3.1 Deliverables Inventory

| Deliverable | File Path | Description |
|---|---|---|
| **API Entrypoint** | `api/main.py` | FastAPI application, lifespan handler, CORS, and exception handling |
| **Pydantic Schemas** | `api/schemas.py` | Strict validation models for request/response payloads |
| **Route Handlers** | `api/endpoints.py` | `/health`, `/predict`, `/predict/batch`, `/metadata`, `/sample-profiles` |
| **Inference Engine** | `src/inference.py` | Preprocessing, feature mapping, LightGBM scoring, recommendation logic |

### 3.2 Endpoint Specifications

| Endpoint | Method | Request Payload | Response Model | Description |
|---|---|---|---|---|
| `/health` | `GET` | None | `HealthResponse` | Verifies model state, uptime, and HTTP 200 OK |
| `/predict` | `POST` | `ChurnPredictionRequest` | `ChurnPredictionResponse` | Single customer churn risk evaluation (< 50ms) |
| `/predict/batch` | `POST` | `BatchPredictionRequest` | `BatchPredictionResponse` | Batch customer risk scoring |
| `/metadata` | `GET` | None | `MetadataResponse` | Model version, 28 features, benchmark metrics |
| `/sample-profiles` | `GET` | None | `dict` | High, medium, and low risk customer fixtures |

---

## 4. Verification & Validation

1. **Validation Tests**:
   - Out-of-bounds input (`tenure = -10.0`) properly rejected with **HTTP 422 Unprocessable Entity**.
   - Invalid category (`Contract = "Infinite"`) properly rejected with **HTTP 422**.
2. **Inference Accuracy & Discrimination**:
   - High-risk sample: `72.5%` churn probability, Risk Level: **High**, Prediction: **Yes**.
   - Low-risk sample: `20.6%` churn probability, Risk Level: **Low**, Prediction: **No**.
3. **Automated Testing**:
   - `tests/test_api.py`: **10 / 10 integration tests PASSED**.

---

## 5. Next Steps

TASK 14 is complete and serves as the backend REST layer for **TASK 15 (Streamlit Web UI Development)**.
