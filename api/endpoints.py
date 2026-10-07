import time
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any, List

from api.schemas import (
    ChurnPredictionRequest,
    ChurnPredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    MetadataResponse
)
from src.inference import get_predictor
from src.utils import get_logger, get_settings, get_system_metrics

logger = get_logger("api.endpoints")
router = APIRouter()

# Module-level startup tracking
_STARTUP_TIME = time.time()


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Service Health Check",
    description="Returns the health status, model load state, uptime, and server telemetry."
)
async def health_check():
    try:
        predictor = get_predictor()
        uptime = round(time.time() - _STARTUP_TIME, 2)
        settings = get_settings()
        model_ver = predictor.metadata.get("model_version", settings.model_version) if predictor.metadata else settings.model_version
        telemetry = get_system_metrics()
        
        return HealthResponse(
            status="healthy",
            model_loaded=predictor._is_loaded,
            model_version=model_ver,
            uptime_seconds=uptime,
            timestamp=datetime.now(timezone.utc).isoformat(),
            environment=settings.app_env,
            system_metrics=telemetry
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )


@router.post(
    "/predict",
    response_model=ChurnPredictionResponse,
    tags=["Inference"],
    summary="Predict Single Customer Churn Risk",
    description="Accepts customer demographic, service, and billing details to estimate churn probability."
)
async def predict_churn(request: ChurnPredictionRequest):
    try:
        predictor = get_predictor()
        raw_dict = request.model_dump()
        result = predictor.predict(raw_dict)
        
        return ChurnPredictionResponse(
            customerID=request.customerID,
            churn_prediction=result["churn_prediction"],
            churn_label=result["churn_label"],
            churn_probability=result["churn_probability"],
            risk_score_percent=result["risk_score_percent"],
            risk_level=result["risk_level"],
            retention_recommendations=result["retention_recommendations"],
            model_version=result["model_version"],
            latency_ms=result["latency_ms"]
        )
    except Exception as e:
        logger.error(f"Prediction failed for customer {request.customerID}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}"
        )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Inference"],
    summary="Predict Batch Customer Churn Risk",
    description="Processes multiple customer records in a single request and returns predictions for all."
)
async def predict_batch_churn(batch_request: BatchPredictionRequest):
    try:
        predictor = get_predictor()
        start_time = time.perf_counter()
        
        predictions_out = []
        for req in batch_request.customers:
            raw_dict = req.model_dump()
            res = predictor.predict(raw_dict)
            predictions_out.append(
                ChurnPredictionResponse(
                    customerID=req.customerID,
                    churn_prediction=res["churn_prediction"],
                    churn_label=res["churn_label"],
                    churn_probability=res["churn_probability"],
                    risk_score_percent=res["risk_score_percent"],
                    risk_level=res["risk_level"],
                    retention_recommendations=res["retention_recommendations"],
                    model_version=res["model_version"],
                    latency_ms=res["latency_ms"]
                )
            )
            
        total_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        avg_latency = round(total_time_ms / max(1, len(predictions_out)), 2)
        
        return BatchPredictionResponse(
            total_processed=len(predictions_out),
            predictions=predictions_out,
            average_latency_ms=avg_latency
        )
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {str(e)}"
        )


@router.get(
    "/metadata",
    response_model=MetadataResponse,
    tags=["Model"],
    summary="Retrieve Model Metadata",
    description="Returns detailed training metadata, feature specifications, and benchmark metrics."
)
async def get_model_metadata():
    try:
        predictor = get_predictor()
        meta = predictor.metadata
        return MetadataResponse(
            model_name=meta.get("model_name", "LightGBMChurnClassifier"),
            model_version=meta.get("model_version", "1.0.0"),
            training_timestamp=meta.get("training_timestamp", ""),
            algorithm=meta.get("algorithm", "LightGBM"),
            feature_count=meta.get("feature_count", len(meta.get("feature_names", []))),
            feature_names=meta.get("feature_names", []),
            test_set_metrics=meta.get("test_set_metrics", {})
        )
    except Exception as e:
        logger.error(f"Failed to fetch metadata: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch metadata: {str(e)}"
        )


@router.get(
    "/sample-profiles",
    tags=["Utilities"],
    summary="Sample Customer Profiles",
    description="Provides pre-configured high, moderate, and low churn risk customer profiles for quick testing."
)
async def get_sample_profiles() -> Dict[str, Any]:
    return {
        "high_risk_customer": {
            "customerID": "SAMPLE-HIGH-01",
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
        },
        "low_risk_customer": {
            "customerID": "SAMPLE-LOW-01",
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 60.0,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 45.00,
            "TotalCharges": 2700.00
        },
        "moderate_risk_customer": {
            "customerID": "SAMPLE-MOD-01",
            "gender": "Male",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 18.0,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "One year",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 80.00,
            "TotalCharges": 1440.00
        }
    }
