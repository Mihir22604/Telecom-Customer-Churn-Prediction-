from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class ChurnPredictionRequest(BaseModel):
    """
    Schema for customer input payload requiring churn risk evaluation.
    Validates demographic attributes, service subscriptions, and billing terms.
    """
    customerID: Optional[str] = Field(
        default="CUST-PREDICT", 
        description="Unique customer identifier code"
    )
    gender: Literal["Male", "Female"] = Field(
        default="Female",
        description="Customer gender"
    )
    SeniorCitizen: int = Field(
        default=0, ge=0, le=1,
        description="Whether the customer is a senior citizen (1: Yes, 0: No)"
    )
    Partner: Literal["Yes", "No"] = Field(
        default="No",
        description="Whether the customer has a partner"
    )
    Dependents: Literal["Yes", "No"] = Field(
        default="No",
        description="Whether the customer has dependents"
    )
    tenure: float = Field(
        default=1.0, ge=0.0, le=120.0,
        description="Number of months customer has stayed with the company"
    )
    PhoneService: Literal["Yes", "No"] = Field(
        default="Yes",
        description="Whether the customer has a phone service subscription"
    )
    MultipleLines: Literal["No", "Yes", "No phone service"] = Field(
        default="No",
        description="Whether customer has multiple lines"
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        default="Fiber optic",
        description="Customer's internet service provider tier"
    )
    OnlineSecurity: Literal["No", "Yes", "No internet service"] = Field(
        default="No",
        description="Whether customer has online security add-on"
    )
    OnlineBackup: Literal["No", "Yes", "No internet service"] = Field(
        default="No",
        description="Whether customer has online backup service"
    )
    DeviceProtection: Literal["No", "Yes", "No internet service"] = Field(
        default="No",
        description="Whether customer has premium device protection"
    )
    TechSupport: Literal["No", "Yes", "No internet service"] = Field(
        default="No",
        description="Whether customer has premium tech support"
    )
    StreamingTV: Literal["No", "Yes", "No internet service"] = Field(
        default="Yes",
        description="Whether customer streams television"
    )
    StreamingMovies: Literal["No", "Yes", "No internet service"] = Field(
        default="Yes",
        description="Whether customer streams movies"
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        default="Month-to-month",
        description="Contract commitment term"
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(
        default="Yes",
        description="Whether customer uses paperless billing"
    )
    PaymentMethod: Literal[
        "Electronic check", 
        "Mailed check", 
        "Bank transfer (automatic)", 
        "Credit card (automatic)"
    ] = Field(
        default="Electronic check",
        description="Customer payment method"
    )
    MonthlyCharges: float = Field(
        default=85.0, ge=0.0, le=500.0,
        description="Monthly recurring charge amount in USD"
    )
    TotalCharges: Optional[float] = Field(
        default=None, ge=0.0,
        description="Total charges accrued to date in USD. Auto-computed if omitted."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1.0,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85
            }
        }
    )


class ChurnPredictionResponse(BaseModel):
    """
    Schema for churn inference response returned by the API.
    """
    customerID: Optional[str] = Field(default=None, description="Customer identifier")
    churn_prediction: int = Field(description="Binary churn prediction (1: Churn, 0: Retain)")
    churn_label: str = Field(description="Human-readable churn prediction ('Yes' or 'No')")
    churn_probability: float = Field(description="Estimated probability of churning (0.0 to 1.0)")
    risk_score_percent: float = Field(description="Churn probability expressed as percentage")
    risk_level: Literal["Low", "Medium", "High"] = Field(description="Categorical churn risk classification")
    retention_recommendations: List[str] = Field(description="Actionable customer retention strategies")
    model_version: str = Field(description="Production model artifact version")
    latency_ms: float = Field(description="Inference latency in milliseconds")


class BatchPredictionRequest(BaseModel):
    """
    Schema for evaluating a batch of multiple customers simultaneously.
    """
    customers: List[ChurnPredictionRequest]


class BatchPredictionResponse(BaseModel):
    """
    Response schema for batch predictions.
    """
    total_processed: int
    predictions: List[ChurnPredictionResponse]
    average_latency_ms: float


class HealthResponse(BaseModel):
    """
    Schema for system health check endpoint.
    """
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
    timestamp: str
    environment: Optional[str] = "production"
    system_metrics: Optional[Dict[str, Any]] = None


class MetadataResponse(BaseModel):
    """
    Schema for model metadata endpoint.
    """
    model_name: str
    model_version: str
    training_timestamp: str
    algorithm: str
    feature_count: int
    feature_names: List[str]
    test_set_metrics: Dict[str, float]
