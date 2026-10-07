import time
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union, Optional

from src.utils import get_logger, get_project_root

logger = get_logger("inference")

CONTINUOUS_COLS = [
    'tenure', 'MonthlyCharges', 'TotalCharges',
    'AvgMonthlyChargeRatio', 'MonthlyToTotalRatio', 'TotalServicesCount'
]

SERVICE_COLS = [
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
]


def raw_to_model_features(df_raw: pd.DataFrame, scaler=None) -> pd.DataFrame:
    """
    Deterministic transformation from raw telecom customer fields to the exact 
    28 feature vector required by the trained production model.
    Matches the training dataset transformations with zero discrepancy.
    """
    df = df_raw.copy()

    # Coerce numeric fields
    tenure = pd.to_numeric(df.get('tenure', 0.0), errors='coerce').fillna(0.0).astype(float)
    monthly = pd.to_numeric(df.get('MonthlyCharges', 0.0), errors='coerce').fillna(0.0).astype(float)
    
    if 'TotalCharges' in df.columns and not df['TotalCharges'].isnull().all():
        total = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(tenure * monthly).astype(float)
    else:
        total = (tenure * monthly).astype(float)

    # Financial ratios
    avg_ratio = (total / (tenure + 1.0)).round(4)
    monthly_ratio = (monthly / (total + 1.0)).round(4)

    # Active services count
    service_count = pd.Series(0, index=df.index)
    for col in SERVICE_COLS:
        if col in df.columns:
            active_mask = ~df[col].astype(str).isin(['No', 'No internet service', 'No phone service', 'nan', 'None'])
            service_count += active_mask.astype(int)

    # Binary demographics and account terms
    gender_bin = df.get('gender', 'Male').astype(str).isin(['Female', '1', 1]).astype(int)
    senior_bin = df.get('SeniorCitizen', 0).isin([1, '1', 'Yes', True]).astype(int)
    partner_bin = df.get('Partner', 'No').astype(str).isin(['Yes', '1', 1, True]).astype(int)
    dependents_bin = df.get('Dependents', 'No').astype(str).isin(['Yes', '1', 1, True]).astype(int)
    paperless_bin = df.get('PaperlessBilling', 'No').astype(str).isin(['Yes', '1', 1, True]).astype(int)
    
    payment_method = df.get('PaymentMethod', '').astype(str)
    auto_pay_bin = payment_method.str.contains('automatic', case=False, na=False).astype(int)
    pm_cc = (payment_method == 'Credit card (automatic)').astype(int)
    pm_echeck = (payment_method == 'Electronic check').astype(int)

    contract = df.get('Contract', 'Month-to-month').astype(str)
    c_one_yr = (contract == 'One year').astype(int)
    c_two_yr = (contract == 'Two year').astype(int)

    internet = df.get('InternetService', 'DSL').astype(str)
    fiber = (internet == 'Fiber optic').astype(int)

    ml_yes = (df.get('MultipleLines', 'No').astype(str) == 'Yes').astype(int)
    sec_yes = (df.get('OnlineSecurity', 'No').astype(str) == 'Yes').astype(int)
    bk_no_int = (df.get('OnlineBackup', 'No').astype(str) == 'No internet service').astype(int)
    bk_yes = (df.get('OnlineBackup', 'No').astype(str) == 'Yes').astype(int)
    dp_yes = (df.get('DeviceProtection', 'No').astype(str) == 'Yes').astype(int)
    ts_yes = (df.get('TechSupport', 'No').astype(str) == 'Yes').astype(int)
    tv_no_int = (df.get('StreamingTV', 'No').astype(str) == 'No internet service').astype(int)
    tv_yes = (df.get('StreamingTV', 'No').astype(str) == 'Yes').astype(int)
    mv_no_int = (df.get('StreamingMovies', 'No').astype(str) == 'No internet service').astype(int)
    mv_yes = (df.get('StreamingMovies', 'No').astype(str) == 'Yes').astype(int)

    cohort_48_72 = ((tenure > 48) & (tenure <= 72)).astype(int)

    feature_matrix = pd.DataFrame({
        'gender': gender_bin,
        'SeniorCitizen': senior_bin,
        'Partner': partner_bin,
        'Dependents': dependents_bin,
        'tenure': tenure,
        'PaperlessBilling': paperless_bin,
        'MonthlyCharges': monthly,
        'TotalCharges': total,
        'AvgMonthlyChargeRatio': avg_ratio,
        'MonthlyToTotalRatio': monthly_ratio,
        'TotalServicesCount': service_count,
        'IsAutomaticPayment': auto_pay_bin,
        'Contract_One year': c_one_yr,
        'Contract_Two year': c_two_yr,
        'InternetService_Fiber optic': fiber,
        'PaymentMethod_Credit card (automatic)': pm_cc,
        'PaymentMethod_Electronic check': pm_echeck,
        'MultipleLines_Yes': ml_yes,
        'OnlineSecurity_Yes': sec_yes,
        'OnlineBackup_No internet service': bk_no_int,
        'OnlineBackup_Yes': bk_yes,
        'DeviceProtection_Yes': dp_yes,
        'TechSupport_Yes': ts_yes,
        'StreamingTV_No internet service': tv_no_int,
        'StreamingTV_Yes': tv_yes,
        'StreamingMovies_No internet service': mv_no_int,
        'StreamingMovies_Yes': mv_yes,
        'TenureCohort_48-72m': cohort_48_72,
    })

    if scaler is not None:
        feature_matrix[CONTINUOUS_COLS] = scaler.transform(feature_matrix[CONTINUOUS_COLS]).round(4)

    return feature_matrix


def generate_retention_recommendations(raw_data: Dict[str, Any], churn_prob: float) -> List[str]:
    """
    Generates tailored, actionable customer retention recommendations based on 
    predicted churn probability and specific customer risk drivers.
    """
    recommendations = []
    
    if churn_prob < 0.35:
        recommendations.append("Customer is at Low Risk. Maintain regular service notifications and loyalty rewards.")
        return recommendations

    # Risk factors evaluation
    contract = str(raw_data.get("Contract", "")).lower()
    if "month-to-month" in contract:
        recommendations.append("Contract Risk: Offer a 1-Year or 2-Year contract with an introductory 15% billing discount.")
        
    payment = str(raw_data.get("PaymentMethod", "")).lower()
    if "electronic check" in payment:
        recommendations.append("Payment Friction Risk: Encourage switching to Automatic Bank Transfer or Credit Card with a one-time $10 bill credit.")
        
    tech_support = str(raw_data.get("TechSupport", "")).lower()
    online_security = str(raw_data.get("OnlineSecurity", "")).lower()
    if tech_support in ["no", "no internet service"] or online_security in ["no", "no internet service"]:
        recommendations.append("Anchor Service Opportunity: Offer a complimentary 3-month trial of TechSupport and OnlineSecurity to boost retention lock-in.")
        
    tenure = float(raw_data.get("tenure", 0))
    if tenure <= 12:
        recommendations.append("Early Lifecycle Alert: Assign to proactive customer success onboarding to resolve early experience friction.")
        
    monthly_charges = float(raw_data.get("MonthlyCharges", 0))
    if monthly_charges > 70.0:
        recommendations.append("Price Sensitivity Risk: Conduct plan audit to recommend an optimized service bundle tailored to actual usage.")

    internet = str(raw_data.get("InternetService", "")).lower()
    if "fiber" in internet and churn_prob >= 0.50:
        recommendations.append("Fiber Optic Service Check: Schedule proactive line quality and speed verification to prevent tech dissatisfaction churn.")

    if not recommendations:
        recommendations.append("Proactively reach out to the customer with a personalized satisfaction survey and loyalty benefit.")
        
    return recommendations


class ChurnPredictor:
    """
    Production-grade inference engine for Telecom Customer Churn prediction.
    Handles raw customer input preprocessing, feature engineering, model scoring,
    and business retention logic.
    """
    def __init__(self, models_dir: Optional[Path] = None):
        self.root = get_project_root()
        self.models_dir = models_dir or (self.root / "models")
        self.pipeline_bundle = None
        self.raw_transformer = None
        self.metadata = None
        self.feature_names = []
        self._is_loaded = False

    def load(self) -> "ChurnPredictor":
        """Loads all serialized artifacts into memory."""
        if self._is_loaded:
            return self

        pipeline_path = self.models_dir / "full_inference_pipeline.joblib"
        metadata_path = self.models_dir / "model_metadata.json"
        raw_trans_path = self.models_dir / "raw_feature_transformer.joblib"

        logger.info(f"Loading inference pipeline from {pipeline_path}")
        self.pipeline_bundle = joblib.load(pipeline_path)

        logger.info(f"Loading metadata from {metadata_path}")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        self.feature_names = self.metadata.get("feature_names", [])

        logger.info(f"Loading raw feature scaler from {raw_trans_path}")
        self.raw_transformer = joblib.load(raw_trans_path)

        self._is_loaded = True
        logger.info("ChurnPredictor loaded successfully.")
        return self

    def preprocess_raw_input(self, data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
        """
        Converts raw telecom input records into the exact 28 model features.
        """
        if not self._is_loaded:
            self.load()

        if isinstance(data, dict):
            df_in = pd.DataFrame([data])
        else:
            df_in = data.copy()

        scaler = getattr(self.raw_transformer, 'scaler', None)
        df_engineered = raw_to_model_features(df_in, scaler=scaler)
        return df_engineered[self.feature_names]

    def predict(self, input_data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Runs single customer churn inference with response formatting and latency tracking.
        """
        if not self._is_loaded:
            self.load()

        start_time = time.perf_counter()

        if isinstance(input_data, dict):
            raw_dict = input_data
            if all(k in input_data for k in self.feature_names):
                X_features = pd.DataFrame([input_data])[self.feature_names]
            else:
                X_features = self.preprocess_raw_input(input_data)
        elif isinstance(input_data, pd.DataFrame):
            raw_dict = input_data.iloc[0].to_dict()
            if all(k in input_data.columns for k in self.feature_names):
                X_features = input_data[self.feature_names]
            else:
                X_features = self.preprocess_raw_input(input_data)
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

        pipeline = self.pipeline_bundle["pipeline"]
        churn_pred = int(pipeline.predict(X_features)[0])
        churn_proba = float(pipeline.predict_proba(X_features)[0, 1])

        # Risk Classification
        if churn_proba >= 0.65:
            risk_level = "High"
        elif churn_proba >= 0.35:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        recommendations = generate_retention_recommendations(raw_dict, churn_proba)

        return {
            "churn_prediction": churn_pred,
            "churn_label": "Yes" if churn_pred == 1 else "No",
            "churn_probability": round(churn_proba, 4),
            "risk_score_percent": round(churn_proba * 100, 2),
            "risk_level": risk_level,
            "retention_recommendations": recommendations,
            "model_version": self.metadata.get("model_version", "1.0.0"),
            "latency_ms": latency_ms,
        }

    def predict_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch inference for multiple customer records."""
        return [self.predict(rec) for rec in records]


_PREDICTOR_INSTANCE: Optional[ChurnPredictor] = None


def get_predictor() -> ChurnPredictor:
    """Returns the singleton ChurnPredictor instance."""
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        _PREDICTOR_INSTANCE = ChurnPredictor().load()
    return _PREDICTOR_INSTANCE
