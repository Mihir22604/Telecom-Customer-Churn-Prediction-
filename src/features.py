import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler

from src.utils import get_project_root, get_logger

logger = get_logger("features")

class FeatureEngineeringTransformer:
    """
    Scikit-learn style Feature Engineering Transformer for Telecom Customer Churn dataset.
    Constructs domain features, financial ratios, service counts, encodings, and scaling.
    Maintains fitted scaler state to ensure zero data leakage during training and inference.
    """
    def __init__(self, scale_numerical: bool = True):
        self.scale_numerical = scale_numerical
        self.is_fitted = False
        self.scaler = StandardScaler()
        
        self.continuous_num_cols = [
            'tenure', 'MonthlyCharges', 'TotalCharges',
            'AvgMonthlyChargeRatio', 'MonthlyToTotalRatio', 'TotalServicesCount'
        ]
        self.feature_columns: List[str] = []

    def _engineer_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Construct domain attributes, tenure cohorts, financial ratios, and service bundling scores.
        """
        df_feat = df.copy()
        
        # 1. Tenure Cohorts / Binning
        if 'tenure' in df_feat.columns:
            df_feat['TenureCohort'] = pd.cut(
                df_feat['tenure'],
                bins=[-1, 12, 24, 48, 72],
                labels=['0-12m', '12-24m', '24-48m', '48-72m']
            ).astype(str)
            
        # 2. Financial Ratios
        if 'TotalCharges' in df_feat.columns and 'tenure' in df_feat.columns:
            df_feat['AvgMonthlyChargeRatio'] = (df_feat['TotalCharges'] / (df_feat['tenure'] + 1.0)).round(4)
            
        if 'MonthlyCharges' in df_feat.columns and 'TotalCharges' in df_feat.columns:
            df_feat['MonthlyToTotalRatio'] = (df_feat['MonthlyCharges'] / (df_feat['TotalCharges'] + 1.0)).round(4)
            
        # 3. Total Services Count (Bundling Index)
        service_cols = [
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        service_count = pd.Series(0, index=df_feat.index)
        for col in service_cols:
            if col in df_feat.columns:
                # Add 1 if customer has an active subscription (not 'No' or 'No internet service' or 'No phone service')
                active_mask = ~df_feat[col].isin(['No', 'No internet service', 'No phone service'])
                service_count += active_mask.astype(int)
                
        df_feat['TotalServicesCount'] = service_count
        
        # 4. Security & Support Anchor Flag
        if 'OnlineSecurity' in df_feat.columns and 'TechSupport' in df_feat.columns:
            df_feat['HasTechSupportSecurity'] = (
                (df_feat['OnlineSecurity'] == 'Yes') & (df_feat['TechSupport'] == 'Yes')
            ).astype(int)
        else:
            df_feat['HasTechSupportSecurity'] = 0
            
        # 5. Automatic Payment Indicator
        if 'PaymentMethod' in df_feat.columns:
            df_feat['IsAutomaticPayment'] = df_feat['PaymentMethod'].str.contains('automatic', case=False, na=False).astype(int)
        else:
            df_feat['IsAutomaticPayment'] = 0
            
        return df_feat

    def fit(self, df: pd.DataFrame) -> "FeatureEngineeringTransformer":
        """
        Fit scaler on continuous numerical features of training DataFrame.
        """
        logger.info("Fitting FeatureEngineeringTransformer...")
        df_domain = self._engineer_domain_features(df)
        
        if self.scale_numerical:
            num_cols_to_scale = [c for c in self.continuous_num_cols if c in df_domain.columns]
            self.scaler.fit(df_domain[num_cols_to_scale])
            logger.info(f"Fitted StandardScaler on continuous features: {num_cols_to_scale}")
            
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform input DataFrame: apply domain features, categorical encodings, and scaling.
        """
        if not self.is_fitted:
            self.fit(df)
            
        logger.info("Transforming DataFrame with feature engineering...")
        df_domain = self._engineer_domain_features(df)
        
        # Preserve customerID metadata if present
        customer_ids = df_domain['customerID'] if 'customerID' in df_domain.columns else None
        
        # Target Binary Encoding
        target_series = None
        if 'Churn' in df_domain.columns:
            target_series = (df_domain['Churn'] == 'Yes').astype(int)
            
        # Binary Categories (0/1)
        binary_map = {'Yes': 1, 'No': 0, 'Female': 1, 'Male': 0}
        binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        
        for col in binary_cols:
            if col in df_domain.columns:
                df_domain[col] = df_domain[col].map(binary_map).fillna(0).astype(int)
                
        # Nominal Multi-Class Categoricals for One-Hot Encoding
        nominal_cols = [
            'Contract', 'InternetService', 'PaymentMethod', 'MultipleLines',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'TenureCohort'
        ]
        nominal_cols_present = [c for c in nominal_cols if c in df_domain.columns]
        
        df_encoded = pd.get_dummies(
            df_domain,
            columns=nominal_cols_present,
            drop_first=True,
            dtype=int
        )
        
        # Apply Scaling to continuous features if configured
        if self.scale_numerical:
            num_cols_to_scale = [c for c in self.continuous_num_cols if c in df_encoded.columns]
            df_encoded[num_cols_to_scale] = self.scaler.transform(df_encoded[num_cols_to_scale]).round(4)
            
        # Re-attach customerID and Churn target if present
        if customer_ids is not None:
            df_encoded['customerID'] = customer_ids
        if target_series is not None:
            df_encoded['Churn'] = target_series
            
        self.feature_columns = [c for c in df_encoded.columns if c not in ['customerID', 'Churn']]
        logger.info(f"Feature engineering complete. Total predictor features: {len(self.feature_columns)}")
        
        return df_encoded

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit transformer and transform DataFrame in a single call.
        """
        return self.fit(df).transform(df)

def engineer_features(cleaned_df_path: Optional[Path] = None, save_processed: bool = True) -> pd.DataFrame:
    """
    Convenience function to load cleaned data, execute feature engineering, and save to data/processed.
    """
    project_root = get_project_root()
    if cleaned_df_path is None:
        cleaned_df_path = project_root / "data" / "processed" / "cleaned_churn_data.csv"
        
    logger.info(f"Loading cleaned dataset for feature engineering: {cleaned_df_path}")
    df_cleaned = pd.read_csv(cleaned_df_path)
    
    transformer = FeatureEngineeringTransformer(scale_numerical=True)
    df_engineered = transformer.fit_transform(df_cleaned)
    
    if save_processed:
        processed_dir = project_root / "data" / "processed"
        os.makedirs(processed_dir, exist_ok=True)
        
        output_path = processed_dir / "engineered_churn_data.csv"
        df_engineered.to_csv(output_path, index=False)
        logger.info(f"Saved engineered dataset to: {output_path}")
        
    return df_engineered

if __name__ == "__main__":
    df_engineered = engineer_features()
    print("Feature Engineering Completed Successfully!")
    print(f"Engineered Dataset Shape: {df_engineered.shape[0]} rows x {df_engineered.shape[1]} columns")
    print(f"Sample Features: {list(df_engineered.columns[:10])}")
