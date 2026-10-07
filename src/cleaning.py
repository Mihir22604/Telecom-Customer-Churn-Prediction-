import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from src.utils import get_project_root, get_logger
from src.data_loader import load_raw_data

logger = get_logger("cleaning")

class TelecomDataCleaner:
    """
    Reusable and fit-transform compatible data cleaner for Telecom Customer Churn dataset.
    Stores median/mode imputation values during training/fitting to ensure reproducible
    inference pipeline consistency without data leakage.
    """
    def __init__(self):
        self.is_fitted = False
        self.numerical_medians: Dict[str, float] = {}
        self.categorical_modes: Dict[str, str] = {}
        self.numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        self.categorical_cols = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod'
        ]

    def fit(self, df: pd.DataFrame) -> "TelecomDataCleaner":
        """
        Compute median values for numerical columns and mode values for categorical columns.
        """
        logger.info("Fitting TelecomDataCleaner on input DataFrame...")
        temp_df = df.copy()
        
        # Coerce TotalCharges to numeric for median computation
        if 'TotalCharges' in temp_df.columns:
            temp_df['TotalCharges'] = pd.to_numeric(
                temp_df['TotalCharges'].replace(' ', np.nan), errors='coerce'
            )
            
        # Compute numerical medians
        for col in self.numerical_cols:
            if col in temp_df.columns:
                self.numerical_medians[col] = float(temp_df[col].median(skipna=True))
                
        # Compute categorical modes
        for col in self.categorical_cols:
            if col in temp_df.columns:
                mode_val = temp_df[col].mode(dropna=True)
                if not mode_val.empty:
                    self.categorical_modes[col] = str(mode_val.iloc[0])
                else:
                    self.categorical_modes[col] = 'Unknown'
                    
        self.is_fitted = True
        logger.info(f"Cleaner fitted. Medians: {self.numerical_medians}, Modes count: {len(self.categorical_modes)}")
        return self

    def fit_transform(self, df: pd.DataFrame, drop_missing_target: bool = True) -> pd.DataFrame:
        """
        Fit cleaner on DataFrame and transform it in a single step.
        """
        return self.fit(df).transform(df, drop_missing_target=drop_missing_target)

    def transform(self, df: pd.DataFrame, drop_missing_target: bool = True) -> pd.DataFrame:
        """
        Apply cleaning transformations to the DataFrame.
        """
        if not self.is_fitted:
            self.fit(df)
            
        logger.info("Transforming DataFrame...")
        cleaned_df = df.copy()
        
        # 1. Target Validation: Drop rows missing Churn if requested
        if drop_missing_target and 'Churn' in cleaned_df.columns:
            initial_count = len(cleaned_df)
            cleaned_df = cleaned_df.dropna(subset=['Churn'])
            dropped_target = initial_count - len(cleaned_df)
            if dropped_target > 0:
                logger.info(f"Dropped {dropped_target} rows missing target 'Churn'.")
                
        # 2. Exact Duplicate Removal
        initial_count = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        dropped_dups = initial_count - len(cleaned_df)
        if dropped_dups > 0:
            logger.info(f"Dropped {dropped_dups} exact duplicate rows.")
            
        # 3. Type Coercion: Convert TotalCharges string space ' ' to 0.0 and float64
        if 'TotalCharges' in cleaned_df.columns:
            cleaned_df['TotalCharges'] = cleaned_df['TotalCharges'].replace(' ', '0.0')
            cleaned_df['TotalCharges'] = pd.to_numeric(cleaned_df['TotalCharges'], errors='coerce')
            
        # 4. Numerical Imputation using fitted medians
        for col in self.numerical_cols:
            if col in cleaned_df.columns:
                missing_before = cleaned_df[col].isnull().sum()
                if missing_before > 0:
                    fill_val = self.numerical_medians.get(col, 0.0)
                    cleaned_df[col] = cleaned_df[col].fillna(fill_val)
                    logger.info(f"Imputed {missing_before} missing values in '{col}' with median: {fill_val}")
                    
        # 5. Categorical Imputation & String Standardization
        for col in self.categorical_cols:
            if col in cleaned_df.columns:
                # Standardize string formatting if object type
                if cleaned_df[col].dtype == 'object':
                    cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
                    cleaned_df[col] = cleaned_df[col].replace({'nan': np.nan, 'None': np.nan, '': np.nan})
                    
                missing_before = cleaned_df[col].isnull().sum()
                if missing_before > 0:
                    fill_val = self.categorical_modes.get(col, 'Unknown')
                    cleaned_df[col] = cleaned_df[col].fillna(fill_val)
                    logger.info(f"Imputed {missing_before} missing values in '{col}' with mode: {fill_val}")
                    
        # 6. Standardize Target 'Churn' strings if present
        if 'Churn' in cleaned_df.columns:
            cleaned_df['Churn'] = cleaned_df['Churn'].astype(str).str.strip()
            
        # Reset Index
        cleaned_df = cleaned_df.reset_index(drop=True)
        logger.info(f"Transform complete. Output shape: {cleaned_df.shape}")
        
        return cleaned_df

def clean_telecom_data(raw_df: Optional[pd.DataFrame] = None, save_processed: bool = True) -> pd.DataFrame:
    """
    Convenience function to load raw data, fit-transform cleaner, and save to data/processed.
    """
    if raw_df is None:
        raw_df = load_raw_data()
        
    cleaner = TelecomDataCleaner()
    cleaned_df = cleaner.fit_transform(raw_df)
    
    if save_processed:
        project_root = get_project_root()
        processed_dir = project_root / "data" / "processed"
        os.makedirs(processed_dir, exist_ok=True)
        
        output_path = processed_dir / "cleaned_churn_data.csv"
        cleaned_df.to_csv(output_path, index=False)
        logger.info(f"Saved cleaned dataset to: {output_path}")
        
    return cleaned_df

if __name__ == "__main__":
    df_cleaned = clean_telecom_data()
    print("Data Cleaning Completed Successfully!")
    print(f"Cleaned Data Shape: {df_cleaned.shape[0]} rows x {df_cleaned.shape[1]} columns")
    print("Null count in cleaned dataset:", df_cleaned.isnull().sum().sum())
