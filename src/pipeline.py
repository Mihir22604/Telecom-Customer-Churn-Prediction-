import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Tuple
from pathlib import Path
from src.utils import get_logger

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

logger = get_logger("pipeline")

CONTINUOUS_FEATURES = [
    'tenure', 'MonthlyCharges', 'TotalCharges',
    'AvgMonthlyChargeRatio', 'MonthlyToTotalRatio', 'TotalServicesCount'
]


class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    StandardScaler applied strictly to continuous numerical features.
    Binary and one-hot encoded categorical features pass through unchanged.
    Ensures safe scikit-learn Pipeline integration and serialization.
    """
    def __init__(self, continuous_cols=None):
        self.continuous_cols = continuous_cols or CONTINUOUS_FEATURES
        self.scaler = StandardScaler()
        self.continuous_cols_ = []

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.continuous_cols_ = [c for c in self.continuous_cols if c in X.columns]
        if self.continuous_cols_:
            self.scaler.fit(X[self.continuous_cols_])
            logger.info(f"Fitted StandardScaler on: {self.continuous_cols_}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_out = X.copy()
        if self.continuous_cols_:
            X_out[self.continuous_cols_] = self.scaler.transform(X[self.continuous_cols_])
        return X_out

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


# Backwards compatibility alias
PreprocessingPipeline = FeatureScaler


def split_data(
    df: pd.DataFrame, 
    target_col: str = 'Churn', 
    test_size: float = 0.20, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the dataset into training and testing sets with stratification.
    """
    logger.info(f"Splitting data... test_size={test_size}, stratify=True")
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
        
    X = df.drop(columns=[target_col])
    if 'customerID' in X.columns:
        X = X.drop(columns=['customerID'])
        
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=random_state
    )
    
    logger.info(f"Split complete. X_train: {X_train.shape}, X_test: {X_test.shape}")
    logger.info(f"y_train class distribution: \n{y_train.value_counts(normalize=True).round(3)}")
    logger.info(f"y_test class distribution: \n{y_test.value_counts(normalize=True).round(3)}")
    
    return X_train, X_test, y_train, y_test


def get_cv_strategy(n_splits: int = 5, random_state: int = 42) -> StratifiedKFold:
    """
    Returns a configured StratifiedKFold object for cross-validation.
    """
    logger.info(f"Configured StratifiedKFold with {n_splits} splits.")
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

