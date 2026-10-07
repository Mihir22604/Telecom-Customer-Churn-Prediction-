import pandas as pd
import json
import joblib
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from src.pipeline import get_cv_strategy
from src.utils import get_logger

logger = get_logger("tuning")

def tune_lightgbm(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Tunes LightGBM parameters using RandomizedSearchCV.
    """
    logger.info("Starting Hyperparameter Tuning for LightGBM...")
    
    param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7, -1],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'class_weight': ['balanced', None]
    }
    
    lgbm = LGBMClassifier(random_state=42, n_jobs=-1)
    cv = get_cv_strategy(n_splits=5)
    
    rs = RandomizedSearchCV(
        estimator=lgbm,
        param_distributions=param_dist,
        n_iter=20,
        scoring='f1',
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    rs.fit(X_train, y_train)
    logger.info(f"LightGBM Best F1 Score: {rs.best_score_:.4f}")
    logger.info(f"LightGBM Best Params: {rs.best_params_}")
    
    return rs.best_estimator_, rs.best_params_, rs.best_score_

def tune_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Tunes Logistic Regression parameters using GridSearchCV.
    """
    logger.info("Starting Hyperparameter Tuning for Logistic Regression...")
    
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'class_weight': ['balanced', None],
        'solver': ['lbfgs', 'liblinear']
    }
    
    lr = LogisticRegression(max_iter=1000, random_state=42)
    cv = get_cv_strategy(n_splits=5)
    
    gs = GridSearchCV(
        estimator=lr,
        param_grid=param_grid,
        scoring='f1',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    
    gs.fit(X_train, y_train)
    logger.info(f"Logistic Regression Best F1 Score: {gs.best_score_:.4f}")
    logger.info(f"Logistic Regression Best Params: {gs.best_params_}")
    
    return gs.best_estimator_, gs.best_params_, gs.best_score_

def execute_tuning_and_selection(X_train: pd.DataFrame, y_train: pd.Series, output_dir: Path):
    """
    Executes tuning on top 2 models (LightGBM and Logistic Regression),
    saves the tuning results, and exports the final best model (Task 11 & 12).
    """
    lgbm_model, lgbm_params, lgbm_score = tune_lightgbm(X_train, y_train)
    lr_model, lr_params, lr_score = tune_logistic_regression(X_train, y_train)
    
    results = {
        "LightGBM": {
            "best_f1": lgbm_score,
            "params": lgbm_params
        },
        "LogisticRegression": {
            "best_f1": lr_score,
            "params": lr_params
        }
    }
    
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "hyperparameter_tuning_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Select best model for Task 12
    if lgbm_score >= lr_score:
        best_model = lgbm_model
        best_name = "LightGBM"
    else:
        best_model = lr_model
        best_name = "LogisticRegression"
        
    logger.info(f"Final Selected Model: {best_name}")
    
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "final_model.pkl"
    joblib.dump(best_model, model_path)
    logger.info(f"Saved final model to {model_path}")
    
    return best_model, best_name
