import pandas as pd
import numpy as np
import time
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.pipeline import split_data, get_cv_strategy
from src.utils import get_logger, get_project_root

logger = get_logger("train")

def get_models():
    """Returns a dictionary of un-tuned baseline models."""
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1),
        'LightGBM': LGBMClassifier(random_state=42, n_jobs=-1),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_jobs=-1)
    }

def train_and_evaluate_baselines(data_path: Path, output_dir: Path):
    logger.info(f"Loading data for training from {data_path}")
    df = pd.read_csv(data_path)
    
    # TASK 08: Split data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Stratified K-Fold setup
    cv = get_cv_strategy()
    models = get_models()
    
    results = []
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    logger.info("Starting baseline model evaluation via cross-validation...")
    for name, model in models.items():
        logger.info(f"Evaluating {name}...")
        start_time = time.time()
        
        # We evaluate on X_train using CV
        cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        
        train_time = time.time() - start_time
        
        results.append({
            'Model': name,
            'Accuracy': cv_results['test_accuracy'].mean(),
            'Precision': cv_results['test_precision'].mean(),
            'Recall': cv_results['test_recall'].mean(),
            'F1_Score': cv_results['test_f1'].mean(),
            'ROC_AUC': cv_results['test_roc_auc'].mean(),
            'CV_Time_Seconds': round(train_time, 2)
        })
        logger.info(f"{name} completed in {train_time:.2f}s - F1: {cv_results['test_f1'].mean():.4f}")
        
    results_df = pd.DataFrame(results).sort_values(by='F1_Score', ascending=False)
    
    output_path = output_dir / "reports" / "baseline_metrics.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    logger.info(f"Baseline evaluation complete. Results saved to {output_path}")
    return results_df

if __name__ == "__main__":
    root = get_project_root()
    data_path = root / "data" / "processed" / "selected_churn_data.csv"
    train_and_evaluate_baselines(data_path, root)
