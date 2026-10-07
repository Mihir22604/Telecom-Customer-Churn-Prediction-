import pandas as pd
from pathlib import Path
from src.utils import get_project_root, get_logger
from src.pipeline import split_data
from src.train import get_models
from src.evaluation import evaluate_models
from src.tuning import execute_tuning_and_selection

def main():
    logger = get_logger("main_execution")
    root = get_project_root()
    
    # Load Data
    data_path = root / "data" / "processed" / "selected_churn_data.csv"
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Split
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Task 10: Multi-Metric Evaluation & Imbalance Analysis
    # We will train the baseline models on X_train and evaluate on X_test
    logger.info("--- TASK 10: Model Evaluation on Test Set ---")
    models = get_models()
    for name, model in models.items():
        model.fit(X_train, y_train)
        
    metrics_df = evaluate_models(models, X_test, y_test, root)
    logger.info(f"Test Set Evaluation Results:\n{metrics_df.to_string()}")
    
    # Task 11 & 12: Hyperparameter Tuning and Final Selection
    logger.info("--- TASK 11 & 12: Tuning and Final Selection ---")
    best_model, best_name = execute_tuning_and_selection(X_train, y_train, root)
    
    # Evaluate final model on test set
    logger.info(f"--- Evaluating Final Tuned Model ({best_name}) on Test Set ---")
    final_metrics = evaluate_models({f"Tuned_{best_name}": best_model}, X_test, y_test, root)
    logger.info(f"Final Model Test Performance:\n{final_metrics.to_string()}")

if __name__ == "__main__":
    main()
