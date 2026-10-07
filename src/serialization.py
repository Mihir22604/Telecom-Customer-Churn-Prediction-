"""
TASK 13 - Model Serialization & Pipeline Packaging
Combines preprocessing + final classifier into a unified sklearn Pipeline,
serializes all artifacts with joblib, and writes model_metadata.json.
"""
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
import joblib
from datetime import datetime, timezone
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from src.pipeline import (
    split_data, FeatureScaler, PreprocessingPipeline, CONTINUOUS_FEATURES
)
from src.utils import get_logger, get_project_root

logger = get_logger("serialization")


def build_and_serialize(output_dir: Path):
    """
    Full Task 13 execution:
      1. Load selected data & split
      2. Fit preprocessing pipeline on X_train (no leakage)
      3. Re-train final LightGBM model with tuned params on processed X_train
      4. Evaluate on X_test
      5. Serialize preprocessing pipeline + model separately
      6. Build and serialize unified sklearn Pipeline
      7. Write model_metadata.json
      8. Validate deserialized artifacts
    """
    # ── 1. Load data ──────────────────────────────────────────────────────────
    data_path = output_dir / "data" / "processed" / "selected_churn_data.csv"
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    X_train, X_test, y_train, y_test = split_data(df)
    feature_names = list(X_train.columns)
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # ── 2. Fit preprocessing pipeline on X_train only (no leakage) ────────────
    logger.info("Fitting preprocessing pipeline on X_train...")
    prep_pipeline = PreprocessingPipeline()
    X_train_proc = prep_pipeline.fit_transform(X_train)
    X_test_proc  = prep_pipeline.transform(X_test)

    # ── 3. Re-train final model with best tuned parameters ────────────────────
    logger.info("Re-training final LightGBM with best tuned hyperparameters...")
    final_model = LGBMClassifier(
        class_weight='balanced',
        colsample_bytree=0.7,
        learning_rate=0.01,
        max_depth=5,
        n_estimators=100,
        subsample=0.8,
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_train_proc, y_train)

    # ── 4. Evaluate on held-out test set ─────────────────────────────────────
    logger.info("Evaluating final model on test set...")
    y_pred      = final_model.predict(X_test_proc)
    y_prob      = final_model.predict_proba(X_test_proc)[:, 1]
    test_metrics = {
        "Accuracy":  round(accuracy_score(y_test, y_pred),  6),
        "Precision": round(precision_score(y_test, y_pred), 6),
        "Recall":    round(recall_score(y_test, y_pred),    6),
        "F1_Score":  round(f1_score(y_test, y_pred),        6),
        "ROC_AUC":   round(roc_auc_score(y_test, y_prob),   6),
    }
    logger.info(f"Test Metrics: {test_metrics}")

    # ── 5. Serialize artifacts ─────────────────────────────────────────────────
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    prep_path  = models_dir / "preprocessing_pipeline.joblib"
    model_path = models_dir / "best_churn_model.joblib"

    joblib.dump(prep_pipeline, prep_path)
    logger.info(f"Saved preprocessing pipeline -> {prep_path}")

    joblib.dump(final_model, model_path)
    logger.info(f"Saved final model           -> {model_path}")

    # ── 6. Unified sklearn Pipeline & inference bundle ────────────────────────
    unified_pipeline = Pipeline([
        ('scaler', prep_pipeline),
        ('classifier', final_model)
    ])
    unified_path = models_dir / "unified_churn_pipeline.joblib"
    joblib.dump(unified_pipeline, unified_path)
    logger.info(f"Saved unified sklearn pipeline -> {unified_path}")

    # We store a combined dict for flexible loading in the FastAPI service
    combined_artifact = {
        "pipeline": unified_pipeline,
        "preprocessing_pipeline": prep_pipeline,
        "classifier": final_model,
        "feature_names": feature_names,
    }
    combined_path = models_dir / "full_inference_pipeline.joblib"
    joblib.dump(combined_artifact, combined_path)
    logger.info(f"Saved combined artifact     -> {combined_path}")

    # ── 7. Write model_metadata.json ──────────────────────────────────────────
    metadata = {
        "model_name": "LightGBMChurnClassifier",
        "model_version": "1.0.0",
        "training_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "algorithm": "LightGBM",
        "best_hyperparameters": {
            "class_weight": "balanced",
            "colsample_bytree": 0.7,
            "learning_rate": 0.01,
            "max_depth": 5,
            "n_estimators": 100,
            "subsample": 0.8,
        },
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "test_set_metrics": test_metrics,
        "artifacts": {
            "preprocessing_pipeline": "models/preprocessing_pipeline.joblib",
            "best_churn_model":       "models/best_churn_model.joblib",
            "full_inference_pipeline": "models/full_inference_pipeline.joblib",
            "unified_churn_pipeline": "models/unified_churn_pipeline.joblib",
        },
        "sklearn_version": __import__('sklearn').__version__,
        "lightgbm_version": __import__('lightgbm').__version__,
        "joblib_version": joblib.__version__,
    }

    metadata_path = models_dir / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Saved model metadata        -> {metadata_path}")

    # ── 8. Validation — deserialize and verify prediction consistency ──────────
    logger.info("Validating serialized artifacts...")
    loaded_pipeline = joblib.load(prep_path)
    loaded_model    = joblib.load(model_path)

    X_test_reprocessed = loaded_pipeline.transform(X_test)
    y_pred_reloaded    = loaded_model.predict(X_test_reprocessed)
    y_prob_reloaded    = loaded_model.predict_proba(X_test_reprocessed)[:, 1]

    # Compare predictions — must be identical (up to 6 decimal places)
    import numpy as np
    pred_match = np.array_equal(y_pred, y_pred_reloaded)
    prob_match = np.allclose(y_prob, y_prob_reloaded, atol=1e-6)

    if pred_match and prob_match:
        logger.info("[PASSED] Validation PASSED: deserialized model predictions match in-memory model.")
    else:
        logger.error("[FAILED] Validation FAILED: prediction mismatch detected!")
        raise RuntimeError("Serialization validation failed: prediction mismatch.")

    return metadata, test_metrics


if __name__ == "__main__":
    root = get_project_root()
    metadata, metrics = build_and_serialize(root)
    print("\n=== TASK 13 Serialization Complete ===")
    print(f"Model:   {metadata['model_name']} v{metadata['model_version']}")
    print(f"Trained: {metadata['training_timestamp']}")
    print(f"\nTest Set Metrics:")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.6f}")
    print(f"\nArtifacts saved to: models/")
