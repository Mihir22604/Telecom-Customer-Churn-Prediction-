import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

from src.utils import get_project_root, get_logger
from src.data_loader import load_raw_data

logger = get_logger("quality_audit")

def run_data_quality_audit(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive data quality audit on raw DataFrame.
    
    Args:
        df (pd.DataFrame): Ingested raw dataset.
        
    Returns:
        Dict[str, Any]: Comprehensive data quality audit dictionary.
    """
    logger.info("Executing Data Quality Audit...")
    total_rows, total_cols = df.shape
    
    # 1. Missing Values Analysis
    missing_counts = df.isnull().sum().to_dict()
    missing_pct = (df.isnull().sum() / total_rows * 100).round(3).to_dict()
    cols_with_missing = {
        col: {"missing_count": int(missing_counts[col]), "missing_pct": float(missing_pct[col])}
        for col in df.columns if missing_counts[col] > 0
    }
    
    # 2. Whitespace & Empty Strings Analysis
    whitespace_counts = {}
    for col in df.select_dtypes(include=['object', 'string', 'str']).columns:
        ws_count = int((df[col].astype(str).str.strip() == '').sum() - df[col].isnull().sum())
        if ws_count > 0:
            whitespace_counts[col] = ws_count
            
    # Specific check for TotalCharges whitespace
    total_charges_blanks = int((df['TotalCharges'] == ' ').sum()) if 'TotalCharges' in df.columns else 0
    tenure_zero_blanks = int(df[(df['tenure'] == 0.0) & (df['TotalCharges'] == ' ')].shape[0]) if ('tenure' in df.columns and 'TotalCharges' in df.columns) else 0
    
    # 3. Duplicate Detection
    exact_duplicates = int(df.duplicated().sum())
    customer_id_duplicates = int(df.duplicated(subset=['customerID']).sum()) if 'customerID' in df.columns else 0
    
    # 4. Target Column Analysis
    target_info = {}
    if 'Churn' in df.columns:
        target_counts = df['Churn'].value_counts(dropna=False).to_dict()
        target_pct = (df['Churn'].value_counts(normalize=True, dropna=False) * 100).round(2).to_dict()
        
        # Convert keys (NaN to string 'NaN' for JSON serialization)
        formatted_counts = {str(k): int(v) for k, v in target_counts.items()}
        formatted_pct = {str(k): float(v) for k, v in target_pct.items()}
        
        target_info = {
            "target_column": "Churn",
            "counts": formatted_counts,
            "percentages": formatted_pct,
            "missing_target_rows": int(df['Churn'].isnull().sum()),
            "imbalance_ratio": f"{formatted_pct.get('No', 0)}% No vs {formatted_pct.get('Yes', 0)}% Yes"
        }
        
    # 5. Data Types Breakdown
    dtypes_summary = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    # Combine audit metrics
    audit_report = {
        "dataset_metrics": {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "exact_duplicate_rows": exact_duplicates,
            "duplicate_customer_ids": customer_id_duplicates
        },
        "missing_values": {
            "total_columns_with_missing": len(cols_with_missing),
            "columns_detail": cols_with_missing
        },
        "whitespace_anomalies": {
            "columns_with_whitespace": whitespace_counts,
            "total_charges_blanks": total_charges_blanks,
            "tenure_zero_total_charges_blanks": tenure_zero_blanks
        },
        "target_analysis": target_info,
        "feature_data_types": dtypes_summary
    }
    
    logger.info(f"Audit Complete: {total_rows} rows, {exact_duplicates} exact duplicates, {total_charges_blanks} blank TotalCharges, {target_info.get('missing_target_rows', 0)} missing target rows.")
    return audit_report

def save_audit_report(audit_data: Dict[str, Any], output_filename: str = "data_quality_audit.json") -> Path:
    """
    Save audit dictionary to JSON file in reports directory.
    
    Args:
        audit_data (Dict[str, Any]): Audit results.
        output_filename (str): Target filename.
        
    Returns:
        Path: Output file path.
    """
    project_root = get_project_root()
    reports_dir = project_root / "reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    output_path = reports_dir / output_filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=4)
        
    logger.info(f"Saved Data Quality Audit Report to: {output_path}")
    return output_path

if __name__ == "__main__":
    df = load_raw_data()
    audit_results = run_data_quality_audit(df)
    save_audit_report(audit_results)
    print("Data Quality Audit Execution Completed Successfully!")
