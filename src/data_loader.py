import os
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

from src.utils import get_project_root, get_logger

logger = get_logger("data_loader")

def load_raw_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw telecom customer dataset safely into a Pandas DataFrame.
    
    Args:
        data_path (Optional[str]): Path to raw dataset. Defaults to <project_root>/data/raw/data_23.csv.
        
    Returns:
        pd.DataFrame: Raw dataset DataFrame.
        
    Raises:
        FileNotFoundError: If the specified raw data file does not exist.
    """
    project_root = get_project_root()
    
    if data_path is None:
        primary_path = project_root / "data" / "raw" / "data_23.csv"
        fallback_path = project_root / "data_23.csv"
        
        if primary_path.exists():
            resolved_path = primary_path
        elif fallback_path.exists():
            resolved_path = fallback_path
        else:
            raise FileNotFoundError(
                f"Raw data file not found at {primary_path} or {fallback_path}."
            )
    else:
        resolved_path = Path(data_path)
        if not resolved_path.is_absolute():
            resolved_path = project_root / resolved_path
            
    if not resolved_path.exists():
        raise FileNotFoundError(f"Raw dataset file not found at {resolved_path}")
        
    logger.info(f"Loading raw data from: {resolved_path}")
    df = pd.read_csv(resolved_path)
    logger.info(f"Successfully loaded dataset with shape: {df.shape} (rows={df.shape[0]}, cols={df.shape[1]})")
    
    return df

def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate structured summary metadata for an ingested DataFrame.
    
    Args:
        df (pd.DataFrame): Target DataFrame.
        
    Returns:
        Dict[str, Any]: Summary metrics dictionary.
    """
    total_rows, total_cols = df.shape
    null_counts = df.isnull().sum().to_dict()
    cols_with_nulls = {k: v for k, v in null_counts.items() if v > 0}
    duplicate_rows = int(df.duplicated().sum())
    
    summary = {
        "num_rows": total_rows,
        "num_columns": total_cols,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "duplicate_rows": duplicate_rows,
        "columns_with_missing": cols_with_nulls,
        "total_missing_values": int(sum(null_counts.values()))
    }
    
    logger.info(f"Dataset Summary: Rows={total_rows}, Cols={total_cols}, Duplicates={duplicate_rows}, Cols with NaNs={len(cols_with_nulls)}")
    return summary

if __name__ == "__main__":
    df = load_raw_data()
    summary = get_dataset_summary(df)
    print("Ingestion Test Successful!")
    print(f"Data Shape: {summary['num_rows']} rows x {summary['num_columns']} columns")
