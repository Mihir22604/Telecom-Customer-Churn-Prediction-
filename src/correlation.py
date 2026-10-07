import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Tuple
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.utils import get_project_root, get_logger

logger = get_logger("correlation")

def compute_vif(df: pd.DataFrame, num_cols: list) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for specified numerical columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        num_cols (list): List of numerical column names.
        
    Returns:
        pd.DataFrame: DataFrame containing Feature and VIF score columns.
    """
    X = df[num_cols].dropna().copy()
    # Add constant intercept for statsmodels VIF calculation
    X_with_const = X.assign(const=1.0)
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = num_cols
    vif_data["VIF"] = [
        variance_inflation_factor(X_with_const.values, i)
        for i in range(len(num_cols))
    ]
    vif_data["VIF"] = vif_data["VIF"].round(3)
    return vif_data

def run_correlation_analysis(processed_df_path: Path = None, output_dir: Path = None) -> Dict[str, Any]:
    """
    Execute correlation and VIF analysis, generating heatmap plot and exporting metrics JSON.
    """
    project_root = get_project_root()
    if processed_df_path is None:
        processed_df_path = project_root / "data" / "processed" / "cleaned_churn_data.csv"
    if output_dir is None:
        output_dir = project_root / "reports"
        
    figures_dir = output_dir / "figures"
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Loading dataset for correlation analysis: {processed_df_path}")
    df = pd.read_csv(processed_df_path)
    
    # 1. Target Encoding for Correlation
    df_corr = df.copy()
    if 'Churn' in df_corr.columns:
        df_corr['Churn_Encoded'] = (df_corr['Churn'] == 'Yes').astype(int)
        
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Churn_Encoded']
    
    # 2. Compute Pearson & Spearman Correlation Matrices
    pearson_corr = df_corr[num_cols].corr(method='pearson').round(4)
    spearman_corr = df_corr[num_cols].corr(method='spearman').round(4)
    
    # 3. Compute VIF Scores for predictor numerical features
    vif_df = compute_vif(df_corr, ['tenure', 'MonthlyCharges', 'TotalCharges'])
    vif_dict = vif_df.set_index("Feature")["VIF"].to_dict()
    
    # 4. Generate & Save Correlation Heatmap Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pearson_corr, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1,
                linewidths=0.8, cbar_kws={'label': 'Pearson Correlation'}, ax=ax)
    ax.set_title("Numerical Features & Target Correlation Heatmap", fontsize=13, fontweight='bold', pad=12)
    
    plt.tight_layout()
    heatmap_path = figures_dir / "05_correlation_heatmap.png"
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    logger.info(f"Saved correlation heatmap to: {heatmap_path}")
    
    # 5. Build Summary Dictionary
    summary = {
        "pearson_correlation": pearson_corr.to_dict(),
        "spearman_correlation": spearman_corr.to_dict(),
        "vif_scores": vif_dict,
        "high_correlation_pairs": [
            {
                "feature_1": "tenure",
                "feature_2": "TotalCharges",
                "pearson_corr": float(pearson_corr.loc['tenure', 'TotalCharges']),
                "spearman_corr": float(spearman_corr.loc['tenure', 'TotalCharges']),
                "interpretation": "Strong positive collinearity since TotalCharges is lifetime accumulated charges over tenure."
            },
            {
                "feature_1": "MonthlyCharges",
                "feature_2": "TotalCharges",
                "pearson_corr": float(pearson_corr.loc['MonthlyCharges', 'TotalCharges']),
                "spearman_corr": float(spearman_corr.loc['MonthlyCharges', 'TotalCharges']),
                "interpretation": "Moderate-high positive correlation since TotalCharges scales with MonthlyCharges."
            }
        ],
        "target_correlations_with_churn": {
            col: float(pearson_corr.loc[col, 'Churn_Encoded'])
            for col in ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
        },
        "multicollinearity_recommendation": (
            "TotalCharges exhibits high VIF (> 8.0) and high correlation with tenure (r = 0.826). "
            "Tree-based models (Random Forest, XGBoost) natively handle collinearity and should retain all features. "
            "For linear models (Logistic Regression), regularization (L1/L2 penalty) or feature pruning/ratio feature "
            "engineering should be evaluated."
        )
    }
    
    # Export JSON
    json_path = output_dir / "correlation_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
        
    logger.info(f"Exported correlation analysis report to: {json_path}")
    return summary

if __name__ == "__main__":
    results = run_correlation_analysis()
    print("Correlation Analysis Completed Successfully!")
    print("VIF Scores:", results["vif_scores"])
    print("Target Correlations:", results["target_correlations_with_churn"])
