import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive background backend for server/script plotting
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any

from src.utils import get_project_root, get_logger

logger = get_logger("eda")

# Set global plotting aesthetic style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

def generate_eda_figures(processed_df_path: Path = None, figures_dir: Path = None) -> Dict[str, Path]:
    """
    Generate and save high-resolution EDA figures for telecom customer churn dataset.
    
    Args:
        processed_df_path (Path): Path to cleaned dataset CSV.
        figures_dir (Path): Output directory for saved plots.
        
    Returns:
        Dict[str, Path]: Dictionary mapping figure names to saved file paths.
    """
    project_root = get_project_root()
    if processed_df_path is None:
        processed_df_path = project_root / "data" / "processed" / "cleaned_churn_data.csv"
    if figures_dir is None:
        figures_dir = project_root / "reports" / "figures"
        
    os.makedirs(figures_dir, exist_ok=True)
    
    logger.info(f"Loading cleaned dataset for EDA from: {processed_df_path}")
    df = pd.read_csv(processed_df_path)
    
    saved_figures = {}
    
    # 1. Target Class Distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    churn_counts = df['Churn'].value_counts()
    churn_pct = df['Churn'].value_counts(normalize=True) * 100
    
    bars = ax.bar(churn_counts.index, churn_counts.values, color=['#2ecc71', '#e74c3c'], width=0.5, edgecolor='black')
    ax.set_title("Target Class Distribution (Churn)", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Customer Count", fontsize=12)
    ax.set_xlabel("Churn Status", fontsize=12)
    
    for bar in bars:
        height = bar.get_height()
        pct = (height / len(df)) * 100
        ax.annotate(f"{height:,}\n({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                    xytext=(0, 0), textcoords="offset points",
                    ha='center', va='center', fontsize=11, fontweight='bold', color='white')
                    
    plt.tight_layout()
    target_fig_path = figures_dir / "01_target_distribution.png"
    plt.savefig(target_fig_path, dpi=300)
    plt.close()
    saved_figures["target_distribution"] = target_fig_path
    
    # 2. Numerical Feature Distributions (Histograms + Box Plots)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    for i, col in enumerate(num_cols):
        # Histogram with KDE
        sns.histplot(data=df, x=col, hue='Churn', kde=True, ax=axes[0, i], palette=['#2ecc71', '#e74c3c'], element="step")
        axes[0, i].set_title(f"Distribution of {col}", fontweight='bold')
        
        # Box Plot vs Churn
        sns.boxplot(data=df, x='Churn', y=col, hue='Churn', ax=axes[1, i], palette=['#2ecc71', '#e74c3c'], legend=False)
        axes[1, i].set_title(f"{col} vs Churn", fontweight='bold')
        
    plt.suptitle("Numerical Feature Distributions and Churn Breakdown", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    num_fig_path = figures_dir / "02_numerical_distributions.png"
    plt.savefig(num_fig_path, dpi=300)
    plt.close()
    saved_figures["numerical_distributions"] = num_fig_path
    
    # 3. Categorical Key Drivers vs Churn (Count Plots)
    cat_cols = ['Contract', 'InternetService', 'PaymentMethod', 'TechSupport', 'OnlineSecurity', 'SeniorCitizen']
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()
    
    for i, col in enumerate(cat_cols):
        sns.countplot(data=df, x=col, hue='Churn', ax=axes[i], palette=['#2ecc71', '#e74c3c'])
        axes[i].set_title(f"Churn by {col}", fontweight='bold')
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Count")
        if col in ['PaymentMethod', 'Contract']:
            axes[i].tick_params(axis='x', rotation=15)
            
    plt.suptitle("Categorical Key Features vs Churn Rate", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    cat_fig_path = figures_dir / "03_categorical_vs_churn.png"
    plt.savefig(cat_fig_path, dpi=300)
    plt.close()
    saved_figures["categorical_vs_churn"] = cat_fig_path
    
    # 4. Multivariate Analysis: Tenure vs MonthlyCharges colored by Churn
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='tenure', y='MonthlyCharges', hue='Churn', alpha=0.6, palette=['#2ecc71', '#e74c3c'], ax=ax)
    ax.set_title("Multivariate Interaction: Tenure vs Monthly Charges by Churn", fontsize=14, fontweight='bold')
    ax.set_xlabel("Tenure (Months)", fontsize=12)
    ax.set_ylabel("Monthly Charges ($)", fontsize=12)
    
    plt.tight_layout()
    mv_fig_path = figures_dir / "04_multivariate_tenure_vs_charges.png"
    plt.savefig(mv_fig_path, dpi=300)
    plt.close()
    saved_figures["multivariate_tenure_charges"] = mv_fig_path
    
    logger.info(f"Successfully generated {len(saved_figures)} EDA figures in: {figures_dir}")
    return saved_figures

def compute_eda_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute key statistical summaries for continuous and categorical features by Churn group.
    """
    summary = {}
    
    # Numerical summaries by Churn status
    num_summary = {}
    for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        num_summary[col] = {
            'mean': df.groupby('Churn')[col].mean().to_dict(),
            'median': df.groupby('Churn')[col].median().to_dict(),
            'std': df.groupby('Churn')[col].std().to_dict()
        }
    
    # Churn rates by contract
    contract_churn = (df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack() * 100).round(2).to_dict()
    
    # Churn rates by internet service
    internet_churn = (df.groupby('InternetService')['Churn'].value_counts(normalize=True).unstack() * 100).round(2).to_dict()
    
    summary = {
        "numerical_by_churn": num_summary,
        "churn_rate_by_contract": contract_churn,
        "churn_rate_by_internet_service": internet_churn
    }
    return summary

if __name__ == "__main__":
    figures = generate_eda_figures()
    print("EDA Figure Generation Execution Completed Successfully!")
    for name, path in figures.items():
        print(f" - {name}: {path}")
