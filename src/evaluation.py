import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
)
from src.utils import get_logger

logger = get_logger("evaluation")

def evaluate_models(models: dict, X_test: pd.DataFrame, y_test: pd.Series, output_dir: Path):
    """
    Evaluates multiple trained models on test set and plots ROC, PR curves, and Confusion Matrices.
    """
    metrics = []
    
    fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
    fig_pr, ax_pr = plt.subplots(figsize=(10, 8))
    
    # Calculate grid size for confusion matrices
    n_models = len(models)
    cols = 3
    rows = int(np.ceil(n_models / cols))
    fig_cm, axes_cm = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes_cm = axes_cm.flatten()
    
    for i, (name, model) in enumerate(models.items()):
        logger.info(f"Evaluating {name} on test set...")
        
        # Predictions
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)
            
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        metrics.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1_Score': f1,
            'ROC_AUC': auc
        })
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax_roc.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        # PR Curve
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
        ax_pr.plot(recall_vals, precision_vals, label=f'{name} (F1 = {f1:.3f})')
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_cm[i])
        axes_cm[i].set_title(f"{name} Confusion Matrix")
        axes_cm[i].set_xlabel("Predicted")
        axes_cm[i].set_ylabel("Actual")
        
    # Formatting ROC plot
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=2)
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Receiver Operating Characteristic (ROC) Curves')
    ax_roc.legend(loc="lower right")
    
    # Formatting PR plot
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curves')
    ax_pr.legend(loc="lower left")
    
    # Hide unused subplots for CMs
    for j in range(i + 1, len(axes_cm)):
        axes_cm[j].set_visible(False)
    
    # Save figures
    fig_dir = output_dir / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    fig_roc.savefig(fig_dir / "model_roc_comparison.png", bbox_inches='tight')
    fig_pr.savefig(fig_dir / "model_pr_comparison.png", bbox_inches='tight')
    fig_cm.savefig(fig_dir / "confusion_matrices.png", bbox_inches='tight')
    
    plt.close('all')
    
    metrics_df = pd.DataFrame(metrics).sort_values(by='F1_Score', ascending=False)
    return metrics_df
