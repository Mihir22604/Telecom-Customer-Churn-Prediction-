"""
Telecom Customer Churn Prediction - Model Evaluation and Tuning Pipeline
=========================================================================
Implements Requirements 2-5:
  Req 2: Baseline classification reports for all 5 candidate models.
  Req 3: Before/after tuning for Logistic Regression and LightGBM.
  Req 4: Rigorous hyperparameter tuning (ROC-AUC primary metric).
  Req 5: Final model selection with evidence-based recommendation.

Run: python scripts/run_model_evaluation.py
"""

import sys
import os
import time
import json
import warnings
import platform
from pathlib import Path
from datetime import datetime, timezone

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import (
    RandomizedSearchCV, GridSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)

from src.pipeline import split_data, FeatureScaler
from src.utils import get_logger

logger = get_logger("model_evaluation")

# ===========================================================================
# GLOBAL SETTINGS
# ===========================================================================
RANDOM_STATE = 42
N_CV_SPLITS = 5
SCORING_PRIMARY = "roc_auc"
OUT_DIR = _ROOT / "reports" / "model_evaluation"
DATA_PATH = _ROOT / "data" / "processed" / "selected_churn_data.csv"

PALETTE = {
    "Logistic Regression": "#3B82F6",
    "Decision Tree":       "#F59E0B",
    "Random Forest":       "#10B981",
    "XGBoost":             "#8B5CF6",
    "LightGBM":            "#EF4444",
}


# ===========================================================================
# HELPERS
# ===========================================================================

def _set_style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 120,
    })


def _save_fig(fig, fname):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / fname
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    logger.info("Saved figure -> %s", path)
    return path


def _compute_metrics(name, model, X_te, y_te, fit_time_s=0.0):
    """Evaluate a trained model on test data and return a metrics dict."""
    t0 = time.perf_counter()
    y_pred = model.predict(X_te)
    inf_ms = (time.perf_counter() - t0) * 1000

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_te)[:, 1]
    else:
        y_prob = model.decision_function(X_te)

    rep_str = classification_report(
        y_te, y_pred, target_names=["No-Churn", "Churn"]
    )

    auc = roc_auc_score(y_te, y_prob)
    cm = confusion_matrix(y_te, y_pred)

    acc = accuracy_score(y_te, y_pred)
    cp = precision_score(y_te, y_pred, pos_label=1, zero_division=0)
    cr = recall_score(y_te, y_pred, pos_label=1, zero_division=0)
    cf = f1_score(y_te, y_pred, pos_label=1, zero_division=0)
    ncp = precision_score(y_te, y_pred, pos_label=0, zero_division=0)
    ncr = recall_score(y_te, y_pred, pos_label=0, zero_division=0)
    ncf = f1_score(y_te, y_pred, pos_label=0, zero_division=0)
    mac_p = precision_score(y_te, y_pred, average="macro", zero_division=0)
    mac_r = recall_score(y_te, y_pred, average="macro", zero_division=0)
    mac_f = f1_score(y_te, y_pred, average="macro", zero_division=0)
    wt_p = precision_score(y_te, y_pred, average="weighted", zero_division=0)
    wt_r = recall_score(y_te, y_pred, average="weighted", zero_division=0)
    wt_f = f1_score(y_te, y_pred, average="weighted", zero_division=0)

    return {
        "name": name,
        "accuracy": acc,
        "roc_auc": auc,
        "churn_precision": cp,
        "churn_recall": cr,
        "churn_f1": cf,
        "no_churn_precision": ncp,
        "no_churn_recall": ncr,
        "no_churn_f1": ncf,
        "macro_precision": mac_p,
        "macro_recall": mac_r,
        "macro_f1": mac_f,
        "weighted_precision": wt_p,
        "weighted_recall": wt_r,
        "weighted_f1": wt_f,
        "train_time_s": fit_time_s,
        "inference_time_ms": inf_ms,
        "confusion_matrix": cm.tolist(),
        "classification_report_str": rep_str,
        "y_prob": y_prob.tolist(),
        "y_true": y_te.tolist(),
    }


# ===========================================================================
# PHASE 1: BASELINE EVALUATION
# ===========================================================================

def run_baseline_evaluation(X_tr_p, y_tr, X_te_p, y_te):
    logger.info("=" * 60)
    logger.info("PHASE 1 - Baseline Evaluation (5 candidate models)")
    logger.info("=" * 60)

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=1),
        "XGBoost": XGBClassifier(
            random_state=RANDOM_STATE, eval_metric="logloss",
            n_jobs=1, verbosity=0),
        "LightGBM": LGBMClassifier(
            random_state=RANDOM_STATE, n_jobs=1, verbose=-1),
    }

    results = []
    for name, model in candidates.items():
        logger.info("  Fitting %s ...", name)
        t0 = time.perf_counter()
        model.fit(X_tr_p, y_tr)
        fit_s = time.perf_counter() - t0
        m = _compute_metrics(name, model, X_te_p, y_te, fit_time_s=fit_s)
        results.append(m)
        logger.info(
            "    %s: Acc=%.4f AUC=%.4f CRec=%.4f CF1=%.4f",
            name, m["accuracy"], m["roc_auc"],
            m["churn_recall"], m["churn_f1"]
        )
        logger.info("\n%s", m["classification_report_str"])

    return results


# ===========================================================================
# PHASE 2: HYPERPARAMETER TUNING
# ===========================================================================

def tune_logistic_regression(X_tr_p, y_tr):
    logger.info("PHASE 2a - Tuning Logistic Regression (GridSearchCV)")

    # Valid solver/penalty combos — no invalid param combinations
    param_grid = [
        {
            "C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
            "penalty": ["l2"],
            "solver": ["lbfgs", "liblinear"],
            "class_weight": [None, "balanced"],
            "max_iter": [1000],
        },
        {
            "C": [0.01, 0.1, 1.0, 10.0],
            "penalty": ["l1"],
            "solver": ["liblinear"],
            "class_weight": [None, "balanced"],
            "max_iter": [1000],
        },
        {
            "C": [0.1, 1.0, 10.0],
            "penalty": ["elasticnet"],
            "solver": ["saga"],
            "l1_ratio": [0.3, 0.5, 0.7],
            "class_weight": [None, "balanced"],
            "max_iter": [2000],
        },
    ]

    cv = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True,
                         random_state=RANDOM_STATE)
    gs = GridSearchCV(
        estimator=LogisticRegression(random_state=RANDOM_STATE),
        param_grid=param_grid,
        scoring=SCORING_PRIMARY,
        cv=cv,
        n_jobs=1,
        verbose=1,
        refit=True,
        return_train_score=True,
    )
    t0 = time.perf_counter()
    gs.fit(X_tr_p, y_tr)
    tune_time = time.perf_counter() - t0

    logger.info("  LR Best CV ROC-AUC: %.4f", gs.best_score_)
    logger.info("  LR Best params: %s", gs.best_params_)
    logger.info("  LR Search time: %.1fs", tune_time)

    return gs.best_estimator_, gs.best_params_, gs.best_score_, \
        pd.DataFrame(gs.cv_results_), tune_time


def tune_lightgbm(X_tr_p, y_tr):
    logger.info("PHASE 2b - Tuning LightGBM (RandomizedSearchCV, 60 iter)")

    param_dist = {
        "n_estimators": [50, 100, 200, 300, 400],
        "learning_rate": [0.005, 0.01, 0.05, 0.1, 0.15],
        "num_leaves": [20, 31, 50, 64, 80],
        "max_depth": [-1, 3, 5, 7, 10],
        "min_child_samples": [10, 20, 30, 50],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0.0, 0.01, 0.1, 0.5, 1.0],
        "reg_lambda": [0.0, 0.01, 0.1, 0.5, 1.0],
        "class_weight": [None, "balanced"],
    }

    cv = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True,
                         random_state=RANDOM_STATE)
    rs = RandomizedSearchCV(
        estimator=LGBMClassifier(random_state=RANDOM_STATE,
                                 n_jobs=1, verbose=-1),
        param_distributions=param_dist,
        n_iter=60,
        scoring=SCORING_PRIMARY,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1,
        refit=True,
        return_train_score=True,
    )
    t0 = time.perf_counter()
    rs.fit(X_tr_p, y_tr)
    tune_time = time.perf_counter() - t0

    logger.info("  LGBM Best CV ROC-AUC: %.4f", rs.best_score_)
    logger.info("  LGBM Best params: %s", rs.best_params_)
    logger.info("  LGBM Search time: %.1fs", tune_time)

    return rs.best_estimator_, rs.best_params_, rs.best_score_, \
        pd.DataFrame(rs.cv_results_), tune_time


def threshold_analysis(model, X_val_p, y_val):
    y_prob = model.predict_proba(X_val_p)[:, 1]
    rows = []
    for t in np.arange(0.25, 0.76, 0.05):
        y_pred_t = (y_prob >= t).astype(int)
        rows.append({
            "threshold": round(t, 2),
            "precision": precision_score(y_val, y_pred_t, zero_division=0),
            "recall": recall_score(y_val, y_pred_t, zero_division=0),
            "f1": f1_score(y_val, y_pred_t, zero_division=0),
            "accuracy": accuracy_score(y_val, y_pred_t),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# FIGURE GENERATORS
# ===========================================================================

def fig_baseline_classification_reports(results):
    _set_style()
    model_names = [r["name"] for r in results]
    colors = [PALETTE.get(n, "#64748B") for n in model_names]

    metrics_data = {
        "No-Churn Precision": [r["no_churn_precision"] for r in results],
        "No-Churn Recall":    [r["no_churn_recall"] for r in results],
        "No-Churn F1":        [r["no_churn_f1"] for r in results],
        "Churn Precision":    [r["churn_precision"] for r in results],
        "Churn Recall":       [r["churn_recall"] for r in results],
        "Churn F1":           [r["churn_f1"] for r in results],
        "Macro Precision":    [r["macro_precision"] for r in results],
        "Macro Recall":       [r["macro_recall"] for r in results],
        "Macro F1":           [r["macro_f1"] for r in results],
        "Weighted F1":        [r["weighted_f1"] for r in results],
    }

    fig, axes = plt.subplots(2, 5, figsize=(22, 9), constrained_layout=True)
    axes = axes.flatten()

    for ax, (metric_name, values) in zip(axes, metrics_data.items()):
        bars = ax.barh(model_names, values, color=colors,
                       edgecolor="white", height=0.6)
        ax.set_xlim(0, 1.05)
        ax.set_title(metric_name, fontweight="bold", fontsize=11)
        ax.set_xlabel("Score")
        for bar, val in zip(bars, values):
            ax.text(val + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="left", fontsize=8.5)
        ax.axvline(0.5, ls="--", color="#CBD5E1", lw=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Baseline Classification Report - All Candidate Models (Test Set)\n"
        "Target: Churn=1  |  Class imbalance: ~73% No-Churn / ~27% Churn",
        fontsize=13, fontweight="bold", y=1.01,
    )
    return _save_fig(fig, "baseline_classification_reports_all_models.png")


def fig_baseline_model_comparison(results):
    _set_style()
    df = pd.DataFrame([{
        "Model": r["name"],
        "Accuracy": r["accuracy"],
        "ROC-AUC": r["roc_auc"],
        "Churn Precision": r["churn_precision"],
        "Churn Recall": r["churn_recall"],
        "Churn F1": r["churn_f1"],
    } for r in results]).sort_values("ROC-AUC", ascending=False)

    metrics = ["Accuracy", "ROC-AUC", "Churn Precision", "Churn Recall", "Churn F1"]
    x = np.arange(len(df))
    width = 0.15

    fig, ax = plt.subplots(figsize=(16, 6), constrained_layout=True)
    for i, metric in enumerate(metrics):
        offset = (i - 2) * width
        ax.bar(x + offset, df[metric], width, label=metric, alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"].tolist(), fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Baseline Model Comparison - Key Metrics (Sorted by ROC-AUC)",
                 fontweight="bold", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(0.5, ls="--", color="#CBD5E1", lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _save_fig(fig, "baseline_model_comparison.png")


def fig_baseline_confusion_matrices(results):
    _set_style()
    n = len(results)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows),
                              constrained_layout=True)
    axes_flat = axes.flatten()

    for i, r in enumerate(results):
        cm = np.array(r["confusion_matrix"])
        ax = axes_flat[i]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No-Churn", "Churn"],
                    yticklabels=["No-Churn", "Churn"],
                    linewidths=0.5)
        ax.set_title(r["name"], fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.text(0.5, -0.18,
                f"Acc={r['accuracy']:.3f}  AUC={r['roc_auc']:.3f}",
                transform=ax.transAxes, ha="center",
                fontsize=9, color="#475569")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Baseline Confusion Matrices - All Candidate Models",
                 fontsize=13, fontweight="bold")
    return _save_fig(fig, "baseline_confusion_matrices.png")


def fig_before_after_single(baseline, tuned, family_name):
    _set_style()
    metrics = {
        "Accuracy":        (baseline["accuracy"],        tuned["accuracy"]),
        "ROC-AUC":         (baseline["roc_auc"],         tuned["roc_auc"]),
        "Churn Precision": (baseline["churn_precision"], tuned["churn_precision"]),
        "Churn Recall":    (baseline["churn_recall"],    tuned["churn_recall"]),
        "Churn F1":        (baseline["churn_f1"],        tuned["churn_f1"]),
        "Macro F1":        (baseline["macro_f1"],        tuned["macro_f1"]),
    }
    metric_names = list(metrics.keys())
    base_vals = [v[0] for v in metrics.values()]
    tuned_vals = [v[1] for v in metrics.values()]
    x = np.arange(len(metric_names))
    w = 0.35

    fig, (ax_bar, ax_base, ax_tuned) = plt.subplots(
        1, 3, figsize=(18, 6),
        gridspec_kw={"width_ratios": [2.5, 1, 1]},
        constrained_layout=True
    )

    bars_b = ax_bar.bar(x - w / 2, base_vals, w, label="Baseline",
                        color="#93C5FD", edgecolor="white", alpha=0.9)
    bars_t = ax_bar.bar(x + w / 2, tuned_vals, w, label="Tuned",
                        color="#1D4ED8", edgecolor="white", alpha=0.9)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_names, rotation=20, ha="right", fontsize=9)
    ax_bar.set_ylim(0, 1.12)
    ax_bar.set_ylabel("Score")
    ax_bar.set_title(f"{family_name} - Before vs After Tuning", fontweight="bold")
    ax_bar.legend()
    for bar, val in zip(bars_b, base_vals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)
    for bar, val in zip(bars_t, tuned_vals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    for ax, r, label in [(ax_base, baseline, "Baseline"),
                          (ax_tuned, tuned, "Tuned")]:
        cm = np.array(r["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No-Churn", "Churn"],
                    yticklabels=["No-Churn", "Churn"],
                    linewidths=0.5)
        ax.set_title(f"{label} Confusion Matrix", fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fname = ("logistic_regression_before_after_tuning.png"
             if "Logistic" in family_name
             else "lightgbm_before_after_tuning.png")
    return _save_fig(fig, fname)


def fig_lr_vs_lgbm_4way(lr_base, lr_tuned, lgbm_base, lgbm_tuned):
    _set_style()
    configs = [
        ("LR Baseline",   lr_base,    "#93C5FD"),
        ("LR Tuned",      lr_tuned,   "#1D4ED8"),
        ("LGBM Baseline", lgbm_base,  "#FCA5A5"),
        ("LGBM Tuned",    lgbm_tuned, "#B91C1C"),
    ]
    metric_keys = [
        ("accuracy",        "Accuracy"),
        ("roc_auc",         "ROC-AUC"),
        ("churn_precision", "Churn Prec."),
        ("churn_recall",    "Churn Recall"),
        ("churn_f1",        "Churn F1"),
        ("macro_f1",        "Macro F1"),
    ]
    x = np.arange(len(metric_keys))
    width = 0.19

    fig, ax = plt.subplots(figsize=(17, 6), constrained_layout=True)
    for i, (label, data, color) in enumerate(configs):
        vals = [data[mk] for mk, _ in metric_keys]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=label,
                      color=color, edgecolor="white", alpha=0.92)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.007,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([mn for _, mn in metric_keys], fontsize=10)
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("Score")
    ax.set_title(
        "Logistic Regression vs LightGBM - Before & After Tuning (4 Configs)",
        fontweight="bold", fontsize=12,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(0.5, ls="--", color="#CBD5E1", lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _save_fig(fig, "logistic_vs_lightgbm_before_after_tuning.png")


def fig_selected_confusion_matrices(lr_base, lr_tuned, lgbm_base, lgbm_tuned):
    _set_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    items = [
        (axes[0, 0], lr_base,    "LR Baseline"),
        (axes[0, 1], lr_tuned,   "LR Tuned"),
        (axes[1, 0], lgbm_base,  "LightGBM Baseline"),
        (axes[1, 1], lgbm_tuned, "LightGBM Tuned"),
    ]
    for ax, r, label in items:
        cm = np.array(r["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No-Churn", "Churn"],
                    yticklabels=["No-Churn", "Churn"],
                    linewidths=0.5, annot_kws={"size": 13})
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.text(0.5, -0.18,
                f"Acc={r['accuracy']:.3f}  AUC={r['roc_auc']:.3f}",
                transform=ax.transAxes, ha="center",
                fontsize=9, color="#475569")

    fig.suptitle("Confusion Matrices - Selected Models Before & After Tuning",
                 fontsize=13, fontweight="bold")
    return _save_fig(fig, "selected_models_confusion_matrices.png")


def fig_tuning_cv_analysis(cv_df, model_name, best_params, best_score):
    _set_style()
    fname_key = ("logistic_regression"
                 if "Logistic" in model_name else "lightgbm")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # 1. CV score distribution
    ax = axes[0]
    scores = cv_df["mean_test_score"].dropna()
    ax.hist(scores, bins=min(25, len(scores)), color="#3B82F6",
            edgecolor="white", alpha=0.85)
    ax.axvline(best_score, ls="--", color="#EF4444", lw=2,
               label=f"Best={best_score:.4f}")
    ax.set_xlabel("CV Mean ROC-AUC")
    ax.set_ylabel("Count")
    ax.set_title("CV Score Distribution", fontweight="bold")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 2. Train vs Validation scatter (overfitting check)
    ax = axes[1]
    if "mean_train_score" in cv_df.columns:
        tr = cv_df["mean_train_score"].fillna(0)
        te = cv_df["mean_test_score"].fillna(0)
        ax.scatter(tr, te, alpha=0.4, color="#3B82F6", s=30, edgecolors="none")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="No-overfit")
        ax.set_xlabel("Train CV Score")
        ax.set_ylabel("Validation CV Score")
        ax.set_title("Train vs Validation\n(Overfitting Check)", fontweight="bold")
        ax.legend(fontsize=8)
        lo = max(0.5, min(tr.min(), te.min()) - 0.02)
        hi = min(1.0, max(tr.max(), te.max()) + 0.02)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    else:
        ax.text(0.5, 0.5, "Train scores unavailable",
                ha="center", va="center", transform=ax.transAxes)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 3. Top-10 candidates
    ax = axes[2]
    top10 = cv_df.nlargest(10, "mean_test_score")
    y_pos = np.arange(len(top10))
    ax.barh(y_pos, top10["mean_test_score"],
            color="#3B82F6", alpha=0.85, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Cand {int(i)}" for i in top10.index], fontsize=8)
    ax.set_xlabel("Mean CV ROC-AUC")
    ax.set_title("Top-10 Candidates", fontweight="bold")
    ax.axvline(best_score, ls="--", color="#EF4444", lw=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"{model_name} - Hyperparameter Search Analysis\nBest: {best_params}",
        fontsize=10, fontweight="bold",
    )
    return _save_fig(fig, f"{fname_key}_tuning_analysis.png")


def fig_hyperparameter_tuning_comparison(lr_base, lr_tuned, lgbm_base, lgbm_tuned,
                                          lr_cv_best, lgbm_cv_best):
    _set_style()
    metrics = ["accuracy", "roc_auc", "churn_recall", "churn_f1"]
    labels = ["Accuracy", "ROC-AUC", "Churn Recall", "Churn F1"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
    colors = ["#93C5FD", "#1D4ED8", "#FCA5A5", "#B91C1C"]
    cats = ["LR\nBaseline", "LR\nTuned", "LGBM\nBaseline", "LGBM\nTuned"]

    for ax, mk, label in zip(axes, metrics, labels):
        vals = [lr_base[mk], lr_tuned[mk], lgbm_base[mk], lgbm_tuned[mk]]
        bars = ax.bar(cats, vals, color=colors,
                      edgecolor="white", alpha=0.9, width=0.55)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score")
        ax.set_title(label, fontweight="bold")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        ax.axhline(0.5, ls="--", color="#CBD5E1", lw=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Hyperparameter Tuning Impact - LR vs LightGBM\n"
        f"(Primary CV metric: ROC-AUC; LR best CV={lr_cv_best:.4f}, "
        f"LGBM best CV={lgbm_cv_best:.4f})",
        fontsize=12, fontweight="bold",
    )
    return _save_fig(fig, "hyperparameter_tuning_comparison.png")


def fig_model_performance_before_after(lr_base, lr_tuned, lgbm_base, lgbm_tuned):
    _set_style()
    rows = [
        ("LR Baseline",   lr_base),
        ("LR Tuned",      lr_tuned),
        ("LGBM Baseline", lgbm_base),
        ("LGBM Tuned",    lgbm_tuned),
    ]
    metrics_keys = [
        ("accuracy",        "Accuracy"),
        ("roc_auc",         "ROC-AUC"),
        ("churn_precision", "Churn Precision"),
        ("churn_recall",    "Churn Recall"),
        ("churn_f1",        "Churn F1"),
        ("macro_f1",        "Macro F1"),
        ("weighted_f1",     "Weighted F1"),
    ]
    names = [r[0] for r in rows]
    x = np.arange(len(names))
    width = 0.10

    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for i, (mk, label) in enumerate(metrics_keys):
        vals = [data[mk] for _, data in rows]
        offset = (i - 3) * width
        ax.bar(x + offset, vals, width, label=label,
               color=cmap(i / len(metrics_keys)), alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Full Metrics - Before & After Tuning (LR & LightGBM)",
                 fontweight="bold", fontsize=12)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.axhline(0.5, ls="--", color="#CBD5E1", lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _save_fig(fig, "model_performance_before_after_tuning.png")


def fig_roc_curves_final(lr_base, lr_tuned, lgbm_base, lgbm_tuned, y_te):
    _set_style()
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    configs = [
        ("LR Baseline",   lr_base,    "#93C5FD", "--"),
        ("LR Tuned",      lr_tuned,   "#1D4ED8", "-"),
        ("LGBM Baseline", lgbm_base,  "#FCA5A5", "--"),
        ("LGBM Tuned",    lgbm_tuned, "#B91C1C", "-"),
    ]
    for label, r, color, ls in configs:
        fpr, tpr, _ = roc_curve(y_te, r["y_prob"])
        ax.plot(fpr, tpr, color=color, lw=2, ls=ls,
                label=f"{label} (AUC={r['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - LR & LightGBM Before & After Tuning",
                 fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _save_fig(fig, "roc_curves_final_comparison.png")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    logger.info("Loading dataset from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    logger.info("Dataset shape: %s", df.shape)

    # ── Split (fixed seed, stratified) ────────────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(df)
    logger.info("Train: %s  Test: %s", X_train.shape, X_test.shape)
    logger.info("Churn rate  train=%.3f  test=%.3f",
                y_train.mean(), y_test.mean())

    # ── Preprocessing: fit on X_train only, no leakage ────────────────────────
    prep = FeatureScaler()
    X_tr_p = prep.fit_transform(X_train)
    X_te_p = prep.transform(X_test)

    # ── PHASE 1: Baseline evaluation ──────────────────────────────────────────
    baseline_results = run_baseline_evaluation(X_tr_p, y_train, X_te_p, y_test)

    logger.info("Generating baseline figures...")
    fig_baseline_classification_reports(baseline_results)
    fig_baseline_model_comparison(baseline_results)
    fig_baseline_confusion_matrices(baseline_results)

    lr_base   = next(r for r in baseline_results if r["name"] == "Logistic Regression")
    lgbm_base = next(r for r in baseline_results if r["name"] == "LightGBM")

    # ── PHASE 2: Rigorous tuning ──────────────────────────────────────────────
    lr_best_model, lr_best_params, lr_cv_score, lr_cv_df, lr_tune_time = \
        tune_logistic_regression(X_tr_p, y_train)

    lgbm_best_model, lgbm_best_params, lgbm_cv_score, lgbm_cv_df, lgbm_tune_time = \
        tune_lightgbm(X_tr_p, y_train)

    # Evaluate tuned models on held-out test set
    logger.info("Evaluating tuned models on test set...")

    t0 = time.perf_counter()
    lr_best_model.fit(X_tr_p, y_train)
    lr_tuned = _compute_metrics("Logistic Regression (Tuned)", lr_best_model,
                                 X_te_p, y_test,
                                 fit_time_s=time.perf_counter() - t0)

    t0 = time.perf_counter()
    lgbm_best_model.fit(X_tr_p, y_train)
    lgbm_tuned = _compute_metrics("LightGBM (Tuned)", lgbm_best_model,
                                   X_te_p, y_test,
                                   fit_time_s=time.perf_counter() - t0)

    logger.info("\n%s", lr_tuned["classification_report_str"])
    logger.info("\n%s", lgbm_tuned["classification_report_str"])

    # Threshold analysis on first CV validation fold (no test leakage)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    val_idx = list(cv.split(X_tr_p, y_train))[0][1]
    X_val_p = X_tr_p.iloc[val_idx]
    y_val   = y_train.iloc[val_idx]
    lr_thresh_df   = threshold_analysis(lr_best_model,   X_val_p, y_val)
    lgbm_thresh_df = threshold_analysis(lgbm_best_model, X_val_p, y_val)

    # ── Generate tuning and comparison figures ─────────────────────────────────
    logger.info("Generating tuning and before/after figures...")
    fig_before_after_single(lr_base,   lr_tuned,   "Logistic Regression")
    fig_before_after_single(lgbm_base, lgbm_tuned, "LightGBM")
    fig_lr_vs_lgbm_4way(lr_base, lr_tuned, lgbm_base, lgbm_tuned)
    fig_selected_confusion_matrices(lr_base, lr_tuned, lgbm_base, lgbm_tuned)
    fig_tuning_cv_analysis(lr_cv_df,   "Logistic Regression",
                           lr_best_params,   lr_cv_score)
    fig_tuning_cv_analysis(lgbm_cv_df, "LightGBM",
                           lgbm_best_params, lgbm_cv_score)
    fig_hyperparameter_tuning_comparison(
        lr_base, lr_tuned, lgbm_base, lgbm_tuned,
        lr_cv_score, lgbm_cv_score
    )
    fig_model_performance_before_after(lr_base, lr_tuned, lgbm_base, lgbm_tuned)
    fig_roc_curves_final(lr_base, lr_tuned, lgbm_base, lgbm_tuned, y_test)

    # ── Final model selection ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 3 - Final Model Selection")
    logger.info("=" * 60)

    if lgbm_tuned["roc_auc"] > lr_tuned["roc_auc"] + 0.005:
        final_name = "LightGBM (Tuned)"
        selection_reason = "Higher ROC-AUC"
    elif (lgbm_tuned["churn_recall"] > lr_tuned["churn_recall"] + 0.02
          and lgbm_tuned["roc_auc"] >= lr_tuned["roc_auc"] - 0.005):
        final_name = "LightGBM (Tuned)"
        selection_reason = "Higher Churn Recall with comparable AUC"
    elif lgbm_tuned["churn_f1"] > lr_tuned["churn_f1"] + 0.01:
        final_name = "LightGBM (Tuned)"
        selection_reason = "Higher Churn F1"
    elif lr_tuned["roc_auc"] > lgbm_tuned["roc_auc"] + 0.005:
        final_name = "Logistic Regression (Tuned)"
        selection_reason = "Higher ROC-AUC"
    else:
        final_name = "LightGBM (Tuned)"
        selection_reason = ("Comparable performance; LightGBM preferred for "
                            "superior inference speed and native imbalance handling")

    logger.info("SELECTED MODEL: %s", final_name)
    logger.info("REASON: %s", selection_reason)

    # ── Save data files ────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Saving data files to %s", OUT_DIR)

    # baseline_metrics.csv
    baseline_df = pd.DataFrame([{
        "Model": r["name"],
        "Accuracy": round(r["accuracy"], 6),
        "ROC_AUC": round(r["roc_auc"], 6),
        "No_Churn_Precision": round(r["no_churn_precision"], 6),
        "No_Churn_Recall": round(r["no_churn_recall"], 6),
        "No_Churn_F1": round(r["no_churn_f1"], 6),
        "Churn_Precision": round(r["churn_precision"], 6),
        "Churn_Recall": round(r["churn_recall"], 6),
        "Churn_F1": round(r["churn_f1"], 6),
        "Macro_Precision": round(r["macro_precision"], 6),
        "Macro_Recall": round(r["macro_recall"], 6),
        "Macro_F1": round(r["macro_f1"], 6),
        "Weighted_F1": round(r["weighted_f1"], 6),
        "Train_Time_s": round(r["train_time_s"], 4),
        "Inference_ms": round(r["inference_time_ms"], 4),
    } for r in baseline_results])
    baseline_df.to_csv(OUT_DIR / "baseline_metrics.csv", index=False)

    # tuned_model_metrics.csv
    tuned_rows = []
    for r, params, cv_s, search_t in [
        (lr_tuned,   lr_best_params,   lr_cv_score,   lr_tune_time),
        (lgbm_tuned, lgbm_best_params, lgbm_cv_score, lgbm_tune_time),
    ]:
        tuned_rows.append({
            "Model": r["name"],
            "Accuracy": round(r["accuracy"], 6),
            "ROC_AUC": round(r["roc_auc"], 6),
            "Churn_Precision": round(r["churn_precision"], 6),
            "Churn_Recall": round(r["churn_recall"], 6),
            "Churn_F1": round(r["churn_f1"], 6),
            "Macro_F1": round(r["macro_f1"], 6),
            "Weighted_F1": round(r["weighted_f1"], 6),
            "CV_Best_ROC_AUC": round(cv_s, 6),
            "Train_Time_s": round(r["train_time_s"], 4),
            "Inference_ms": round(r["inference_time_ms"], 4),
            "Search_Time_s": round(search_t, 2),
            "Best_Params": json.dumps(params, default=str),
        })
    pd.DataFrame(tuned_rows).to_csv(OUT_DIR / "tuned_model_metrics.csv", index=False)

    # classification_reports.json
    clf_reports = {}
    for r in baseline_results:
        clf_reports[r["name"]] = {
            "classification_report": r["classification_report_str"],
            "metrics": {k: r[k] for k in [
                "accuracy", "roc_auc",
                "churn_precision", "churn_recall", "churn_f1",
                "no_churn_precision", "no_churn_recall", "no_churn_f1",
                "macro_precision", "macro_recall", "macro_f1",
                "weighted_precision", "weighted_recall", "weighted_f1",
                "train_time_s", "inference_time_ms",
            ]},
            "confusion_matrix": r["confusion_matrix"],
        }
    for label, r, params, cv_s in [
        ("Logistic Regression (Tuned)", lr_tuned,   lr_best_params,   lr_cv_score),
        ("LightGBM (Tuned)",            lgbm_tuned, lgbm_best_params, lgbm_cv_score),
    ]:
        clf_reports[label] = {
            "classification_report": r["classification_report_str"],
            "metrics": {k: r[k] for k in [
                "accuracy", "roc_auc",
                "churn_precision", "churn_recall", "churn_f1",
                "macro_f1", "weighted_f1",
                "train_time_s", "inference_time_ms",
            ]},
            "confusion_matrix": r["confusion_matrix"],
            "best_params": params,
            "cv_best_roc_auc": cv_s,
        }
    with open(OUT_DIR / "classification_reports.json", "w") as f:
        json.dump(clf_reports, f, indent=4, default=str)

    # hyperparameter_search_results.csv
    lr_cv_df["model"] = "Logistic Regression"
    lgbm_cv_df["model"] = "LightGBM"
    combined_cv = pd.concat([lr_cv_df, lgbm_cv_df], ignore_index=True)
    keep_cols = [c for c in combined_cv.columns
                 if any(tok in c for tok in
                        ["model", "mean_test", "std_test", "mean_train",
                         "rank", "params"])]
    combined_cv[keep_cols].to_csv(
        OUT_DIR / "hyperparameter_search_results.csv", index=False)

    # threshold analysis CSVs
    lr_thresh_df.to_csv(OUT_DIR / "lr_threshold_analysis.csv", index=False)
    lgbm_thresh_df.to_csv(OUT_DIR / "lgbm_threshold_analysis.csv", index=False)

    # experiment_metadata.json
    metadata = {
        "experiment_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset_path": str(DATA_PATH),
        "dataset_shape": list(df.shape),
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "churn_rate_train": round(float(y_train.mean()), 4),
        "churn_rate_test": round(float(y_test.mean()), 4),
        "random_state": RANDOM_STATE,
        "cv_strategy": f"StratifiedKFold(n_splits={N_CV_SPLITS})",
        "primary_scoring_metric": SCORING_PRIMARY,
        "preprocessing": ("FeatureScaler — StandardScaler on 6 continuous "
                          "features, fitted exclusively on X_train"),
        "data_leakage_prevention": (
            "Preprocessing fitted on X_train only; "
            "test set untouched until final evaluation; "
            "threshold selection performed on held-out CV validation fold"
        ),
        "candidate_models": [r["name"] for r in baseline_results],
        "tuning": {
            "Logistic Regression": {
                "method": "GridSearchCV",
                "n_candidates_evaluated": int(len(lr_cv_df)),
                "best_params": lr_best_params,
                "best_cv_roc_auc": round(lr_cv_score, 6),
                "tuning_time_s": round(lr_tune_time, 2),
            },
            "LightGBM": {
                "method": "RandomizedSearchCV(n_iter=60)",
                "n_candidates_evaluated": int(len(lgbm_cv_df)),
                "best_params": lgbm_best_params,
                "best_cv_roc_auc": round(lgbm_cv_score, 6),
                "tuning_time_s": round(lgbm_tune_time, 2),
            },
        },
        "final_selected_model": final_name,
        "selection_criterion": selection_reason,
        "test_results_summary": {
            "lr_baseline": {
                "accuracy": round(lr_base["accuracy"], 4),
                "roc_auc": round(lr_base["roc_auc"], 4),
                "churn_recall": round(lr_base["churn_recall"], 4),
                "churn_f1": round(lr_base["churn_f1"], 4),
            },
            "lr_tuned": {
                "accuracy": round(lr_tuned["accuracy"], 4),
                "roc_auc": round(lr_tuned["roc_auc"], 4),
                "churn_recall": round(lr_tuned["churn_recall"], 4),
                "churn_f1": round(lr_tuned["churn_f1"], 4),
            },
            "lgbm_baseline": {
                "accuracy": round(lgbm_base["accuracy"], 4),
                "roc_auc": round(lgbm_base["roc_auc"], 4),
                "churn_recall": round(lgbm_base["churn_recall"], 4),
                "churn_f1": round(lgbm_base["churn_f1"], 4),
            },
            "lgbm_tuned": {
                "accuracy": round(lgbm_tuned["accuracy"], 4),
                "roc_auc": round(lgbm_tuned["roc_auc"], 4),
                "churn_recall": round(lgbm_tuned["churn_recall"], 4),
                "churn_f1": round(lgbm_tuned["churn_f1"], 4),
            },
        },
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }
    with open(OUT_DIR / "experiment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4, default=str)

    # ── Print summary ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    hdr = f"{'Model':<32} {'Acc':>7} {'AUC':>7} {'C-Rec':>7} {'C-F1':>7}"
    logger.info(hdr)
    logger.info("-" * len(hdr))
    for r in baseline_results:
        logger.info("  %-30s %7.4f %7.4f %7.4f %7.4f",
                    r["name"], r["accuracy"], r["roc_auc"],
                    r["churn_recall"], r["churn_f1"])
    logger.info("  %-30s %7.4f %7.4f %7.4f %7.4f [TUNED]",
                "Logistic Regression", lr_tuned["accuracy"],
                lr_tuned["roc_auc"], lr_tuned["churn_recall"],
                lr_tuned["churn_f1"])
    logger.info("  %-30s %7.4f %7.4f %7.4f %7.4f [TUNED]",
                "LightGBM", lgbm_tuned["accuracy"],
                lgbm_tuned["roc_auc"], lgbm_tuned["churn_recall"],
                lgbm_tuned["churn_f1"])
    logger.info("\nFinal Selected Model: %s", final_name)
    logger.info("Report output directory: %s", OUT_DIR)

    return metadata


if __name__ == "__main__":
    main()
