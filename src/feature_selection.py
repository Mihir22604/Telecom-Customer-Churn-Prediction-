import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.feature_selection import mutual_info_classif, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from typing import Tuple, List, Dict

logger = logging.getLogger(__name__)

class FeatureSelector:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1, class_weight='balanced')
        self.selected_features_ = None
        self.feature_importances_ = None

    def compute_mutual_info(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        logger.info("Computing Mutual Information Scores...")
        mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
        mi_series = pd.Series(mi_scores, index=X.columns, name="Mutual_Information")
        return mi_series.sort_values(ascending=False)

    def compute_rf_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        logger.info("Computing Random Forest Feature Importances...")
        self.rf_model.fit(X, y)
        importances = self.rf_model.feature_importances_
        rf_series = pd.Series(importances, index=X.columns, name="RF_Importance")
        return rf_series.sort_values(ascending=False)

    def perform_rfecv(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], pd.DataFrame]:
        logger.info("Performing Recursive Feature Elimination with Cross-Validation (RFECV)...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        rfecv = RFECV(estimator=self.rf_model, step=1, cv=cv, scoring='f1', n_jobs=-1, min_features_to_select=10)
        rfecv.fit(X, y)
        
        selected_features = list(X.columns[rfecv.support_])
        logger.info(f"RFECV selected {len(selected_features)} out of {X.shape[1]} features.")
        
        # Create RFECV ranking dataframe
        ranking_df = pd.DataFrame({
            'Feature': X.columns,
            'RFECV_Rank': rfecv.ranking_,
            'RFECV_Selected': rfecv.support_
        }).sort_values('RFECV_Rank')
        
        self.selected_features_ = selected_features
        return selected_features, ranking_df

    def rank_and_select_features(self, X: pd.DataFrame, y: pd.Series, output_path: Path = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Runs Mutual Info, RF Importance, and RFECV to produce a comprehensive ranking.
        """
        mi_scores = self.compute_mutual_info(X, y)
        rf_scores = self.compute_rf_importance(X, y)
        selected_features, rfecv_ranking = self.perform_rfecv(X, y)
        
        # Combine all rankings
        importance_df = pd.DataFrame(index=X.columns)
        importance_df = importance_df.join(mi_scores).join(rf_scores)
        importance_df = importance_df.join(rfecv_ranking.set_index('Feature'))
        
        # Sort by RFECV Rank first, then RF Importance
        importance_df = importance_df.sort_values(by=['RFECV_Rank', 'RF_Importance'], ascending=[True, False])
        self.feature_importances_ = importance_df
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            importance_df.to_csv(output_path)
            logger.info(f"Saved feature importance ranking to: {output_path}")
            
        return importance_df, selected_features

def run_feature_selection(data_path: Path, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes the full feature selection pipeline on the engineered data.
    Returns the comprehensive ranking and the pruned dataframe.
    """
    from src.utils import get_logger
    logger = get_logger("feature_selection")
    
    logger.info(f"Loading engineered data from {data_path}")
    df = pd.read_csv(data_path)
    
    if 'Churn' not in df.columns:
        raise ValueError("Target column 'Churn' not found in dataset.")
        
    X = df.drop(columns=['Churn', 'customerID'])
    y = df['Churn']
    
    selector = FeatureSelector()
    ranking_path = output_dir / "reports" / "feature_importance_ranking.csv"
    
    importance_df, selected_features = selector.rank_and_select_features(X, y, output_path=ranking_path)
    
    # Prune dataframe
    cols_to_keep = ['customerID'] + selected_features + ['Churn']
    pruned_df = df[cols_to_keep]
    
    pruned_path = output_dir / "data" / "processed" / "selected_churn_data.csv"
    pruned_df.to_csv(pruned_path, index=False)
    logger.info(f"Saved pruned dataset with {len(selected_features)} features to: {pruned_path}")
    
    return importance_df, pruned_df

if __name__ == "__main__":
    from src.utils import get_project_root
    root = get_project_root()
    data_path = root / "data" / "processed" / "engineered_churn_data.csv"
    run_feature_selection(data_path, root)
