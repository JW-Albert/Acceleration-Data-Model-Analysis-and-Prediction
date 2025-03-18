import numpy as np
import pandas as pd
from typing import Tuple, List

class FisherScore:
    def __init__(self):
        self.scores = None
        self.feature_names = None
    
    def calculate_score(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Calculate Fisher Score for each feature.
        
        Args:
            X: Feature matrix
            y: Labels (0 for normal, 1 for anomalous)
            
        Returns:
            Tuple of (fisher scores, feature names)
        """
        self.feature_names = X.columns.tolist()
        n_features = X.shape[1]
        self.scores = np.zeros(n_features)
        
        for i in range(n_features):
            feature = X.iloc[:, i]
            normal_data = feature[y == 0]
            anomaly_data = feature[y == 1]
            
            # 添加數值檢查
            if len(normal_data) == 0 or len(anomaly_data) == 0:
                self.scores[i] = 0
                continue
            
            # 添加 nan 檢查
            if normal_data.isna().any() or anomaly_data.isna().any():
                self.scores[i] = 0
                continue
            
            # Calculate means
            mean_normal = np.mean(normal_data)
            mean_anomaly = np.mean(anomaly_data)
            mean_total = np.mean(feature)
            
            # Calculate variances
            var_normal = np.var(normal_data)
            var_anomaly = np.var(anomaly_data)
            
            # Calculate Fisher Score with safe division
            numerator = (mean_normal - mean_total)**2 + (mean_anomaly - mean_total)**2
            denominator = var_normal + var_anomaly
            
            self.scores[i] = numerator / denominator if denominator != 0 else 0
            
        return self.scores, self.feature_names
    
    def select_features(self, X: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """
        Select features based on Fisher Score threshold.
        
        Args:
            X: Feature matrix
            threshold: Minimum Fisher Score to keep feature
            
        Returns:
            DataFrame with selected features
        """
        if self.scores is None:
            raise ValueError("Calculate Fisher Scores first using calculate_score()")
            
        selected_features = [name for score, name in zip(self.scores, self.feature_names)
                           if score >= threshold]
        return X[selected_features] 