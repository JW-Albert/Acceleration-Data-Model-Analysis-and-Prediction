import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from typing import Tuple, Dict

class TSPEAnalysis:
    def __init__(self, n_components: int = None, alpha: float = 0.05):
        self.n_components = n_components
        self.alpha = alpha
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.t2_limit = None
        self.spe_limit = None
        
    def fit(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Fit the model and calculate control limits.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary containing T² and SPE control limits
        """
        # Standardize and fit PCA
        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)
        
        # Calculate T² limit
        n_samples = X.shape[0]
        p = self.n_components
        f_value = stats.f.ppf(1 - self.alpha, p, n_samples - p)
        self.t2_limit = (p * (n_samples - 1) * (n_samples + 1) * f_value) / (n_samples * (n_samples - p))
        
        # Calculate SPE limit
        X_transformed = self.pca.transform(X_scaled)
        X_reconstructed = self.pca.inverse_transform(X_transformed)
        spe = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)
        self.spe_limit = np.percentile(spe, (1 - self.alpha) * 100)
        
        return {'t2_limit': self.t2_limit, 'spe_limit': self.spe_limit}
    
    def calculate_metrics(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate T² and SPE metrics for data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (T² values, SPE values)
        """
        X_scaled = self.scaler.transform(X)
        X_transformed = self.pca.transform(X_scaled)
        
        # Calculate T²
        t2 = np.sum((X_transformed ** 2) / self.pca.explained_variance_ratio_[:self.n_components], axis=1)
        
        # Calculate SPE
        X_reconstructed = self.pca.inverse_transform(X_transformed)
        spe = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)
        
        return t2, spe
    
    def detect_anomalies(self, X: pd.DataFrame) -> np.ndarray:
        """
        Detect anomalies using T² and SPE metrics.
        
        Returns:
            Boolean array indicating anomalies
        """
        t2, spe = self.calculate_metrics(X)
        return (t2 > self.t2_limit) | (spe > self.spe_limit) 