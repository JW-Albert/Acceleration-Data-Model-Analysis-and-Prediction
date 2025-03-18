import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple

class PCAAnalysis:
    def __init__(self, n_components: int = None):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.explained_variance_ratio_ = None
        
    def fit_transform(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit PCA and transform data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (transformed data, explained variance ratios)
        """
        # Standardize the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit and transform PCA
        X_pca = self.pca.fit_transform(X_scaled)
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        
        return X_pca, self.explained_variance_ratio_
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted PCA."""
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)
    
    def inverse_transform(self, X_pca: np.ndarray) -> np.ndarray:
        """Inverse transform PCA data back to original space."""
        X_scaled = self.pca.inverse_transform(X_pca)
        return self.scaler.inverse_transform(X_scaled) 