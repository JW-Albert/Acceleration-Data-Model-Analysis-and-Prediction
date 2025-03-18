import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from typing import Tuple

class LDAAnalysis:
    def __init__(self, n_components: int = None):
        self.n_components = n_components
        self.lda = LinearDiscriminantAnalysis(n_components=n_components)
        self.scaler = StandardScaler()
        
    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """
        Fit LDA and transform data.
        
        Args:
            X: Feature matrix
            y: Labels (0 for normal, 1 for anomalous)
            
        Returns:
            Transformed data
        """
        # Standardize the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit and transform LDA
        X_lda = self.lda.fit_transform(X_scaled, y)
        
        return X_lda
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted LDA."""
        X_scaled = self.scaler.transform(X)
        return self.lda.transform(X_scaled)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict classes for new data."""
        X_scaled = self.scaler.transform(X)
        return self.lda.predict(X_scaled) 