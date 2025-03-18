import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns

class AnomalyLogisticRegression:
    def __init__(self, random_state: int = 42):
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,      # 增加最大迭代次數
            tol=1e-4,          # 調整收斂容差
            solver='lbfgs',     # 使用 lbfgs 求解器
            n_jobs=-1          # 使用所有可用的 CPU 核心
        )
        self.metrics = {}
        self.scaler = StandardScaler()
        
    def train(self, X: pd.DataFrame, y: np.ndarray, 
              test_size: float = 0.2) -> Dict[str, float]:
        """
        Train the logistic regression model and evaluate performance.
        
        Args:
            X: Feature matrix
            y: Labels (0 for normal, 1 for anomalous)
            test_size: Proportion of dataset to include in the test split
            
        Returns:
            Dictionary containing performance metrics
        """
        # 標準化數據
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )
        
        # Train the model
        logging.info("Training Logistic Regression model...")
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        logging.info(f"Logistic Regression metrics: {self.metrics}")
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

def train_and_evaluate_lr(norm_all_features, selected_features, output_dir=None):
    """使用邏輯回歸模型進行訓練和評估"""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 準備數據
    X = norm_all_features.iloc[:, selected_features]
    y = norm_all_features.index
    
    # 創建並訓練模型
    lr = AnomalyLogisticRegression()
    metrics = lr.train(X, y)
    
    # 獲取健康指數
    health_index = lr.predict_proba(X)[:, 1]
    
    # 保存結果
    if output_dir:
        # 保存評估指標
        pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, 'metrics.csv'))
        
        # 保存模型參數
        pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': lr.model.coef_[0]
        }).to_csv(os.path.join(output_dir, 'coefficients.csv'))
        
        # 繪製並保存混淆矩陣
        plt.figure(figsize=(8, 6))
        y_pred = lr.predict(X)
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Logistic Regression')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()
        
        # 繪製並保存ROC曲線
        fpr, tpr, _ = roc_curve(y, health_index)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic - Logistic Regression')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        plt.close()
    
    return lr.model, health_index 