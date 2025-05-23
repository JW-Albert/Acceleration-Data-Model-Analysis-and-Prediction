import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                           roc_curve, auc)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_and_evaluate_rf(norm_all_features, selected_features, output_dir=None):
    # 創建輸出目錄
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 創建模型
    rf = RandomForestClassifier(n_estimators=100)
    
    # 訓練模型
    rf.fit(norm_all_features.iloc[:, selected_features], norm_all_features.index)
    
    # 預測
    y_pred = rf.predict(norm_all_features.iloc[:, selected_features])
    health_index = rf.predict_proba(norm_all_features.iloc[:, selected_features])[:, 1]
    
    # 評估
    accuracy = accuracy_score(norm_all_features.index, y_pred)
    cv_scores = cross_val_score(rf, norm_all_features.iloc[:, selected_features], 
                              norm_all_features.index, cv=5)
    
    # 保存結果
    if output_dir:
        # 保存評估指標
        metrics = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, 'metrics.csv'))
        
        # 保存分類報告
        with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
            f.write(classification_report(norm_all_features.index, y_pred))
        
        # 繪製並保存混淆矩陣
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(norm_all_features.index, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Random Forest')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()
        
        # 繪製並保存ROC曲線
        fpr, tpr, _ = roc_curve(norm_all_features.index, health_index)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic - Random Forest')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        plt.close()
        
        # 額外保存特徵重要性圖（Random Forest特有）
        feature_importance = pd.DataFrame({
            'feature': norm_all_features.columns[selected_features],
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
        plt.title('Feature Importance - Random Forest')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        plt.close()
    
    return rf, health_index 