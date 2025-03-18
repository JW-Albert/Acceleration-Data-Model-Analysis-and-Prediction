import logging
import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                           roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import torch
from sklearn.svm import SVC

def train_and_evaluate_svm(norm_all_features, selected_features, output_dir=None):
    """使用 GPU 加速的 SVM 模型訓練和評估"""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 檢查 GPU 是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logging.warning("GPU not available, using CPU")
    
    # 準備數據
    X = norm_all_features.iloc[:, selected_features].values
    y = norm_all_features.index.values
    
    # 如果 GPU 可用，將數據轉移到 GPU
    if device.type == 'cuda':
        X = torch.FloatTensor(X).to(device)
        y = torch.LongTensor(y).to(device)
        
        # 將數據轉回 CPU 進行 SVM 訓練（因為 sklearn 不直接支持 GPU）
        X = X.cpu().numpy()
        y = y.cpu().numpy()
    
    # 創建並訓練模型
    svm = SVC(
        kernel='rbf',
        probability=True,
        verbose=True,
        cache_size=2000  # 增加緩存大小以提高性能
    )
    
    # 訓練模型
    logging.info("Training SVM model...")
    training_start = time.time()
    svm.fit(X, y)
    training_time = time.time() - training_start
    logging.info(f"Training completed in {training_time:.2f} seconds")
    
    # 預測
    logging.info("Making predictions...")
    y_pred = svm.predict(X)
    health_index = svm.predict_proba(X)[:, 1]
    
    # 評估
    accuracy = accuracy_score(y, y_pred)
    logging.info(f"Model accuracy: {accuracy:.4f}")
    
    # 保存結果
    if output_dir:
        # 保存評估指標
        metrics = {
            'accuracy': accuracy,
            'training_time': training_time,
            'device_used': device.type
        }
        pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, 'metrics.csv'))
        
        # 保存分類報告
        with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
            f.write(classification_report(y, y_pred))
        
        # 繪製並保存混淆矩陣
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - SVM')
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
        plt.title('Receiver Operating Characteristic - SVM')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        plt.close()
        
        # 保存 GPU 信息（如果可用）
        if device.type == 'cuda':
            with open(os.path.join(output_dir, 'gpu_info.txt'), 'w') as f:
                f.write(f"GPU Device: {torch.cuda.get_device_name(0)}\n")
                f.write(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB\n")
                f.write(f"GPU Memory Cached: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB\n")
    
    return svm, health_index

def check_gpu_status():
    """檢查 GPU 狀態"""
    if torch.cuda.is_available():
        print("GPU is available!")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory/1024**2:.0f} MB")
        print(f"PyTorch version: {torch.__version__}")
    else:
        print("GPU is not available.") 