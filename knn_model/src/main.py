import logging
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 添加父目錄到系統路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 導入所需模組
from utils.data_processing import process_lvm_files, calculate_velocity
from utils.feature_extraction import time_domain, frequency_domain
from models.fisher_score import FisherScore
from models.pca_analysis import PCAAnalysis
from models.lda_analysis import LDAAnalysis
from models.tspe_analysis import TSPEAnalysis
from models.knn_model import train_and_evaluate_knn

def save_fisher_score_analysis(fisher_scores_tuple, feature_names, output_dir):
    """保存 Fisher Score 分析結果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 從元組中解析出分數數組和特徵名稱
    if isinstance(fisher_scores_tuple, tuple):
        fisher_scores = fisher_scores_tuple[0]  # 獲取分數數組
        feature_names = fisher_scores_tuple[1]  # 使用返回的特徵名稱
    else:
        fisher_scores = fisher_scores_tuple
    
    # 打印長度信息以進行調試
    logging.info(f"Fisher scores length: {len(fisher_scores)}")
    logging.info(f"Feature names length: {len(feature_names)}")
    
    # 創建 DataFrame
    scores_df = pd.DataFrame({
        'Feature': feature_names,
        'Fisher_Score': fisher_scores
    })
    
    # 按 Fisher Score 降序排序
    scores_df = scores_df.sort_values('Fisher_Score', ascending=False)
    
    # 保存分數
    scores_df.to_csv(os.path.join(output_dir, 'fisher_scores.csv'), index=False)
    
    # 繪製 Top 20 特徵的 Fisher Score
    plt.figure(figsize=(12, 6))
    top_20_df = scores_df.head(20)
    sns.barplot(data=top_20_df, x='Fisher_Score', y='Feature')
    plt.title('Top 20 Features by Fisher Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_features.png'))
    plt.close()
    
    return scores_df

def save_pca_analysis(pca_obj, X_pca, output_dir):
    """保存 PCA 分析結果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存解釋方差比
    explained_variance_df = pd.DataFrame({
        'Component': range(1, len(pca_obj.explained_variance_ratio_) + 1),
        'Explained_Variance_Ratio': pca_obj.explained_variance_ratio_,
        'Cumulative_Variance_Ratio': np.cumsum(pca_obj.explained_variance_ratio_)
    })
    explained_variance_df.to_csv(os.path.join(output_dir, 'explained_variance.csv'))
    
    # 繪製碎石圖
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca_obj.explained_variance_ratio_) + 1),
             np.cumsum(pca_obj.explained_variance_ratio_), 'bo-')
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.title('Cumulative Explained Variance Ratio')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'scree_plot.png'))
    plt.close()

def save_lda_analysis(lda_obj, X_lda, y, output_dir):
    """保存 LDA 分析結果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存判別係數
    coef_df = pd.DataFrame(lda_obj.coef_, columns=[f'Feature_{i}' for i in range(lda_obj.coef_.shape[1])])
    coef_df.to_csv(os.path.join(output_dir, 'discriminant_coefficients.csv'))
    
    # 繪製 LDA 投影
    if X_lda.shape[1] >= 2:
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y)
        plt.title('LDA Projection')
        plt.xlabel('First Discriminant')
        plt.ylabel('Second Discriminant')
        plt.colorbar(scatter)
        plt.savefig(os.path.join(output_dir, 'lda_projection.png'))
        plt.close()

def save_tspe_analysis(t2, spe, t2_limit, spe_limit, output_dir):
    """保存 T-squared & SPE 分析結果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存指標
    metrics_df = pd.DataFrame({
        'T2': t2,
        'SPE': spe
    })
    metrics_df.to_csv(os.path.join(output_dir, 'tspe_metrics.csv'))
    
    # 繪製 T2 vs SPE 圖
    plt.figure(figsize=(10, 6))
    plt.scatter(t2, spe, alpha=0.5)
    plt.axhline(y=spe_limit, color='r', linestyle='--', label='SPE Limit')
    plt.axvline(x=t2_limit, color='g', linestyle='--', label='T² Limit')
    plt.xlabel('T² Statistic')
    plt.ylabel('SPE')
    plt.title('T² vs SPE Plot')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'tspe_plot.png'))
    plt.close()

def main():
    # 設置日誌
    logging.basicConfig(level=logging.INFO)
    
    # 數據處理參數
    DATA_DIR = "./data"  # 請根據實際情況修改
    UNIT_SIZE = 1024
    FS = 25600
    BASE_FREQ = 24
    N_HARMONICS = 3
    
    # 創建輸出目錄
    output_dir = "model_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 讀取數據
    logging.info("Processing data files...")
    normal_data, anomalous_data = process_lvm_files(DATA_DIR)
    
    # 檢查數據質量
    logging.info("Checking data quality...")
    logging.info(f"Normal data shape: {normal_data.shape}")
    logging.info(f"Anomalous data shape: {anomalous_data.shape}")
    
    # 特徵提取
    logging.info("Extracting features...")
    columns_to_evaluate = ["Acceleration_1_x", "Acceleration_1_y", "Acceleration_1_z"]
    
    # 時域特徵
    normal_time_features = time_domain(normal_data, columns_to_evaluate, UNIT_SIZE)
    anomaly_time_features = time_domain(anomalous_data, columns_to_evaluate, UNIT_SIZE)
    
    # 頻域特徵
    normal_freq_features = frequency_domain(normal_data, columns_to_evaluate, 
                                          FS, BASE_FREQ, N_HARMONICS, UNIT_SIZE)
    anomaly_freq_features = frequency_domain(anomalous_data, columns_to_evaluate,
                                           FS, BASE_FREQ, N_HARMONICS, UNIT_SIZE)
    
    # 合併特徵
    normal_features = pd.concat([normal_time_features, normal_freq_features], axis=1)
    anomaly_features = pd.concat([anomaly_time_features, anomaly_freq_features], axis=1)
    
    # 準備數據集
    X = pd.concat([normal_features, anomaly_features])
    y = np.concatenate([np.zeros(len(normal_features)), np.ones(len(anomaly_features))])
    
    # 處理 NaN 值
    logging.info("Handling missing values...")
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    # 準備用於模型訓練的數據框
    norm_all_features = pd.DataFrame(X)
    norm_all_features.index = y
    selected_features = list(range(X.shape[1]))
    
    # Fisher Score 分析
    logging.info("Performing Fisher Score analysis...")
    fisher = FisherScore()
    fisher_scores = fisher.calculate_score(X, y)
    
    # 保存 Fisher Score 分析結果
    scores_df = save_fisher_score_analysis(
        fisher_scores,  # 分數數組
        X.columns,      # 特徵名稱
        os.path.join(output_dir, 'fisher_score')
    )
    
    # PCA 分析
    logging.info("Performing PCA analysis...")
    pca = PCAAnalysis(n_components=0.95)  # 保留95%的方差
    X_pca = pca.fit_transform(X)
    save_pca_analysis(pca.pca, X_pca, os.path.join(output_dir, 'pca'))
    
    # LDA 分析
    logging.info("Performing LDA analysis...")
    lda = LDAAnalysis()
    X_lda = lda.fit_transform(X, y)
    save_lda_analysis(lda.lda, X_lda, y, os.path.join(output_dir, 'lda'))
    
    # T-squared & SPE 分析
    logging.info("Performing T-squared & SPE analysis...")
    tspe = TSPEAnalysis(n_components=5)
    control_limits = tspe.fit(X)
    t2, spe = tspe.calculate_metrics(X)
    save_tspe_analysis(t2, spe, tspe.t2_limit, tspe.spe_limit,
                      os.path.join(output_dir, 'tspe'))
    
    # 訓練各個模型
    logging.info("Training models...")
    
    # KNN
    logging.info("Training KNN model...")
    knn_model, knn_health_index = train_and_evaluate_knn(
        norm_all_features, selected_features, 
        os.path.join(output_dir, 'knn')
    )
    
    # 保存健康指數
    logging.info("Saving health indices...")
    health_indices = pd.DataFrame({
        'KNN': knn_health_index
    })
    health_indices.to_csv(os.path.join(output_dir, 'health_indices.csv'))
    
    logging.info("All analyses and models have been completed!")

if __name__ == "__main__":
    main() 