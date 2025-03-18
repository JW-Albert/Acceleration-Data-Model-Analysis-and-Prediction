import os
import logging
import pandas as pd
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, filtfilt

def read_lvm(path: str) -> pd.DataFrame:
    """Read LVM file and return DataFrame."""
    try:
        logging.info("Reading data from %s.", path)
        df = pd.read_csv(
            path,
            sep="\t",
            skiprows=23,
            header=None,
            names=[
                "Time",
                "Voltage",
                "Acceleration_2_x",  # z-axis
                "Acceleration_2_y",  # x-axis
                "Acceleration_2_z",  # y-axis
                "Acceleration_1_x",  # x-axis
                "Acceleration_1_y",  # y-axis
                "Acceleration_1_z",  # z-axis
            ],
        )
        # 檢查數據完整性
        if df.isna().any().any():
            logging.warning(f"Missing values found in {path}")
            df = df.fillna(method='ffill')  # 使用前向填充處理缺失值
        return df
    except Exception as e:
        logging.error(f"Error reading file {path}: {str(e)}")
        return pd.DataFrame()  # 返回空的 DataFrame

def process_lvm_files(data_dir: str):
    """Process all LVM files in directory."""
    normal_files = []
    anomalous_files = []

    first_level_dirs = [d for d in os.listdir(data_dir) 
                       if os.path.isdir(os.path.join(data_dir, d))]

    for dir_name in first_level_dirs:
        dir_path = os.path.join(data_dir, dir_name)
        if "normal" in dir_name:
            for root, _, files in os.walk(dir_path):
                normal_files.extend([os.path.join(root, f) for f in files if f.endswith(".lvm")])
        else:
            for root, _, files in os.walk(dir_path):
                anomalous_files.extend([os.path.join(root, f) for f in files if f.endswith(".lvm")])

    normal_data = [read_lvm(file) for file in normal_files]
    anomalous_data = [read_lvm(file) for file in anomalous_files]

    all_normal_data = pd.concat(normal_data, ignore_index=True) if normal_data else pd.DataFrame()
    all_anomalous_data = pd.concat(anomalous_data, ignore_index=True) if anomalous_data else pd.DataFrame()

    return all_normal_data, all_anomalous_data

def calculate_velocity(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate velocity from acceleration data."""
    velocity = cumulative_trapezoid(
        data["Acceleration_1_x"],
        initial=0,
    )

    cutoff_freq = 0.1
    nyquist_freq = 0.5 / (data["Time"][1] - data["Time"][0])
    b, a = butter(2, cutoff_freq / nyquist_freq, btype="high")
    corrected_velocity = filtfilt(b, a, velocity)

    return pd.DataFrame({"Velocity": corrected_velocity}) 