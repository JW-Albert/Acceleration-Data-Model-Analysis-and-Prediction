import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.fftpack import fft

def time_domain(data: pd.DataFrame, columns_to_evaluate: list, unit_size: int) -> pd.DataFrame:
    """Extract time domain features."""
    time_domain_features = {
        "_rms": lambda x: np.sqrt(np.mean(x**2)),
        "_mean": np.mean,
        "_std": np.std,
        "_peak_to_peak": lambda x: np.max(x) - np.min(x),
        "_kurtosis": kurtosis,
        "_skewness": skew,
        "_crest_indicator": lambda x: abs(np.max(x)) / np.sqrt(np.mean(x**2)),
        "_clearance_indicator": lambda x: abs(np.max(x)) / np.mean(np.sqrt(abs(x))) ** 2,
        "_shape_indicator": lambda x: np.sqrt(np.mean(x**2)) / np.mean(abs(x)),
        "_impulse_indicator": lambda x: abs(np.max(x)) / np.mean(abs(x)),
    }

    results = []
    for start in range(0, data.shape[0], unit_size):
        unit_data = data.iloc[start:start + unit_size]
        unit_results = [
            feature_function(unit_data[column])
            for column in columns_to_evaluate
            for feature_function in time_domain_features.values()
        ]
        results.append(unit_results)

    feature_names = [
        f"{column}{feature_name}"
        for column in columns_to_evaluate
        for feature_name in time_domain_features
    ]

    return pd.DataFrame(results, columns=feature_names)

def frequency_domain(data: pd.DataFrame, columns_to_frequency: list, 
                    fs: int, base_freq: int, n: int, unit_size: int) -> pd.DataFrame:
    """Extract frequency domain features."""
    all_units_features = []

    for start in range(0, data.shape[0], unit_size):
        unit_data = data.iloc[start:start + unit_size]
        frequency_domain_features = []
        df = fs / len(unit_data)
        freq = np.linspace(0, len(unit_data) // 2 - 1, len(unit_data) // 2) * df

        for column in columns_to_frequency:
            fft_data = abs(fft(unit_data[column].values)) * 2 / unit_data.shape[0]
            fft_data = pd.DataFrame(fft_data[:len(unit_data) // 2], index=freq)

            for i in range(1, n + 1):
                target_freq = base_freq * i
                max_value = fft_data.loc[target_freq - 8:target_freq + 8].max()
                feature_value = float(max_value.iloc[0]) if isinstance(max_value, pd.Series) else float(max_value)
                frequency_domain_features.append(feature_value)

        all_units_features.append(frequency_domain_features)

    feature_names = [f"{column}_freq_{i}" 
                    for column in columns_to_frequency 
                    for i in range(1, n + 1)]

    return pd.DataFrame(all_units_features, columns=feature_names) 