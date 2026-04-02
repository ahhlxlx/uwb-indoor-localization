import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

# Step 3 of feature importance pipeline
# Prune unneeded features
# Final step before passing over to Classification

def extract_features(df):
    cir_cols = [f'CIR{i}' for i in range(1016)]
    cir_data = df[cir_cols].values

    # Calculate Power (P = amplitude^2)
    power = cir_data ** 2
    total_power = np.sum(power, axis=1, keepdims=True)

    # Create a time index (0, 1, 2... 1015)
    t = np.arange(1016)

    # Mean Excess Delay
    mean_delay = np.sum(power * t, axis=1, keepdims=True) / total_power

    # RMS Delay Spread
    rms_delay = np.sqrt(np.sum(power * (t ** 2), axis=1, keepdims=True) / total_power - mean_delay ** 2)

    # New Feature Dataframe
    features = pd.DataFrame()
    features['rms_delay'] = rms_delay.flatten()
    features['kurtosis'] = kurtosis(cir_data, axis=1)
    features['skewness'] = skew(cir_data, axis=1)
    features['peak_amp'] = np.max(cir_data, axis=1)

    return features

# --- EXECUTION ---
# Start from 5.2
df_old = pd.read_csv('../data/processed/old_enhanced_features.csv')

# Define the "Optimised+" set (The middle ground)
optimised_plus_cols = [
    'NLOS', 'RANGE', 'FP_IDX', 'FP_AMP1',
       'STDEV_NOISE', 'CIR_PWR', 'MAX_NOISE', 'RXPACC',
        'rms_delay', 'kurtosis', 'peak_amp'
]

df_final = df_old[optimised_plus_cols]

# Save the much smaller, high-value dataset
df_final.to_csv('../data/processed/enhanced_features.csv', index=False)

print("Success! Final dataset created with labels.")
print(f"Columns: {df_final.columns.tolist()}")
print(f"Shape: {df_final.shape}")