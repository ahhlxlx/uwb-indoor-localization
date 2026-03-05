"""
data_cleaning.py

- Load all raw CSVs
- Run EDA (shape, nulls, class balance)
- Remove duplicates and invalid rows
- Detect anomalies using Isolation Forest
- Save cleaned data to data/processed/cleaned_data.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

# ─────────────────────────────────────────────
# COLUMN REFERENCE (from actual dataset)
# ─────────────────────────────────────────────
LABEL_COL = 'NLOS'
RANGE_COL = 'RANGE'
CIR_COLS = [f'CIR{i}' for i in range(1016)]
SCALAR_FEATURE_COLS = [
    'FP_IDX', 'FP_AMP1', 'FP_AMP2', 'FP_AMP3',
    'STDEV_NOISE', 'CIR_PWR', 'MAX_NOISE', 'RXPACC',
    'CH', 'FRAME_LEN', 'PREAM_LEN', 'BITRATE', 'PRFR'
]

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load_all_csvs(folder='../data/raw/'):
    """Load and combine all CSV files from the raw folder."""
    all_dfs = []
    for file in sorted(os.listdir(folder)):
        if file.endswith('.csv'):
            path = os.path.join(folder, file)
            df = pd.read_csv(path, header=0)
            df['source_file'] = file  # track which environment each row came from
            all_dfs.append(df)
            print(f"Loaded {file}: {df.shape[0]} rows")

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal combined shape: {combined.shape}")
    return combined

# ─────────────────────────────────────────────
# EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────
def run_eda(df):
    """Print basic EDA info about the dataset."""
    print("\n========== EDA ==========")
    print(f"Shape: {df.shape}")
    print(f"\nScalar feature columns: {SCALAR_FEATURE_COLS}")
    print(f"CIR columns: CIR0 to CIR1015 ({len(CIR_COLS)} total)")
    print(f"\nMissing values (scalar features only):")
    print(df[SCALAR_FEATURE_COLS + [LABEL_COL, RANGE_COL]].isnull().sum())
    print(f"\nBasic stats (scalar features):")
    print(df[SCALAR_FEATURE_COLS + [RANGE_COL]].describe())

    # Class balance
    print(f"\nClass balance:")
    counts = df[LABEL_COL].value_counts()
    print(f"  LOS  (NLOS=0): {counts.get(0, 0)} samples")
    print(f"  NLOS (NLOS=1): {counts.get(1, 0)} samples")

# ─────────────────────────────────────────────
# CLEAN
# ─────────────────────────────────────────────
def clean_data(df):
    """Remove duplicates, missing values, and obviously invalid rows."""
    original_size = len(df)

    # Drop duplicates
    df = df.drop_duplicates()
    print(f"\nAfter removing duplicates: {len(df)} rows (removed {original_size - len(df)})")

    # Drop rows with missing values
    df = df.dropna()
    print(f"After dropping nulls: {len(df)} rows")

    # Drop rows where RANGE is negative or zero
    df = df[df[RANGE_COL] > 0]
    print(f"After removing invalid RANGE values: {len(df)} rows")

    # Drop rows where NLOS label is not 0 or 1
    df = df[df[LABEL_COL].isin([0, 1])]
    print(f"After removing invalid NLOS labels: {len(df)} rows")
    return df

# ─────────────────────────────────────────────
# ANOMALY DETECTION
# ─────────────────────────────────────────────
def remove_anomalies(df, contamination=0.05):
    """
    Use Isolation Forest on scalar features only (not CIR).
    CIR columns are too high-dimensional for Isolation Forest.
    contamination=0.05 means we expect ~5% of data to be anomalous.
    """
    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(df[SCALAR_FEATURE_COLS])

    df = df.copy()
    df['anomaly'] = preds  # -1 = anomaly, 1 = normal

    n_anomalies = (df['anomaly'] == -1).sum()
    print(f"\nIsolation Forest detected {n_anomalies} anomalies ({n_anomalies/len(df)*100:.1f}%)")

    df_clean = df[df['anomaly'] == 1].drop(columns=['anomaly'])
    df_anomalies = df[df['anomaly'] == -1].drop(columns=['anomaly'])

    return df_clean, df_anomalies

# ─────────────────────────────────────────────
# VISUALIZE
# ─────────────────────────────────────────────
def plot_class_balance(df_before, df_after, save_path='../results/figures/'):
    """Compare class balance before and after cleaning."""
    os.makedirs(save_path, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    before_counts = df_before[LABEL_COL].value_counts().sort_index()
    before_counts.index = ['LOS (0)', 'NLOS (1)']
    before_counts.plot(kind='bar', ax=axes[0], color=['steelblue', 'coral'])
    axes[0].set_title('Before Cleaning')
    axes[0].set_ylabel('Count')

    after_counts = df_after[LABEL_COL].value_counts().sort_index()
    after_counts.index = ['LOS (0)', 'NLOS (1)']
    after_counts.plot(kind='bar', ax=axes[1], color=['steelblue', 'coral'])
    axes[1].set_title('After Cleaning')
    axes[1].set_ylabel('Count')

    plt.suptitle('Class Balance: LOS vs NLOS')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'class_balance.png'))
    print(f"Saved class balance plot to {save_path}class_balance.png")
    plt.close()

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs('../data/processed', exist_ok=True)
    os.makedirs('../results/figures', exist_ok=True)

    # Load
    df_raw = load_all_csvs('../data/raw/')

    # EDA
    run_eda(df_raw)

    # Basic cleaning
    df_cleaned = clean_data(df_raw)

    # Anomaly removal
    df_final, df_anomalies = remove_anomalies(df_cleaned, contamination=0.05)

    # Visualize before vs after
    plot_class_balance(df_raw, df_final)

    # Save outputs
    df_final.to_csv('../data/processed/cleaned_data.csv', index=False)
    df_anomalies.to_csv('../data/processed/anomalies.csv', index=False)

    print(f"\nCleaned data saved to ../data/processed/cleaned_data.csv")
    print(f"Final shape: {df_final.shape}")
    print(f"Anomalies saved to ../data/processed/anomalies.csv")