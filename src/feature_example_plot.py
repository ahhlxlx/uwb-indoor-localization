import pandas as pd
import matplotlib.pyplot as plt

# Step 1.5 :

# 1. Load the files
# df_raw contains the 1016 CIR columns
# df_enhanced contains the engineered 'rms_delay' column
df_raw = pd.read_csv('../data/processed/cleaned_data.csv')
df_enhanced = pd.read_csv('../data/processed/old_enhanced_features.csv')


def get_median_index(df, target_class):
    # Filter for the class (0 for LOS, 1 for NLOS)
    class_df = df[df['NLOS'] == target_class]
    median_val = class_df['rms_delay'].median()

    # Return the index of the row closest to that median value
    return (class_df['rms_delay'] - median_val).abs().idxmin()


# 2. Find the representative indices
idx_los = get_median_index(df_enhanced, 0)
idx_nlos = get_median_index(df_enhanced, 1)

# 3. Pull raw signals (Assuming CIR data is in columns 15 to 1031)
sig_los = df_raw.iloc[idx_los, 15:1031]
sig_nlos = df_raw.iloc[idx_nlos, 15:1031]

# 4. Plotting
plt.figure(figsize=(12, 6))
plt.plot(sig_los.values, label=f'Median LOS (Row {idx_los})', color='forestgreen', linewidth=1.5)
plt.plot(sig_nlos.values, label=f'Median NLOS (Row {idx_nlos})', color='crimson', alpha=0.8, linewidth=1.5)

plt.title('Comparison of Median Raw CIR Signals (Statistically Representative)')
plt.xlabel('Time Bin')
plt.ylabel('Amplitude')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.savefig('../results/figures/feature_cir_comparison.png')
plt.show()

print(f"Median LOS Index: {idx_los}")
print(f"Median NLOS Index: {idx_nlos}")