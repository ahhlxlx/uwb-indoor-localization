import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Using 'Agg' backend to avoid hanging on macOS
import matplotlib
matplotlib.use('Agg')

# 1. Load the data
df = pd.read_csv('../data/processed/enhanced_features.csv')

# 2. Calculate Pearson Correlation
corr_matrix = df.corr()

# 3. Generate the Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, linewidths=0.5)
plt.title("Pearson Correlation Heatmap of UWB Enhanced Features")

# 4. Save the figure
plt.tight_layout()
plt.savefig('../results/figures/correlation_heatmap.png')
print("Heatmap saved as 'correlation_heatmap.png'")

# 5. Print insights to console
print("\n--- Correlation with NLOS ---")
print(corr_matrix['NLOS'].sort_values(ascending=False))

# Identify redundancy
corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
redundant = corr_pairs[(corr_pairs > 0.8) & (corr_pairs < 1.0)]
print("\n--- Redundant Feature Pairs (>0.8) ---")
print(redundant)