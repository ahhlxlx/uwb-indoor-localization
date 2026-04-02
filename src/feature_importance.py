import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap
import sys

# Step 2 of feature engineering pipeline

# 1. Load Data
print("Loading data...")
df = pd.read_csv('../data/processed/old_enhanced_features.csv').dropna()
X = df.drop('NLOS', axis=1)
y = df['NLOS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Train Model
print("Training Model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# --- PART 3: CALCULATE SHAP (SAFE VERSION) ---
print("Calculating SHAP values...")
explainer = shap.TreeExplainer(rf_model)
X_sample = X_test.head(100)
shap_values = explainer.shap_values(X_sample, check_additivity=False)

# SHAP version/model handling to prevent AssertionError
if isinstance(shap_values, list):
    # Format 1: List of arrays [class0, class1]
    # We want index 1 for the 'NLOS' class
    shap_to_plot = shap_values[1]
elif len(shap_values.shape) == 3:
    # Format 2: 3D Array (samples, features, classes)
    # We want the slice for the second class index
    shap_to_plot = shap_values[:, :, 1]
else:
    # Format 3: Single 2D array (already correctly shaped)
    shap_to_plot = shap_values

# Generate Plot
print("Generating SHAP plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_to_plot, X_sample, plot_type="dot", show=False)
plt.title("SHAP Feature Impact (NLOS)")
plt.tight_layout()
plt.savefig('../results/figures/feature_shap.png')
plt.close()
print("Success! Plot saved as 'feature_shap.png'")

# 5. Save Gini Plot
print("Generating Gini plot...")
plt.figure(figsize=(10, 6))
importances = rf_model.feature_importances_
indices = np.argsort(importances)
plt.barh(range(len(indices)), importances[indices], color='skyblue')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.title("Feature Importance Ranking")
plt.savefig('../results/figures/feature_gini_importance.png')
plt.close()

print("\nDONE!")