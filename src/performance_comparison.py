import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Use a non-interactive backend to prevent macOS hangs
import matplotlib
matplotlib.use('Agg')

# 1. Load your enhanced dataset
df = pd.read_csv('../data/processed/enhanced_features.csv').dropna()

# 2. Define the two feature sets
# Basic: Only the original scalar values
basic_cols = ['RANGE', 'FP_AMP1', 'STDEV_NOISE']
# Enhanced: Everything (Basic + your engineered CIR features)
enhanced_cols = list(df.drop('NLOS', axis=1).columns)

X = df.drop('NLOS', axis=1)
y = df['NLOS']

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Train & Evaluate Basic Model
print("Training Basic Model...")
rf_basic = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_basic.fit(X_train[basic_cols], y_train)
acc_basic = accuracy_score(y_test, rf_basic.predict(X_test[basic_cols]))

# 5. Train & Evaluate Enhanced Model
print("Training Enhanced Model...")
rf_enhanced = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_enhanced.fit(X_train, y_train)
acc_enhanced = accuracy_score(y_test, rf_enhanced.predict(X_test))

# 6. Generate Comparison Diagram
plt.figure(figsize=(8, 6))
labels = ['Basic Features\n(Original Scalars)', 'Enhanced Features\n(With CIR Engineering)']
accuracies = [acc_basic, acc_enhanced]

bars = plt.bar(labels, accuracies, color=['#d3d3d3', '#5bc0de'])
plt.ylim(0.5, 1.0) # Zoom in to show the difference clearly
plt.ylabel('Accuracy Score')
plt.title('Comparison: Basic vs. Enhanced UWB Features')

# Add text labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.2%}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('../results/figures/performance_comparison.png')
print(f"\nRESULTS:\nBasic Accuracy: {acc_basic:.2%}")
print(f"Enhanced Accuracy: {acc_enhanced:.2%}")
print(f"Improvement: {acc_enhanced - acc_basic:.2%}")
print("\nDiagram saved as 'performance_comparison.png'")