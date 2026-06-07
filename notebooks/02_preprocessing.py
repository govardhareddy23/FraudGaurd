"""
=============================================================================
Phase 2: Data Preprocessing Demonstration
Credit Card Fraud Detection System
=============================================================================

This script runs the preprocessor pipeline on the raw creditcard.csv data
and verifies feature scaling, train/test split, and imbalance handling.

Usage:
    python notebooks/02_preprocessing.py
=============================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.pipeline import FraudPreprocessor

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "creditcard.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "preprocessing")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)

# ── Load Data ─────────────────────────────────────────────────────────────
print("=" * 70)
print("PHASE 2: DATA PREPROCESSING DEMONSTRATION")
print("=" * 70)

if not os.path.exists(DATA_PATH):
    print(f"\n✗ Dataset not found at: {DATA_PATH}")
    print("  Please download and place creditcard.csv in data/ directory first.")
    sys.exit(1)

print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"✓ Loaded {len(df):,} transactions.")

# Initialize preprocessor
preprocessor = FraudPreprocessor(scale_columns=["Time", "Amount"], test_size=0.2, random_state=42)

# Run Fit Transform
print("\nRunning fit_transform (scaling and splitting)...")
X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)

# Save the scaler for production use
preprocessor.save_scaler(os.path.join(PROJECT_ROOT, "models", "scaler.pkl"))
print("✓ RobustScaler saved to models/scaler.pkl")

# Verify scaling
print("\nVerifying Scaling:")
time_idx = preprocessor.get_feature_names().index("Time")
amount_idx = preprocessor.get_feature_names().index("Amount")

print(f"  Scaled Time - Mean: {X_train[:, time_idx].mean():.4f}, Std: {X_train[:, time_idx].std():.4f}")
print(f"  Scaled Amount - Mean: {X_train[:, amount_idx].mean():.4f}, Std: {X_train[:, amount_idx].std():.4f}")

# ── Test Resampling Strategies ─────────────────────────────────────────────
print("\nTesting Class Imbalance Strategies...")

# Compute Class Weights
class_weights = preprocessor.compute_class_weights(y_train)
print(f"✓ Class weights: {class_weights}")

# Apply SMOTE
X_smote, y_smote = preprocessor.apply_smote(X_train, y_train, sampling_strategy=0.1)

# Apply Undersampling
X_under, y_under = preprocessor.apply_undersampling(X_train, y_train, sampling_strategy=0.5)

# Visualizing distribution of labels after resampling
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = ["#2ecc71", "#e74c3c"]

# Original
axes[0].bar(["Legit", "Fraud"], [np.sum(y_train == 0), np.sum(y_train == 1)], color=colors, edgecolor="black")
axes[0].set_title(f"Original Training Set\n(Legit: {np.sum(y_train == 0):,}, Fraud: {np.sum(y_train == 1):,})", fontsize=12)
axes[0].set_ylabel("Count")

# SMOTE
axes[1].bar(["Legit", "Fraud"], [np.sum(y_smote == 0), np.sum(y_smote == 1)], color=colors, edgecolor="black")
axes[1].set_title(f"SMOTE Resampled Set (strategy=0.1)\n(Legit: {np.sum(y_smote == 0):,}, Fraud: {np.sum(y_smote == 1):,})", fontsize=12)

# Undersampled
axes[2].bar(["Legit", "Fraud"], [np.sum(y_under == 0), np.sum(y_under == 1)], color=colors, edgecolor="black")
axes[2].set_title(f"Random Undersampled Set (strategy=0.5)\n(Legit: {np.sum(y_under == 0):,}, Fraud: {np.sum(y_under == 1):,})", fontsize=12)

plt.suptitle("Label Distributions Across Resampling Strategies", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()

# Save the comparison plot
plot_path = os.path.join(OUTPUT_DIR, "resampling_comparison.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ Saved resampling comparison visualization to: {plot_path}")
print("\n✓ PREPROCESSING STAGE COMPLETED SUCCESSFULLY")
print("=" * 70)
