"""
=============================================================================
Phase 1: Exploratory Data Analysis (EDA)
Credit Card Fraud Detection System
=============================================================================

This script performs a comprehensive EDA on the Kaggle Credit Card Fraud
Detection dataset. Run this FIRST before any preprocessing or training.

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Place 'creditcard.csv' in the data/ directory before running.

Usage:
    python notebooks/01_eda.py
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "creditcard.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "eda")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style settings for all plots
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
FIGSIZE = (14, 6)
DPI = 150


def save_fig(name: str):
    """Save current figure to output directory."""
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"  ✓ Saved: {path}")
    plt.close()


# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("=" * 70)
print("PHASE 1: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

if not os.path.exists(DATA_PATH):
    print(f"\n✗ Dataset not found at: {DATA_PATH}")
    print("  Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("  Place 'creditcard.csv' in the data/ directory.")
    sys.exit(1)

print(f"\n[1/10] Loading dataset from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
print(f"  ✓ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# =============================================================================
# 2. BASIC DATASET INFO
# =============================================================================
print(f"\n[2/10] Basic Dataset Information")
print("-" * 50)
print(f"  Shape:          {df.shape}")
print(f"  Memory Usage:   {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print(f"  Columns:        {list(df.columns)}")
print(f"\n  Data Types:")
print(f"  {df.dtypes.value_counts().to_dict()}")

print(f"\n  First 5 rows:")
print(df.head().to_string(index=False))

print(f"\n  Statistical Summary:")
print(df.describe().round(4).to_string())

# =============================================================================
# 3. MISSING VALUES ANALYSIS
# =============================================================================
print(f"\n[3/10] Missing Values Analysis")
print("-" * 50)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(4)
missing_report = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
missing_report = missing_report[missing_report["Missing Count"] > 0]

if missing_report.empty:
    print("  ✓ No missing values found in the dataset!")
else:
    print("  ✗ Missing values detected:")
    print(missing_report.to_string())
    print("\n  Strategy: Will impute using median values in preprocessing.")

# Also check for infinite values
inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
print(f"  Infinite values: {inf_count}")

# Check for duplicates
dup_count = df.duplicated().sum()
print(f"  Duplicate rows:  {dup_count} ({dup_count / len(df) * 100:.2f}%)")

# =============================================================================
# 4. CLASS DISTRIBUTION
# =============================================================================
print(f"\n[4/10] Class Distribution Analysis")
print("-" * 50)
class_counts = df["Class"].value_counts()
class_pct = df["Class"].value_counts(normalize=True) * 100

print(f"  Legitimate (0): {class_counts[0]:>8,}  ({class_pct[0]:.4f}%)")
print(f"  Fraud      (1): {class_counts[1]:>8,}  ({class_pct[1]:.4f}%)")
print(f"  Imbalance Ratio: 1:{class_counts[0] // class_counts[1]}")
print(f"\n  ⚠ SEVERE CLASS IMBALANCE — Fraud is only {class_pct[1]:.3f}% of data")
print(f"    This means naive accuracy would be {class_pct[0]:.2f}% just by predicting all 'Legitimate'")

# ── Plot: Class Distribution (Pie + Bar) ──
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

# Pie chart
colors = ["#2ecc71", "#e74c3c"]
explode = (0, 0.1)
axes[0].pie(class_counts, explode=explode, labels=["Legitimate", "Fraud"],
            colors=colors, autopct="%1.3f%%", shadow=True, startangle=90,
            textprops={"fontsize": 12, "fontweight": "bold"})
axes[0].set_title("Class Distribution (Pie)", fontsize=14, fontweight="bold", pad=15)

# Bar chart
bars = axes[1].bar(["Legitimate\n(Class 0)", "Fraud\n(Class 1)"],
                   class_counts.values, color=colors, edgecolor="black", linewidth=0.5)
for bar, count in zip(bars, class_counts.values):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                 f"{count:,}", ha="center", va="bottom", fontweight="bold", fontsize=12)
axes[1].set_title("Class Distribution (Bar)", fontsize=14, fontweight="bold")
axes[1].set_ylabel("Count", fontsize=12)

plt.suptitle("Credit Card Fraud — Class Distribution", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig("01_class_distribution")

# =============================================================================
# 5. TRANSACTION AMOUNT ANALYSIS
# =============================================================================
print(f"\n[5/10] Transaction Amount Analysis")
print("-" * 50)

fraud = df[df["Class"] == 1]
legit = df[df["Class"] == 0]

print(f"  {'Metric':<25} {'Legitimate':>15} {'Fraud':>15}")
print(f"  {'-' * 55}")
print(f"  {'Count':<25} {len(legit):>15,} {len(fraud):>15,}")
print(f"  {'Mean Amount':<25} ${legit['Amount'].mean():>14.2f} ${fraud['Amount'].mean():>14.2f}")
print(f"  {'Median Amount':<25} ${legit['Amount'].median():>14.2f} ${fraud['Amount'].median():>14.2f}")
print(f"  {'Std Amount':<25} ${legit['Amount'].std():>14.2f} ${fraud['Amount'].std():>14.2f}")
print(f"  {'Min Amount':<25} ${legit['Amount'].min():>14.2f} ${fraud['Amount'].min():>14.2f}")
print(f"  {'Max Amount':<25} ${legit['Amount'].max():>14.2f} ${fraud['Amount'].max():>14.2f}")
print(f"  {'25th Percentile':<25} ${legit['Amount'].quantile(0.25):>14.2f} ${fraud['Amount'].quantile(0.25):>14.2f}")
print(f"  {'75th Percentile':<25} ${legit['Amount'].quantile(0.75):>14.2f} ${fraud['Amount'].quantile(0.75):>14.2f}")

# ── Plot: Amount Distribution ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histogram — Legitimate
axes[0].hist(legit["Amount"], bins=80, color="#2ecc71", alpha=0.8, edgecolor="black", linewidth=0.3)
axes[0].set_title("Legitimate Transaction Amounts", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Amount ($)")
axes[0].set_ylabel("Frequency")
axes[0].set_xlim(0, legit["Amount"].quantile(0.99))

# Histogram — Fraud
axes[1].hist(fraud["Amount"], bins=50, color="#e74c3c", alpha=0.8, edgecolor="black", linewidth=0.3)
axes[1].set_title("Fraudulent Transaction Amounts", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Amount ($)")
axes[1].set_ylabel("Frequency")

# Box plot comparison
data_to_plot = [legit["Amount"], fraud["Amount"]]
bp = axes[2].boxplot(data_to_plot, labels=["Legitimate", "Fraud"], patch_artist=True,
                     medianprops={"color": "black", "linewidth": 2})
bp["boxes"][0].set_facecolor("#2ecc71")
bp["boxes"][1].set_facecolor("#e74c3c")
axes[2].set_title("Amount Distribution Comparison", fontsize=12, fontweight="bold")
axes[2].set_ylabel("Amount ($)")

plt.suptitle("Transaction Amount Analysis", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig("02_amount_distribution")

# =============================================================================
# 6. TIME DISTRIBUTION ANALYSIS
# =============================================================================
print(f"\n[6/10] Time Distribution Analysis")
print("-" * 50)
print(f"  Time range: {df['Time'].min():.0f}s to {df['Time'].max():.0f}s")
print(f"  Duration:   {df['Time'].max() / 3600:.1f} hours (~{df['Time'].max() / 86400:.1f} days)")

# ── Plot: Time Distribution ──
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

# Overall time distribution
axes[0].hist(legit["Time"] / 3600, bins=48, color="#2ecc71", alpha=0.6, label="Legitimate", edgecolor="black", linewidth=0.2)
axes[0].hist(fraud["Time"] / 3600, bins=48, color="#e74c3c", alpha=0.8, label="Fraud", edgecolor="black", linewidth=0.2)
axes[0].set_title("Transaction Time Distribution", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Time (hours)")
axes[0].set_ylabel("Frequency")
axes[0].legend()

# Fraud time distribution zoomed
axes[1].hist(fraud["Time"] / 3600, bins=48, color="#e74c3c", alpha=0.8, edgecolor="black", linewidth=0.3)
axes[1].set_title("Fraud Transactions Over Time", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Time (hours)")
axes[1].set_ylabel("Frequency")

plt.suptitle("Time Analysis", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig("03_time_distribution")

# =============================================================================
# 7. CORRELATION ANALYSIS
# =============================================================================
print(f"\n[7/10] Feature Correlation Analysis")
print("-" * 50)

# Correlation with target (Class)
corr_with_target = df.corr()["Class"].drop("Class").sort_values()
print(f"\n  Top 5 NEGATIVELY correlated features with Fraud:")
for feat, corr in corr_with_target.head(5).items():
    print(f"    {feat:<8} r = {corr:+.4f}")

print(f"\n  Top 5 POSITIVELY correlated features with Fraud:")
for feat, corr in corr_with_target.tail(5).items():
    print(f"    {feat:<8} r = {corr:+.4f}")

# ── Plot: Correlation with Target ──
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Bar chart of correlations
colors_corr = ["#e74c3c" if c < 0 else "#2ecc71" for c in corr_with_target.values]
axes[0].barh(corr_with_target.index, corr_with_target.values, color=colors_corr, edgecolor="black", linewidth=0.3)
axes[0].set_title("Feature Correlation with Fraud (Class)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Pearson Correlation Coefficient")
axes[0].axvline(x=0, color="black", linewidth=0.5)

# Heatmap of top correlated features
top_features = list(corr_with_target.head(7).index) + list(corr_with_target.tail(7).index) + ["Class"]
corr_matrix = df[top_features].corr()
im = axes[1].imshow(corr_matrix, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
axes[1].set_xticks(range(len(top_features)))
axes[1].set_yticks(range(len(top_features)))
axes[1].set_xticklabels(top_features, rotation=45, ha="right", fontsize=8)
axes[1].set_yticklabels(top_features, fontsize=8)
axes[1].set_title("Correlation Heatmap (Top Features)", fontsize=12, fontweight="bold")
plt.colorbar(im, ax=axes[1], shrink=0.8)

plt.suptitle("Feature Correlation Analysis", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig("04_correlation_analysis")

# =============================================================================
# 8. KEY FEATURE DISTRIBUTIONS (V14, V17, V12, V10)
# =============================================================================
print(f"\n[8/10] Key Feature Distribution Analysis")
print("-" * 50)

key_features = ["V14", "V17", "V12", "V10", "V4", "V11", "V3", "V16"]
print(f"  Analyzing features with highest fraud correlation: {key_features}")

# ── Plot: Distribution of Key Features ──
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, feat in enumerate(key_features):
    ax = axes[i]
    ax.hist(legit[feat], bins=50, alpha=0.6, color="#2ecc71", label="Legit", density=True, edgecolor="black", linewidth=0.2)
    ax.hist(fraud[feat], bins=50, alpha=0.7, color="#e74c3c", label="Fraud", density=True, edgecolor="black", linewidth=0.2)
    ax.set_title(f"{feat} Distribution", fontsize=11, fontweight="bold")
    ax.set_xlabel(feat)
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

plt.suptitle("Key Feature Distributions — Fraud vs Legitimate",
             fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig("05_key_feature_distributions")

# ── Plot: Box Plots for Key Features ──
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, feat in enumerate(key_features):
    ax = axes[i]
    data = [legit[feat], fraud[feat]]
    bp = ax.boxplot(data, labels=["Legit", "Fraud"], patch_artist=True,
                    medianprops={"color": "black", "linewidth": 2})
    bp["boxes"][0].set_facecolor("#2ecc71")
    bp["boxes"][1].set_facecolor("#e74c3c")
    ax.set_title(f"{feat} — Box Plot", fontsize=11, fontweight="bold")
    ax.set_ylabel(feat)

plt.suptitle("Key Feature Box Plots — Fraud vs Legitimate",
             fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig("06_key_feature_boxplots")

# =============================================================================
# 9. STATISTICAL TESTS
# =============================================================================
print(f"\n[9/10] Statistical Significance Tests (Mann-Whitney U)")
print("-" * 50)
print(f"  {'Feature':<10} {'U Statistic':>15} {'p-value':>15} {'Significant?':>14}")
print(f"  {'-' * 55}")

all_v_features = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
significant_features = []

for feat in all_v_features:
    u_stat, p_val = stats.mannwhitneyu(legit[feat], fraud[feat], alternative="two-sided")
    sig = "YES" if p_val < 0.05 else "NO"
    if p_val < 0.05:
        significant_features.append(feat)
    if feat in key_features or feat in ["Time", "Amount"]:
        print(f"  {feat:<10} {u_stat:>15.0f} {p_val:>15.2e} {sig:>14}")

print(f"\n  Total significant features (p < 0.05): {len(significant_features)}/{len(all_v_features)}")
print(f"  Features: {', '.join(significant_features)}")

# =============================================================================
# 10. SUMMARY & INSIGHTS
# =============================================================================
print(f"\n[10/10] EDA Summary & Key Insights")
print("=" * 70)

insights = f"""
  DATASET OVERVIEW
  ─────────────────
  • {df.shape[0]:,} transactions over ~{df['Time'].max() / 86400:.0f} days
  • {df.shape[1]} features: Time, Amount, V1-V28 (PCA-transformed), Class
  • Missing values: {'None' if missing.sum() == 0 else f'{missing.sum()} found'}
  • Duplicate rows: {dup_count}

  CLASS IMBALANCE
  ─────────────────
  • Legitimate: {class_counts[0]:,} ({class_pct[0]:.3f}%)
  • Fraud:      {class_counts[1]:,} ({class_pct[1]:.3f}%)
  • Ratio:      1:{class_counts[0] // class_counts[1]} — SEVERE IMBALANCE
  • Strategy:   Must use SMOTE / Class Weights / Undersampling

  TRANSACTION AMOUNTS
  ─────────────────
  • Fraud transactions tend to be SMALLER (mean ${fraud['Amount'].mean():.2f} vs ${legit['Amount'].mean():.2f})
  • Many fraud transactions are under $20

  KEY DISCRIMINATIVE FEATURES
  ─────────────────
  • V14 (r={corr_with_target['V14']:+.4f}): STRONGEST negative correlation with fraud
  • V17 (r={corr_with_target['V17']:+.4f}): Second strongest negative correlation
  • V12 (r={corr_with_target['V12']:+.4f}): Third strongest
  • V11 (r={corr_with_target['V11']:+.4f}): Strong positive correlation
  • V4  (r={corr_with_target['V4']:+.4f}): Strong positive correlation

  NEXT STEPS
  ─────────────────
  → Phase 2: Scale Time + Amount, handle class imbalance
  → Phase 3: Build and train ANN model
"""

print(insights)

# Save summary to file
with open(os.path.join(OUTPUT_DIR, "eda_summary.txt"), "w") as f:
    f.write(insights)
print(f"  ✓ Summary saved to {OUTPUT_DIR}/eda_summary.txt")

print(f"\n  ✓ All plots saved to: {OUTPUT_DIR}/")
print(f"  ✓ EDA COMPLETE — Proceed to Phase 2 (Preprocessing)")
print("=" * 70)
