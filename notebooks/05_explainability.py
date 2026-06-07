"""
=============================================================================
Phase 5: Explainable AI Demonstration
Credit Card Fraud Detection System
=============================================================================

This script runs SHAP (SHapley Additive exPlanations) on the trained ANN
model to visualize feature contributions for safe and fraudulent predictions.

Usage:
    python notebooks/05_explainability.py
=============================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.pipeline import FraudPreprocessor
from src.explainability.shap_explainer import FraudExplainer

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "creditcard.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "explainability")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("PHASE 5: EXPLAINABLE AI (SHAP)")
    print("=" * 70)

    # 1. Check Model and Data
    ann_path = os.path.join(MODELS_DIR, "best_model.keras")
    
    if not os.path.exists(DATA_PATH) or not os.path.exists(ann_path):
        print("\n✗ Dataset or trained model missing. Creating mock explanation metadata...")
        mock_explain = {
            "prediction": 0.875,
            "base_value": 0.0017,
            "contributions": {
                "V14": -0.42,
                "V17": -0.28,
                "V4": 0.15,
                "Time": -0.05,
                "Amount": 0.03
            }
        }
        import json
        with open(os.path.join(OUTPUT_DIR, "mock_explanation.json"), "w") as f:
            json.dump(mock_explain, f, indent=4)
        print(f"✓ Saved mock explanation results to {OUTPUT_DIR}/mock_explanation.json")
        sys.exit(0)

    print("\nLoading and preprocessing dataset...")
    df = pd.read_csv(DATA_PATH)
    preprocessor = FraudPreprocessor(scale_columns=["Time", "Amount"], test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)

    print(f"Loading ANN model from {ann_path}...")
    model = tf.keras.models.load_model(ann_path)

    # 2. Initialize Explainer
    feature_names = preprocessor.get_feature_names()
    explainer = FraudExplainer(model, X_train, feature_names=feature_names)

    # 3. Find specific samples to explain
    print("\nSelecting transactions to explain...")
    # Find a fraud transaction in the test set
    fraud_indices = np.where(y_test == 1)[0]
    legit_indices = np.where(y_test == 0)[0]

    if len(fraud_indices) > 0 and len(legit_indices) > 0:
        fraud_sample = X_test[fraud_indices[0]]
        legit_sample = X_test[legit_indices[0]]
        
        # Explain Fraud instance
        print("\n[1/3] Explaining Fraudulent Transaction...")
        fraud_explanation = explainer.explain_instance(fraud_sample)
        print(f"  Prediction Prob: {fraud_explanation['prediction']:.4f}")
        print("  Top features boosting risk:")
        for k, v in list(fraud_explanation["contributions"].items())[:5]:
            print(f"    {k:<8}: {v:+.4f}")
            
        explainer.save_force_plot(
            fraud_sample,
            os.path.join(OUTPUT_DIR, "shap_force_plot_fraud.png")
        )

        # Explain Legitimate instance
        print("\n[2/3] Explaining Legitimate Transaction...")
        legit_explanation = explainer.explain_instance(legit_sample)
        print(f"  Prediction Prob: {legit_explanation['prediction']:.4f}")
        print("  Top features reducing risk:")
        for k, v in list(legit_explanation["contributions"].items())[:5]:
            print(f"    {k:<8}: {v:+.4f}")
            
        explainer.save_force_plot(
            legit_sample,
            os.path.join(OUTPUT_DIR, "shap_force_plot_legit.png")
        )

        # Explain a sample batch for a summary plot
        print("\n[3/3] Generating SHAP Summary Plot...")
        # Take a balanced mixture of 15 legit and 15 fraud samples for a clean summary plot
        mixed_indices = np.concatenate([fraud_indices[:15], legit_indices[:15]])
        X_mixed = X_test[mixed_indices]
        
        explainer.save_summary_plot(
            X_mixed,
            os.path.join(OUTPUT_DIR, "shap_summary_plot.png")
        )
    else:
        print("✗ Could not locate both fraudulent and legitimate transactions to explain.")

    print("\n✓ EXPLAINABILITY PIPELINE RUN SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
