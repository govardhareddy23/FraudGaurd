"""
=============================================================================
Phase 3: ANN Training Pipeline
Credit Card Fraud Detection System
=============================================================================

This script loads raw data, splits it, runs preprocessing, trains
multiple ANNs to compare strategies (Class Weights, SMOTE, Undersampling),
selects the best model, and saves it.

Usage:
    python notebooks/03_ann_training.py
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
from src.models.ann_model import build_ann_model
from src.training.trainer import train_model
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "creditcard.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Main Training Runner ───────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("PHASE 3: ANN DEVELOPMENT AND MODEL TRAINING")
    print("=" * 70)

    if not os.path.exists(DATA_PATH):
        print(f"\n✗ Dataset not found at: {DATA_PATH}")
        print("  Please download and place creditcard.csv in data/ directory first.")
        print("  Training aborted. Creating a dummy model file for development...")
        # Save a basic model for mock pipeline testing if the user wants to start the API
        model = build_ann_model(input_dim=30)
        model.save(os.path.join(MODELS_DIR, "best_model.keras"))

        # Save dummy scaler
        from sklearn.preprocessing import StandardScaler
        import joblib
        scaler = StandardScaler()
        scaler.fit(np.random.randn(10, 30))
        joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
        print("✓ Dummy model and scaler written to models/ so API can start.")
        sys.exit(0)

    print("\n[1/5] Loading data and preparing splits...")
    df = pd.read_csv(DATA_PATH)
    preprocessor = FraudPreprocessor(scale_columns=["Time", "Amount"], test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)

    # Validation split for training callback checks
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    # Save scaler for API usage
    preprocessor.save_scaler(os.path.join(MODELS_DIR, "scaler.pkl"))

    # ── Strategy 1: Class Weights ──
    print("\n[2/5] Strategy 1: Training ANN using Class Weights...")
    class_weights = preprocessor.compute_class_weights(y_train_split)

    model_weights = build_ann_model(input_dim=30, num_layers=3, initial_neurons=128, dropout_rate=0.3, learning_rate=1e-3)
    train_model(
        model_weights,
        X_train_split, y_train_split,
        X_val, y_val,
        epochs=30,
        batch_size=2048,
        class_weight=class_weights,
        checkpoint_filepath=os.path.join(MODELS_DIR, "ann_class_weights.keras"),
        history_filepath=os.path.join(MODELS_DIR, "history_class_weights.json")
    )

    # ── Strategy 2: SMOTE ──
    print("\n[3/5] Strategy 2: Resampling with SMOTE and training ANN...")
    X_smote, y_smote = preprocessor.apply_smote(X_train_split, y_train_split, sampling_strategy=0.1)

    # Shuffle SMOTE resampled data
    idx = np.arange(len(X_smote))
    np.random.shuffle(idx)
    X_smote, y_smote = X_smote[idx], y_smote[idx]

    model_smote = build_ann_model(input_dim=30, num_layers=3, initial_neurons=128, dropout_rate=0.3, learning_rate=1e-3)
    train_model(
        model_smote,
        X_smote, y_smote,
        X_val, y_val,
        epochs=30,
        batch_size=2048,
        checkpoint_filepath=os.path.join(MODELS_DIR, "ann_smote.keras"),
        history_filepath=os.path.join(MODELS_DIR, "history_smote.json")
    )

    # ── Strategy 3: Undersampling ──
    print("\n[4/5] Strategy 3: Undersampling and training ANN...")
    X_under, y_under = preprocessor.apply_undersampling(X_train_split, y_train_split, sampling_strategy=0.5)

    model_under = build_ann_model(input_dim=30, num_layers=3, initial_neurons=128, dropout_rate=0.3, learning_rate=1e-3)
    train_model(
        model_under,
        X_under, y_under,
        X_val, y_val,
        epochs=30,
        batch_size=512,  # Smaller dataset, smaller batch size
        checkpoint_filepath=os.path.join(MODELS_DIR, "ann_undersampling.keras"),
        history_filepath=os.path.join(MODELS_DIR, "history_undersampling.json")
    )

    # ── Compare Validation Performance ──
    print("\n[5/5] Comparing all strategies and selecting best model...")
    results = {}
    for name, path in [
        ("Class Weights", os.path.join(MODELS_DIR, "ann_class_weights.keras")),
        ("SMOTE", os.path.join(MODELS_DIR, "ann_smote.keras")),
        ("Undersampling", os.path.join(MODELS_DIR, "ann_undersampling.keras"))
    ]:
        if os.path.exists(path):
            m = tf.keras.models.load_model(path)
            preds = (m.predict(X_test, verbose=0) >= 0.5).astype(int).flatten()

            from sklearn.metrics import recall_score, precision_score, f1_score
            rec = recall_score(y_test, preds)
            prec = precision_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            print(f"  {name:<15} | Recall: {rec:.4f} | Precision: {prec:.4f} | F1: {f1:.4f}")
            results[name] = {"recall": rec, "f1": f1, "path": path}

    # Select best model based on F1 score / Recall compromise (Prioritize Recall)
    best_strategy = max(results, key=lambda k: results[k]["recall"] + results[k]["f1"])
    print(f"\n✓ Selected strategy: {best_strategy} as the best model.")

    # Save as default production model
    import shutil
    shutil.copy(results[best_strategy]["path"], os.path.join(MODELS_DIR, "best_model.keras"))
    print(f"✓ Copied best model to models/best_model.keras for API inference.")
    print("=" * 70)


if __name__ == "__main__":
    main()
