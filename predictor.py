import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging

logger = logging.getLogger(__name__)

MODEL_PATH  = os.getenv("MODEL_PATH",  "models/best_model.keras")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")

_model  = None
_scaler = None


def load_model():
    global _model, _scaler
    try:
        import tensorflow as tf
        _model = tf.keras.models.load_model(MODEL_PATH)
        logger.info(f"ANN model loaded from {MODEL_PATH}")
    except Exception as e:
        logger.warning(f"Could not load TF model ({e}). Using mock predictor.")
        _model = None

    try:
        _scaler = joblib.load(SCALER_PATH)
        logger.info("Scaler loaded.")
    except Exception:
        logger.warning("Scaler not found — fitting a default scaler.")
        _scaler = StandardScaler()


def _mock_predict(features: np.ndarray) -> float:
    """Deterministic mock when no model file is present (dev mode)."""
    v14 = features[0][15]   # index 15 = V14 (after Time, Amount, V1-V14)
    v17 = features[0][18]   # V17
    amount = features[0][1]
    score = 0.05
    if v14 < -5:   score += 0.4
    if v17 < -5:   score += 0.3
    if amount < 5: score += 0.15
    return min(float(score), 0.99)


def predict_fraud(feature_dict: dict) -> dict:
    """
    feature_dict keys: time, amount, v1..v28
    Returns: probability, is_fraud, risk_level
    """
    if _model is None and _scaler is None:
        load_model()

    ordered = [
        feature_dict.get("time", 0.0),
        feature_dict.get("amount", 0.0),
        *[feature_dict.get(f"v{i}", 0.0) for i in range(1, 29)],
    ]
    features = np.array(ordered).reshape(1, -1)

    # Scale Time and Amount (indices 0 and 1)
    if _scaler is not None and hasattr(_scaler, "mean_") and len(getattr(_scaler, "mean_", [])) == 2:
        features[:, :2] = _scaler.transform(features[:, :2])
    elif _scaler is not None and hasattr(_scaler, "mean_") and len(getattr(_scaler, "mean_", [])) == 30:
        features = _scaler.transform(features)

    if _model is not None:
        prob = float(_model.predict(features, verbose=0)[0][0])
    else:
        prob = _mock_predict(features)

    if prob >= 0.80:
        risk = "HIGH"
    elif prob >= 0.40:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "fraud_probability": round(prob, 4),
        "is_fraud": prob >= 0.50,
        "risk_level": risk,
        "alert_triggered": prob >= 0.80,
    }
