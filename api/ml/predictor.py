import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
import os
import logging

logger = logging.getLogger(__name__)

# Use relative paths from file location or system env variables
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH  = os.getenv("MODEL_PATH",  os.path.join(BASE_DIR, "models", "best_model.keras"))
SCALER_PATH = os.getenv("SCALER_PATH", os.path.join(BASE_DIR, "models", "scaler.pkl"))

_model  = None
_scaler = None


def load_model():
    global _model, _scaler
    
    # 1. Load TensorFlow Keras Model
    try:
        import tensorflow as tf
        if os.path.exists(MODEL_PATH):
            _model = tf.keras.models.load_model(MODEL_PATH)
            logger.info(f"✓ ANN model loaded from {MODEL_PATH}")
        else:
            logger.warning(f"ANN Model file not found at {MODEL_PATH}. Using mock predictions.")
            _model = None
    except Exception as e:
        logger.warning(f"Could not load TF model ({e}). Using mock predictor.")
        _model = None

    # 2. Load RobustScaler
    try:
        if os.path.exists(SCALER_PATH):
            _scaler = joblib.load(SCALER_PATH)
            logger.info(f"✓ Scaler loaded from {SCALER_PATH}")
        else:
            logger.warning(f"Scaler not found at {SCALER_PATH} — fitting a default scaler.")
            _scaler = RobustScaler()
            # fit with dummy values
            _scaler.fit(np.random.randn(10, 2))
    except Exception as e:
        logger.warning(f"Scaler load error ({e}) — fitting a default scaler.")
        _scaler = RobustScaler()
        _scaler.fit(np.random.randn(10, 2))


def _mock_predict(features: np.ndarray) -> float:
    """Deterministic mock prediction when no model file is present (dev mode)."""
    # Grab v14 and v17 from the features vector
    # Ordered list is: time, amount, v1..v28.
    # Therefore: time=index 0, amount=index 1, v14=index 15, v17=index 18.
    v14 = features[0][15]
    v17 = features[0][18]
    amount = features[0][1]
    
    score = 0.05
    # If key features V14 or V17 are heavily negative (which correlates with fraud)
    if v14 < -5:
        score += 0.4
    if v17 < -5:
        score += 0.3
    if amount < 5:
        score += 0.15
        
    return min(float(score), 0.99)


def predict_fraud(feature_dict: dict) -> dict:
    """
    Given a raw transaction, scale it and predict fraud probability.
    
    feature_dict keys: time, amount, v1..v28
    Returns: dict with probability, is_fraud, risk_level, alert_triggered
    """
    global _model, _scaler
    if _model is None and _scaler is None:
        load_model()

    # Build the exact sequence of 30 features in order
    ordered = [
        feature_dict.get("time", 0.0),
        feature_dict.get("amount", 0.0),
        *[feature_dict.get(f"v{i}", 0.0) for i in range(1, 29)],
    ]
    features = np.array(ordered).reshape(1, -1)

    # Scale Time and Amount (indices 0 and 1)
    if _scaler is not None:
        try:
            # Handle if scaler was fit on entire 30 dimensions or just 2
            if hasattr(_scaler, "n_features_in_") and _scaler.n_features_in_ == 2:
                # Scale Time and Amount
                features[:, :2] = _scaler.transform(features[:, :2])
            elif hasattr(_scaler, "mean_") and len(getattr(_scaler, "mean_", [])) == 2:
                features[:, :2] = _scaler.transform(features[:, :2])
            else:
                # Fallback: scale all features if fit that way
                features = _scaler.transform(features)
        except Exception as e:
            logger.warning(f"Scaling error: {e}. Skipping scaling.")

    # Run Prediction
    if _model is not None:
        try:
            prob = float(_model.predict(features, verbose=0)[0][0])
        except Exception as e:
            logger.warning(f"Inference error ({e}). Using mock prediction.")
            prob = _mock_predict(features)
    else:
        prob = _mock_predict(features)

    # Risk level threshold rules
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


def get_model_and_scaler():
    """Exposes internal objects for explainers or advanced tools."""
    global _model, _scaler
    if _model is None and _scaler is None:
        load_model()
    return _model, _scaler
