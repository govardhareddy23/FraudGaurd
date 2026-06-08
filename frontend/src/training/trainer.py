"""
=============================================================================
ANN Training Runner & Trainer Utilities
Credit Card Fraud Detection System
=============================================================================

Handles training execution, early stopping, checkpointing, and callbacks.
=============================================================================
"""

import os
import json
import logging
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

logger = logging.getLogger(__name__)


def train_model(
    model: tf.keras.Model,
    X_train, y_train,
    X_val, y_val,
    epochs: int = 50,
    batch_size: int = 2048,
    class_weight: dict = None,
    checkpoint_filepath: str = "models/best_model.keras",
    history_filepath: str = "models/training_history.json"
):
    """
    Executes deep ANN training with production-quality callbacks.

    Parameters:
        model: Keras model compiled
        X_train, y_train: Training sets
        X_val, y_val: Validation sets
        epochs: Max epochs to train
        batch_size: Number of samples per gradient update
        class_weight: Weights for class imbalance handling
        checkpoint_filepath: Path to save the best model weights
        history_filepath: Path to save training log JSON

    Returns:
        history object
    """
    # Create directory for checkpoints if not exists
    os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)

    # 1. Early Stopping (Avoids overfitting, stops when validation loss stops improving)
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
        mode="min"
    )

    # 2. Model Checkpoint (Saves best model weights automatically based on validation recall or loss)
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
        mode="min"
    )

    # 3. Learning Rate Scheduler (Reduces LR when training plateaus to converge precisely)
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1,
        mode="min"
    )

    callbacks = [early_stopping, checkpoint, reduce_lr]

    logger.info(f"Starting model training (Batch Size: {batch_size}, Max Epochs: {epochs})...")

    # Fit Model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    # Save history
    history_dict = {k: [float(val) for val in v] for k, v in history.history.items()}
    with open(history_filepath, "w") as f:
        json.dump(history_dict, f, indent=4)

    logger.info(f"✓ Best model saved to {checkpoint_filepath}")
    logger.info(f"✓ Training history saved to {history_filepath}")

    return history
