"""
=============================================================================
Preprocessing Pipeline
Credit Card Fraud Detection System
=============================================================================

Handles:
  1. Feature scaling (StandardScaler for Time and Amount)
  2. Train/test split (stratified)
  3. Class imbalance handling:
     - Class Weights (computed automatically)
     - SMOTE (Synthetic Minority Oversampling)
     - Random Undersampling
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
import logging

logger = logging.getLogger(__name__)


class FraudPreprocessor:
    """
    Complete preprocessing pipeline for credit card fraud detection.

    Usage:
        preprocessor = FraudPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)

        # Get resampled data for different strategies
        X_smote, y_smote = preprocessor.apply_smote(X_train, y_train)
        X_under, y_under = preprocessor.apply_undersampling(X_train, y_train)
        class_weights = preprocessor.compute_class_weights(y_train)
    """

    def __init__(self, scale_columns=("Time", "Amount"), test_size=0.2, random_state=42):
        self.scale_columns = list(scale_columns)
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = RobustScaler()  # Robust to outliers — better for fraud data
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame, target_col: str = "Class"):
        """
        Complete preprocessing: scale features, split into train/test.

        Parameters:
            df: Raw DataFrame with all columns
            target_col: Name of the target column

        Returns:
            X_train, X_test, y_train, y_test (numpy arrays)
        """
        logger.info("Starting preprocessing pipeline...")

        # Step 1: Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col].values

        # Step 2: Scale Time and Amount
        logger.info(f"Scaling columns: {self.scale_columns}")
        X_scaled = X.copy()
        X_scaled[self.scale_columns] = self.scaler.fit_transform(X[self.scale_columns])
        self._fitted = True

        # Store feature names for later use
        self.feature_names = list(X_scaled.columns)

        # Step 3: Convert to numpy arrays
        X_np = X_scaled.values.astype(np.float32)
        y_np = y.astype(np.float32)

        # Step 4: Stratified train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_np, y_np,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_np
        )

        logger.info(f"Train set: {X_train.shape[0]:,} samples "
                     f"(Fraud: {int(y_train.sum()):,}, Legit: {int(len(y_train) - y_train.sum()):,})")
        logger.info(f"Test set:  {X_test.shape[0]:,} samples "
                     f"(Fraud: {int(y_test.sum()):,}, Legit: {int(len(y_test) - y_test.sum()):,})")

        return X_train, X_test, y_train, y_test

    def apply_smote(self, X_train, y_train, sampling_strategy=0.5):
        """
        Apply SMOTE (Synthetic Minority Oversampling Technique).

        Why SMOTE?
        - Generates synthetic fraud examples by interpolating between existing ones
        - Better than simple duplication (reduces overfitting risk)
        - sampling_strategy=0.5 means fraud will be 50% of legitimate count

        Parameters:
            X_train: Training features
            y_train: Training labels
            sampling_strategy: Ratio of minority to majority class

        Returns:
            X_resampled, y_resampled
        """
        logger.info(f"Applying SMOTE (strategy={sampling_strategy})...")
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state,
            k_neighbors=5
        )
        X_res, y_res = smote.fit_resample(X_train, y_train)

        logger.info(f"  Before SMOTE: Legit={int((y_train == 0).sum()):,}, Fraud={int((y_train == 1).sum()):,}")
        logger.info(f"  After SMOTE:  Legit={int((y_res == 0).sum()):,}, Fraud={int((y_res == 1).sum()):,}")

        return X_res, y_res

    def apply_undersampling(self, X_train, y_train, sampling_strategy=0.5):
        """
        Apply Random Undersampling to reduce the majority class.

        Why Undersampling?
        - Reduces the majority class to balance the dataset
        - Training is MUCH faster (fewer samples)
        - Risk: may lose important legitimate transaction patterns

        Parameters:
            X_train: Training features
            y_train: Training labels
            sampling_strategy: Ratio of minority to majority class

        Returns:
            X_resampled, y_resampled
        """
        logger.info(f"Applying Random Undersampling (strategy={sampling_strategy})...")
        rus = RandomUnderSampler(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state
        )
        X_res, y_res = rus.fit_resample(X_train, y_train)

        logger.info(f"  Before: Legit={int((y_train == 0).sum()):,}, Fraud={int((y_train == 1).sum()):,}")
        logger.info(f"  After:  Legit={int((y_res == 0).sum()):,}, Fraud={int((y_res == 1).sum()):,}")

        return X_res, y_res

    def compute_class_weights(self, y_train):
        """
        Compute balanced class weights for training.

        Why Class Weights?
        - Tells the model to penalize misclassifying fraud MORE heavily
        - No data modification needed — just adjusts the loss function
        - Best approach when you want to keep all data

        Returns:
            dict: {0: weight_legit, 1: weight_fraud}
        """
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes.astype(int), weights))

        logger.info(f"Class weights: {class_weight_dict}")
        logger.info(f"  Fraud class gets ~{class_weight_dict[1] / class_weight_dict[0]:.0f}x more weight")

        return class_weight_dict

    def save_scaler(self, path: str):
        """Save the fitted scaler for production use."""
        if not self._fitted:
            raise ValueError("Scaler not fitted yet. Call fit_transform first.")
        joblib.dump(self.scaler, path)
        logger.info(f"Scaler saved to {path}")

    def get_feature_names(self):
        """Return the list of feature names in order."""
        return self.feature_names
