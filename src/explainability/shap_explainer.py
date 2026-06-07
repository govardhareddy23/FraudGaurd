"""
=============================================================================
Explainable AI (SHAP Explainer)
Credit Card Fraud Detection System
=============================================================================

Implements SHAP to explain ANN predictions, identifying features that
influence high-risk decisions.
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import logging

logger = logging.getLogger(__name__)


class FraudExplainer:
    """
    SHAP Explainer module for the Credit Card Fraud ANN.
    """

    def __init__(self, model, background_data, feature_names=None):
        """
        Parameters:
            model: Trained Keras/TensorFlow model object
            background_data: A representative sample of the training features
                             (used as reference for SHAP baseline expectations)
            feature_names: List of column names (e.g. Time, Amount, V1-V28)
        """
        self.model = model
        # Limit background data to 100 samples for computational efficiency
        self.background_data = background_data[:100]
        self.feature_names = feature_names if feature_names else [f"V{i}" for i in range(1, 29)]
        
        # Initialize Explainer (KernelExplainer is model-agnostic, works reliably with Keras models)
        logger.info("Initializing SHAP KernelExplainer...")
        self.explainer = shap.KernelExplainer(self.predict_wrapper, self.background_data)

    def predict_wrapper(self, x_input):
        """
        Wrapper to ensure data types match between SHAP and Keras.
        """
        return self.model.predict(x_input, verbose=0)

    def explain_instance(self, instance) -> dict:
        """
        Explains a single transaction's prediction.

        Parameters:
            instance: 1D numpy array of shape (30,) or 2D of shape (1, 30)

        Returns:
            dict containing base value, prediction value, and feature contributions (shap values)
        """
        inst = instance.reshape(1, -1)
        shap_values = self.explainer.shap_values(inst, silent=True)
        
        # Extract results
        if isinstance(shap_values, list):
            # For multi-output or list structures
            shap_arr = shap_values[0][0]
        else:
            shap_arr = shap_values[0]

        # Map features to their SHAP values
        feature_importance = {}
        for name, val in zip(self.feature_names, shap_arr):
            feature_importance[name] = float(val)

        # Sort by absolute contribution
        sorted_importance = dict(sorted(feature_importance.items(), key=lambda item: abs(item[1]), reverse=True))

        return {
            "base_value": float(self.explainer.expected_value[0] if isinstance(self.explainer.expected_value, np.ndarray) else self.explainer.expected_value),
            "prediction": float(self.model.predict(inst, verbose=0)[0][0]),
            "contributions": sorted_importance
        }

    def save_force_plot(self, instance, save_path: str):
        """
        Creates and saves a SHAP force plot for an individual transaction.
        """
        inst = instance.reshape(1, -1)
        shap_values = self.explainer.shap_values(inst, silent=True)
        
        expected_val = self.explainer.expected_value
        if isinstance(expected_val, np.ndarray):
            expected_val = expected_val[0]

        plt.figure(figsize=(10, 3))
        shap.force_plot(
            expected_val,
            shap_values[0] if isinstance(shap_values, list) else shap_values,
            inst,
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        plt.title("SHAP Force Plot — Feature Contributions", fontsize=12, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"✓ SHAP force plot saved to {save_path}")

    def save_summary_plot(self, X_sample, save_path: str):
        """
        Creates and saves a SHAP summary plot for a batch of transactions.
        """
        logger.info("Computing SHAP values for summary plot (batch explanations)...")
        shap_values = self.explainer.shap_values(X_sample[:30], silent=True)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values[0] if isinstance(shap_values, list) else shap_values,
            X_sample[:30],
            feature_names=self.feature_names,
            show=False
        )
        plt.title("SHAP Feature Importance Summary", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"✓ SHAP summary plot saved to {save_path}")
