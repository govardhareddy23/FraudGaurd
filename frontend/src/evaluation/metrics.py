"""
=============================================================================
Model Evaluation Metrics & Plotting Tools
Credit Card Fraud Detection System
=============================================================================

Calculates performance metrics (Accuracy, Precision, Recall, F1, AUC) and
saves curves (ROC, Precision-Recall, Confusion Matrix, Training History).
=============================================================================
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)


def calculate_metrics(y_true, y_pred_prob, threshold=0.5) -> dict:
    """
    Computes all standard binary classification metrics.
    """
    y_pred = (y_pred_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_pred_prob)),
        "pr_auc": float(average_precision_score(y_true, y_pred_prob))
    }
    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path: str):
    """
    Plots interactive confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"]
    )
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.ylabel("True Class", fontsize=12)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_true, y_pred_prob, save_path: str):
    """
    Plots the Receiver Operating Characteristic (ROC) curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    auc_val = roc_auc_score(y_true, y_pred_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2980b9", lw=2, label=f"ANN (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], color="#7f8c8d", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)")
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_precision_recall_curve(y_true, y_pred_prob, save_path: str):
    """
    Plots Precision-Recall curve (ideal for imbalanced data).
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    ap = average_precision_score(y_true, y_pred_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color="#27ae60", lw=2, label=f"ANN (PR-AUC / AP = {ap:.4f})")
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision (Positive Predictive Value)")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_history(history_dict: dict, output_dir: str):
    """
    Plots training vs validation loss and accuracy curves.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Loss History
    if "loss" in history_dict:
        plt.figure(figsize=(8, 5))
        plt.plot(history_dict["loss"], label="Train Loss", color="#c0392b", lw=2)
        if "val_loss" in history_dict:
            plt.plot(history_dict["val_loss"], label="Val Loss", color="#e67e22", lw=2, linestyle="--")
        plt.title("Model Loss History", fontsize=14, fontweight="bold")
        plt.xlabel("Epochs")
        plt.ylabel("Loss (Binary Cross Entropy)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_loss.png"), dpi=150)
        plt.close()

    # 2. Accuracy History
    if "accuracy" in history_dict:
        plt.figure(figsize=(8, 5))
        plt.plot(history_dict["accuracy"], label="Train Accuracy", color="#27ae60", lw=2)
        if "val_accuracy" in history_dict:
            plt.plot(history_dict["val_accuracy"], label="Val Accuracy", color="#2ecc71", lw=2, linestyle="--")
        plt.title("Model Accuracy History", fontsize=14, fontweight="bold")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_accuracy.png"), dpi=150)
        plt.close()
