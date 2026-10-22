"""
Model evaluation utilities.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
)
import seaborn as sns

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"


def evaluation_report(
    y_true         : np.ndarray,
    y_pred         : np.ndarray,
    anomaly_scores : np.ndarray,
) -> dict:
    """
    Generate a full evaluation report.

    Args:
        y_true:         binary ground truth
        y_pred:         binary predictions
        anomaly_scores: continuous anomaly scores

    Returns:
        dict with all metrics
    """
    roc_auc = roc_auc_score(y_true, anomaly_scores)

    report = classification_report(
        y_true, y_pred,
        target_names=["normal", "anomaly"],
        output_dict=True,
        zero_division=0
    )

    print("=== Evaluation Report ===\n")
    print(classification_report(
        y_true, y_pred,
        target_names=["normal", "anomaly"],
        zero_division=0
    ))
    print(f"ROC-AUC: {roc_auc:.4f}")

    return {
        "roc_auc"          : round(roc_auc, 4),
        "precision_anomaly": round(report["anomaly"]["precision"], 4),
        "recall_anomaly"   : round(report["anomaly"]["recall"], 4),
        "f1_anomaly"       : round(report["anomaly"]["f1-score"], 4),
        "accuracy"         : round(report["accuracy"], 4),
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save  : bool = True,
) -> None:
    """Plot and optionally save confusion matrix."""
    PLOTS_DIR.mkdir(exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["pred normal", "pred anomaly"],
        yticklabels=["true normal", "true anomaly"],
        ax=ax
    )
    ax.set_title("Confusion Matrix", fontsize=11)
    plt.tight_layout()

    if save:
        path = PLOTS_DIR / "confusion_matrix.png"
        plt.savefig(path, dpi=150)
        print(f"[evaluate] Saved → {path}")
    plt.show()


def plot_pr_curve(
    y_true        : np.ndarray,
    anomaly_scores: np.ndarray,
    save          : bool = True,
) -> None:
    """Plot precision-recall curve."""
    PLOTS_DIR.mkdir(exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_true, anomaly_scores)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(recall, precision, color="#378ADD", linewidth=2)
    ax.axhline(
        y=y_true.mean(), color="red",
        linestyle="--", linewidth=1,
        label=f"baseline ({y_true.mean():.3f})"
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        path = PLOTS_DIR / "pr_curve.png"
        plt.savefig(path, dpi=150)
        print(f"[evaluate] Saved → {path}")
    plt.show()