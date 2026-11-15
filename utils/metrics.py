"""Evaluation helpers: accuracy, precision, recall, F1, reports, confusion matrix."""
import matplotlib

matplotlib.use("Agg")  # no GUI needed, works on servers / lab PCs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import config


def compute_metrics(y_true, y_pred):
    """Return a dict with the standard metrics (weighted averages for P/R/F1)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "report": classification_report(
            y_true, y_pred, labels=config.LABELS, zero_division=0
        ),
        "confusion": confusion_matrix(y_true, y_pred, labels=config.LABELS),
    }


def save_confusion_matrix(matrix, path=config.CONFUSION_MATRIX_PATH):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="rocket",
        xticklabels=config.LABELS,
        yticklabels=config.LABELS,
    )
    plt.title("Tweet Sentiment Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_reports(metrics, timing, n_train, n_test):
    config.REPORTS_DIR.mkdir(exist_ok=True)
    lines = [
        "TWEET SENTIMENT ANALYSIS - ACCURACY REPORT",
        "=" * 44,
        f"Training tweets          : {n_train}",
        f"Test tweets              : {n_test}",
        f"Accuracy                 : {metrics['accuracy'] * 100:.2f} %",
        f"Precision (weighted)     : {metrics['precision'] * 100:.2f} %",
        f"Recall (weighted)        : {metrics['recall'] * 100:.2f} %",
        f"F1-Score (weighted)      : {metrics['f1'] * 100:.2f} %",
        f"Avg. processing time     : {timing['average_ms']:.2f} ms / tweet",
        f"First prediction (warm-up): {timing['first_ms']:.2f} ms",
    ]
    config.ACCURACY_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    config.CLASSIFICATION_REPORT_PATH.write_text(metrics["report"], encoding="utf-8")
    save_confusion_matrix(metrics["confusion"])
