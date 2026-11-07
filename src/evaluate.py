"""
Evaluation module for Intel Image Classification models.
Provides metric calculation (Accuracy, Precision, Recall, F1),
confusion matrix heatmap plotting, classification report generation,
and comprehensive multi-model benchmarking.
"""

import os
from typing import Dict, List, Any, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import tensorflow as tf

from src.config import OUTPUT_DIR, CLASS_NAMES


def plot_training_history(
    history: Union[tf.keras.callbacks.History, dict, pd.DataFrame],
    model_name: str,
    save_dir: str = OUTPUT_DIR,
) -> str:
    """Plots training and validation accuracy and loss curves over epochs.

    Args:
        history: Keras History object, dict, or pandas DataFrame.
        model_name: Name of the model.
        save_dir: Directory where the plot image will be saved.

    Returns:
        Path to the saved figure.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_history.png")

    if hasattr(history, "history"):
        h = history.history
    elif isinstance(history, pd.DataFrame):
        h = history.to_dict(orient="list")
    else:
        h = history

    epochs_range = range(1, len(h["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    axes[0].plot(epochs_range, h["accuracy"], label="Train Accuracy", color="#1f77b4", lw=2)
    if "val_accuracy" in h:
        axes[0].plot(epochs_range, h["val_accuracy"], label="Val Accuracy", color="#ff7f0e", lw=2, linestyle="--")
    axes[0].set_title(f"{model_name} - Accuracy", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epochs", fontsize=11)
    axes[0].set_ylabel("Accuracy", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="lower right")

    # Loss Plot
    axes[1].plot(epochs_range, h["loss"], label="Train Loss", color="#2ca02c", lw=2)
    if "val_loss" in h:
        axes[1].plot(epochs_range, h["val_loss"], label="Val Loss", color="#d62728", lw=2, linestyle="--")
    axes[1].set_title(f"{model_name} - Loss", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epochs", fontsize=11)
    axes[1].set_ylabel("Loss", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Evaluation] Saved training curve plot to {save_path}")
    return save_path


def evaluate_model(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: List[str] = CLASS_NAMES,
    model_name: str = "model",
    save_dir: str = OUTPUT_DIR,
    training_time: float = 0.0,
) -> Dict[str, Any]:
    """Evaluates a trained model on the test dataset and exports metrics.

    Calculates:
        - Accuracy
        - Precision (Macro & Weighted)
        - Recall (Macro & Weighted)
        - F1-Score (Macro & Weighted)
        - Classification Report (saved to TXT and CSV)
        - Confusion Matrix Heatmap (saved to PNG)

    Args:
        model: Trained tf.keras.Model.
        test_ds: Evaluation tf.data.Dataset.
        class_names: List of class names.
        model_name: Identifier for the model.
        save_dir: Directory to save evaluation artifacts.
        training_time: Recorded training duration in seconds.

    Returns:
        Dictionary containing all evaluation metrics.
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n[Evaluation] Evaluating '{model_name}' on test set...")

    # Extract true labels
    y_true_list = []
    for _, labels in test_ds:
        y_true_list.append(labels.numpy())
    y_true = np.concatenate(y_true_list, axis=0)

    # Predict probabilities and classes
    y_pred_probs = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Core Metrics
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Classification Report
    clf_report_text = classification_report(
        y_true, y_pred, target_names=class_names, digits=4
    )
    clf_report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, digits=4
    )

    report_txt_path = os.path.join(save_dir, f"{model_name}_classification_report.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Classification Report - {model_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(clf_report_text)

    report_df = pd.DataFrame(clf_report_dict).transpose()
    report_csv_path = os.path.join(save_dir, f"{model_name}_classification_report.csv")
    report_df.to_csv(report_csv_path)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title(f"Confusion Matrix - {model_name}\nAccuracy: {acc*100:.2f}%", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class", fontsize=11)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    results = {
        "model_name": model_name,
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
        "training_time_sec": training_time,
        "confusion_matrix_path": cm_path,
        "report_txt_path": report_txt_path,
        "y_true": y_true,
        "y_pred": y_pred,
    }

    print(
        f"[Evaluation] {model_name} => Accuracy: {acc*100:.2f}% | "
        f"F1-Macro: {f1_macro:.4f} | F1-Weighted: {f1_weighted:.4f}"
    )
    return results


def compare_models(
    results_list_or_dict: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]],
    save_dir: str = OUTPUT_DIR,
) -> pd.DataFrame:
    """Builds a comparison DataFrame and comparative bar plots for multiple models.

    Args:
        results_list_or_dict: List or dictionary of evaluation result dicts.
        save_dir: Directory where comparison table and plots are saved.

    Returns:
        Pandas DataFrame containing benchmark metrics.
    """
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "model_comparison.csv")
    plot_path = os.path.join(save_dir, "model_comparison.png")

    if isinstance(results_list_or_dict, dict):
        raw_items = list(results_list_or_dict.values())
    else:
        raw_items = results_list_or_dict

    rows = []
    for item in raw_items:
        rows.append({
            "Model": item.get("model_name", "Unknown"),
            "Accuracy": round(float(item.get("accuracy", 0.0)), 4),
            "Precision (Macro)": round(float(item.get("precision_macro", 0.0)), 4),
            "Recall (Macro)": round(float(item.get("recall_macro", 0.0)), 4),
            "F1-Score (Macro)": round(float(item.get("f1_macro", 0.0)), 4),
            "F1-Score (Weighted)": round(float(item.get("f1_weighted", 0.0)), 4),
            "Training Time (s)": round(float(item.get("training_time_sec", 0.0)), 2),
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\n[Comparison] Saved model comparison summary to {csv_path}:\n{df.to_string(index=False)}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    models = df["Model"]
    x = np.arange(len(models))
    width = 0.35

    # Chart 1: Accuracy & F1-Macro
    axes[0].bar(x - width/2, df["Accuracy"] * 100, width, label="Accuracy (%)", color="#2b5c8f")
    axes[0].bar(x + width/2, df["F1-Score (Macro)"] * 100, width, label="F1-Macro (%)", color="#e07a5f")
    axes[0].set_title("Classification Performance Comparison", fontsize=13, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15, ha="right")
    axes[0].set_ylabel("Score (%)", fontsize=11)
    axes[0].set_ylim(0, 105)
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)
    axes[0].legend()

    # Chart 2: Training Time
    axes[1].bar(models, df["Training Time (s)"], color="#81b29a", width=0.45)
    axes[1].set_title("Training Duration Comparison (Seconds)", fontsize=13, fontweight="bold")
    axes[1].set_xticklabels(models, rotation=15, ha="right")
    axes[1].set_ylabel("Seconds", fontsize=11)
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)
    for i, v in enumerate(df["Training Time (s)"]):
        axes[1].text(i, v + max(df["Training Time (s)"])*0.01, f"{v:.1f}s", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Comparison] Saved comparative bar chart to {plot_path}")
    return df
