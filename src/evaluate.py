"""
evaluate.py
===========
Reusable evaluation functions for HateLens.

WHY THIS FILE EXISTS:
Evaluation logic is used repeatedly across baseline, SMOTE, class-weights,
and threshold-tuned models. Centralizing it prevents inconsistency —
every model is evaluated identically, making comparison charts valid.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    auc,
)


def print_classification_report(y_true, y_pred, model_name: str = "Model"):
    """
    Print a formatted classification report with a clear header.

    WHY: We use classification_report rather than a single accuracy score
    because accuracy is misleading on imbalanced datasets. A model that
    predicts "not hate" for every tweet achieves ~86% accuracy while
    catching zero hate speech. F1, precision, and recall on the HATE class
    are the metrics that actually matter for Trust & Safety work.
    """
    print(f"\n{'='*60}")
    print(f"  Classification Report: {model_name}")
    print(f"{'='*60}")
    print(classification_report(
        y_true, y_pred, target_names=['Not Hate (0)', 'Hate (1)']
    ))


def plot_confusion_matrix(
    y_true,
    y_pred,
    title: str = "Confusion Matrix",
    save_path: str = None,
):
    """
    Plot and optionally save a seaborn confusion matrix heatmap.

    WHY WE USE A HEATMAP:
    Raw confusion matrix numbers are hard to interpret at a glance.
    A color-coded heatmap immediately reveals if the model is over-predicting
    one class (a common symptom of imbalance), making the problem visually
    undeniable to a non-technical audience.

    WHY THESE LABELS:
    - True Negative (top-left): correctly ignored non-hate → good
    - False Positive (top-right): wrongly flagged non-hate → adds queue volume
    - False Negative (bottom-left): missed real hate speech → causes direct harm
    - True Positive (bottom-right): correctly caught hate speech → good

    The False Negative cell is the most important to minimize in a
    safety-first context.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Pred: Not Hate', 'Pred: Hate'],
        yticklabels=['True: Not Hate', 'True: Hate'],
        ax=ax,
    )
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Actual Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)

    # Annotate cells with semantic meaning to aid non-technical readers.
    # WHY: A recruiter or policy analyst reading this chart should immediately
    # understand which cell represents missed hate speech (false negatives).
    labels = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            ax.text(
                j + 0.5, i + 0.75,
                labels[i][j],
                ha='center', va='center',
                fontsize=9, color='grey',
            )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved confusion matrix → {save_path}")
    plt.show()
    return fig


def plot_precision_recall_curve(
    y_true,
    y_proba,
    default_threshold: float = 0.50,
    save_path: str = None,
):
    """
    Plot the Precision-Recall curve for the hate speech class.

    WHY PRECISION-RECALL AND NOT ROC-AUC:
    ROC-AUC is optimistic on imbalanced datasets because it gives equal
    weight to both classes. On a 7:1 dataset, a model can achieve high
    ROC-AUC while having terrible recall on the minority class.
    The Precision-Recall curve focuses exclusively on the positive class
    (hate speech) — which is what we care about detecting.

    WHY WE MARK THE DEFAULT THRESHOLD:
    The default 0.5 threshold is arbitrary. Marking it shows visually
    that a lower threshold (higher recall, lower precision) is achievable
    and may be preferable in a safety context.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(recall, precision, color='steelblue', lw=2,
            label=f'PR Curve (AUC = {pr_auc:.3f})')
    ax.fill_between(recall, precision, alpha=0.1, color='steelblue')

    # Mark where the default 0.5 threshold falls on the curve.
    # WHY: This makes it concrete — you can see the precision/recall values
    # you're giving up (or gaining) by moving away from the default.
    if len(thresholds) > 0:
        default_idx = np.argmin(np.abs(thresholds - default_threshold))
        ax.scatter(
            recall[default_idx], precision[default_idx],
            color='red', s=80, zorder=5,
            label=f'Default threshold = {default_threshold}'
        )

    ax.set_xlabel('Recall (Hate Speech Class)', fontsize=12)
    ax.set_ylabel('Precision (Hate Speech Class)', fontsize=12)
    ax.set_title('Precision-Recall Curve — Hate Speech Class', fontsize=13,
                 fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim([0.0, 1.02])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved PR curve → {save_path}")
    plt.show()
    return fig, thresholds, precision, recall


def evaluate_thresholds(
    y_true,
    y_proba,
    thresholds_to_test=None,
) -> pd.DataFrame:
    """
    Evaluate hate speech class performance at multiple decision thresholds.

    WHY THRESHOLD TUNING:
    The default threshold of 0.50 says: "flag a tweet only if we're MORE
    than 50% confident it's hate speech." But why 50%? That's arbitrary.
    In a safety-first context with human review, we may prefer to flag at
    0.35 — accepting more false positives (extra queue volume) in exchange
    for catching more genuine hate speech (higher recall).

    This function makes that trade-off explicit and numeric.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_proba : array-like
        Predicted probabilities for the hate (positive) class.
    thresholds_to_test : list of float, optional
        Thresholds to evaluate. Defaults to [0.30, 0.35, 0.40, 0.45, 0.50].

    Returns
    -------
    pd.DataFrame
        One row per threshold with Precision, Recall, F1 for hate class.
    """
    if thresholds_to_test is None:
        thresholds_to_test = [0.30, 0.35, 0.40, 0.45, 0.50]

    rows = []
    for t in thresholds_to_test:
        y_pred_t = (np.array(y_proba) >= t).astype(int)
        rows.append({
            'Threshold': t,
            'Precision (Hate)': round(precision_score(y_true, y_pred_t, pos_label=1, zero_division=0), 4),
            'Recall (Hate)':    round(recall_score(y_true, y_pred_t, pos_label=1, zero_division=0), 4),
            'F1 (Hate)':        round(f1_score(y_true, y_pred_t, pos_label=1, zero_division=0), 4),
            'Accuracy':         round(accuracy_score(y_true, y_pred_t), 4),
        })

    results = pd.DataFrame(rows)

    # Mark the threshold that maximizes F1 on the hate class.
    # WHY F1 as the optimization target: F1 balances precision and recall.
    # It prevents us from picking a threshold so low that precision collapses
    # (everything gets flagged) or so high that recall collapses (nothing gets
    # flagged). F1 gives us the best single-number summary of hate class performance.
    best_idx = results['F1 (Hate)'].idxmax()
    results['Best'] = ''
    results.loc[best_idx, 'Best'] = '← BEST F1'

    return results


def build_model_comparison_table(results_dict: dict) -> pd.DataFrame:
    """
    Build a clean comparison table across all model variants.

    Parameters
    ----------
    results_dict : dict
        Keys are model names (str), values are dicts with keys:
        'accuracy', 'hate_f1', 'hate_precision', 'hate_recall'

    Returns
    -------
    pd.DataFrame
        Formatted comparison table suitable for display and plotting.
    """
    rows = []
    for model_name, metrics in results_dict.items():
        rows.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Hate F1': metrics['hate_f1'],
            'Hate Precision': metrics['hate_precision'],
            'Hate Recall': metrics['hate_recall'],
        })
    return pd.DataFrame(rows)


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    save_path: str = None,
):
    """
    Plot a grouped bar chart comparing all model variants.

    WHY A GROUPED BAR CHART:
    A table of numbers is hard to compare at a glance. A grouped bar chart
    makes it immediately obvious which model wins on each metric, and
    reveals the trade-off between approaches visually — which is exactly
    what you'd present to a policy team in a debrief.
    """
    metrics = ['Accuracy', 'Hate F1', 'Hate Precision', 'Hate Recall']
    n_models = len(comparison_df)
    n_metrics = len(metrics)
    x = np.arange(n_models)
    width = 0.20

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#4878CF', '#6ACC65', '#D65F5F', '#B47CC7']

    for i, metric in enumerate(metrics):
        ax.bar(
            x + i * width,
            comparison_df[metric],
            width,
            label=metric,
            color=colors[i],
            alpha=0.85,
        )

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(
        'Model Comparison: Hate Speech Class Performance',
        fontsize=13, fontweight='bold'
    )
    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(comparison_df['Model'], rotation=15, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.1])
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved model comparison chart → {save_path}")
    plt.show()
    return fig


def get_metrics_dict(y_true, y_pred) -> dict:
    """
    Extract all four comparison metrics from a set of predictions.
    Returns a dict suitable for build_model_comparison_table().
    """
    return {
        'accuracy':       round(accuracy_score(y_true, y_pred), 4),
        'hate_f1':        round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        'hate_precision': round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        'hate_recall':    round(recall_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
    }
