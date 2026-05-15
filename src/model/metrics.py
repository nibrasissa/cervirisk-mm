"""Comprehensive binary classification metrics for CerviRisk-MM.

Two kinds of metrics live here:

  THRESHOLD-FREE (depend only on probabilities, not on a cut-off):
    - AUROC, AUPRC, Brier score, log loss, prevalence

  THRESHOLD-DEPENDENT (require a 0/1 decision via a cut-off):
    - Accuracy, balanced accuracy
    - Sensitivity (recall, TPR), specificity (TNR)
    - Precision (PPV), NPV
    - F1, F0.5, F2 (different precision/recall weightings)
    - FPR, FNR
    - MCC (Matthews correlation coefficient), Cohen's kappa
    - Likelihood ratios LR+, LR-
    - Diagnostic odds ratio
    - Youden's J statistic
    - Confusion-matrix counts: TP, TN, FP, FN

Threshold selection
-------------------
For nested evaluation, the threshold must be picked on DEV OOF predictions
and APPLIED to TEST predictions (never picked on test data). The helpers
youden_threshold() and f1_threshold() do the picking on DEV.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------
def youden_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Threshold that maximizes Youden's J = sensitivity + specificity - 1.

    Equivalent to the point on the ROC curve closest to the top-left corner.
    Returns a probability in [0, 1].
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thresholds[idx])


def f1_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Threshold that maximizes F1 score on the precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns one more precision/recall than thresholds.
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    idx = int(np.argmax(f1))
    return float(thresholds[idx])


# ---------------------------------------------------------------------------
# Safe division — many metrics blow up when a denominator is zero
# ---------------------------------------------------------------------------
def _safe_div(num: float, den: float) -> float | None:
    return float(num) / float(den) if den != 0 else None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def compute_all_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Compute every relevant binary metric.

    Parameters
    ----------
    y_true : array of 0/1 labels
    y_proba : array of predicted probabilities for the positive class
    threshold : if given, also compute threshold-dependent metrics at this cut-off

    Returns
    -------
    dict of metric_name -> value. Threshold-dependent metrics that are
    undefined (e.g. sensitivity when there are no positives) return None.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    out: dict[str, Any] = {}
    n = int(len(y_true))
    n_pos = int(y_true.sum())
    n_neg = n - n_pos
    prev = n_pos / n if n else 0.0

    out["n"] = n
    out["n_positive"] = n_pos
    out["n_negative"] = n_neg
    out["prevalence"] = float(prev)

    # ---- Threshold-free ----
    out["auroc"] = float(roc_auc_score(y_true, y_proba)) if n_pos and n_neg else None
    out["auprc"] = float(average_precision_score(y_true, y_proba)) if n_pos and n_neg else None
    out["brier"] = float(brier_score_loss(y_true, y_proba)) if n else None

    proba_clipped = np.clip(y_proba, 1e-7, 1 - 1e-7)
    out["log_loss"] = float(log_loss(y_true, proba_clipped, labels=[0, 1])) if n else None

    # ---- Threshold-dependent ----
    if threshold is not None:
        out["threshold"] = float(threshold)
        y_pred = (y_proba >= threshold).astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp = int(cm[0, 0]), int(cm[0, 1])
        fn, tp = int(cm[1, 0]), int(cm[1, 1])
        out.update({"tn": tn, "fp": fp, "fn": fn, "tp": tp})

        out["accuracy"] = _safe_div(tp + tn, tp + tn + fp + fn)
        sens = _safe_div(tp, tp + fn)  # recall, sensitivity, TPR
        spec = _safe_div(tn, tn + fp)  # specificity, TNR
        prec = _safe_div(tp, tp + fp)  # precision, PPV
        npv = _safe_div(tn, tn + fn)

        out["sensitivity"] = sens
        out["recall"] = sens  # alias
        out["specificity"] = spec
        out["precision"] = prec
        out["ppv"] = prec  # alias
        out["npv"] = npv
        out["fpr"] = (1 - spec) if spec is not None else None
        out["fnr"] = (1 - sens) if sens is not None else None

        # F-beta family
        if prec is not None and sens is not None and (prec + sens) > 0:
            out["f1"] = 2 * prec * sens / (prec + sens)
            out["f0_5"] = (1 + 0.25) * prec * sens / (0.25 * prec + sens)
            out["f2"] = (1 + 4) * prec * sens / (4 * prec + sens)
        else:
            out["f1"] = out["f0_5"] = out["f2"] = None

        out["balanced_accuracy"] = (
            (sens + spec) / 2 if (sens is not None and spec is not None) else None
        )
        out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
        out["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))
        out["youden_j"] = (
            (sens + spec - 1) if (sens is not None and spec is not None) else None
        )

        # Likelihood ratios — meaningful clinically
        if spec is not None and spec < 1 and sens is not None:
            out["lr_positive"] = sens / (1 - spec) if (1 - spec) > 0 else None
        else:
            out["lr_positive"] = None
        if sens is not None and sens < 1 and spec is not None and spec > 0:
            out["lr_negative"] = (1 - sens) / spec
        else:
            out["lr_negative"] = None

        # Diagnostic odds ratio
        if all(v is not None for v in (out["lr_positive"], out["lr_negative"])) \
                and out["lr_negative"] not in (None, 0):
            out["dor"] = out["lr_positive"] / out["lr_negative"]
        else:
            out["dor"] = None

    return out


# ---------------------------------------------------------------------------
# Aggregate metrics across CV folds
# ---------------------------------------------------------------------------
def aggregate_fold_metrics(fold_metrics: list[dict]) -> dict[str, Any]:
    """Mean and std for each numeric metric across a list of fold dicts.

    None values are skipped (don't count toward mean/std). If all folds had
    None for a metric, the aggregate is None.
    """
    if not fold_metrics:
        return {}

    all_keys = set().union(*(m.keys() for m in fold_metrics))
    out: dict[str, Any] = {}
    for k in sorted(all_keys):
        values = [m.get(k) for m in fold_metrics]
        numeric = [v for v in values if isinstance(v, (int, float)) and v is not None]
        if numeric:
            out[f"{k}_mean"] = float(np.mean(numeric))
            out[f"{k}_std"] = float(np.std(numeric))
            out[f"{k}_n_folds"] = len(numeric)
        else:
            out[f"{k}_mean"] = None
            out[f"{k}_std"] = None
            out[f"{k}_n_folds"] = 0
    return out


# ---------------------------------------------------------------------------
# Pretty printer for human-readable reports
# ---------------------------------------------------------------------------
_METRIC_ORDER = [
    # (key, label, format_spec)
    # {:.1%} auto-multiplies by 100 and appends '%'.
    # Counts use {:.0f} so they handle both int (single-fold) and float (mean-of-folds).
    ("n", "Total samples", "{:.0f}"),
    ("n_positive", "Positives", "{:.0f}"),
    ("n_negative", "Negatives", "{:.0f}"),
    ("prevalence", "Prevalence", "{:.1%}"),
    ("threshold", "Threshold", "{:.4f}"),
    ("tp", "True positives", "{:.0f}"),
    ("fn", "False negatives", "{:.0f}"),
    ("fp", "False positives", "{:.0f}"),
    ("tn", "True negatives", "{:.0f}"),
    ("accuracy", "Accuracy", "{:.1%}"),
    ("balanced_accuracy", "Balanced accuracy", "{:.1%}"),
    ("sensitivity", "Sensitivity (recall, TPR)", "{:.1%}"),
    ("specificity", "Specificity (TNR)", "{:.1%}"),
    ("precision", "Precision (PPV)", "{:.1%}"),
    ("npv", "Negative predictive value", "{:.1%}"),
    ("fpr", "False positive rate", "{:.1%}"),
    ("fnr", "False negative rate", "{:.1%}"),
    ("f1", "F1 score", "{:.1%}"),
    ("f0_5", "F0.5 (precision-weighted)", "{:.1%}"),
    ("f2", "F2 (recall-weighted)", "{:.1%}"),
    ("mcc", "Matthews correlation", "{:.3f}"),
    ("cohen_kappa", "Cohen's kappa", "{:.3f}"),
    ("youden_j", "Youden's J", "{:.3f}"),
    ("lr_positive", "Likelihood ratio +", "{:.2f}"),
    ("lr_negative", "Likelihood ratio -", "{:.2f}"),
    ("dor", "Diagnostic odds ratio", "{:.2f}"),
    ("auroc", "AUROC", "{:.1%}"),
    ("auprc", "AUPRC", "{:.1%}"),
    ("brier", "Brier score", "{:.4f}"),
    ("log_loss", "Log loss", "{:.4f}"),
]


def format_metrics_block(metrics: dict, title: str = "") -> str:
    """Multi-line formatter for a single metrics dict (one fold or one CV pool)."""
    lines = []
    if title:
        lines.append(title)
        lines.append("-" * len(title))
    for key, label, fmt in _METRIC_ORDER:
        if key in metrics and metrics[key] is not None:
            try:
                lines.append(f"  {label:<32s} {fmt.format(metrics[key])}")
            except (ValueError, TypeError):
                lines.append(f"  {label:<32s} {metrics[key]}")
    return "\n".join(lines)


def format_aggregated_metrics_block(aggregated: dict, title: str = "") -> str:
    """Formatter for aggregate_fold_metrics() output (keys like 'auprc_mean')."""
    lines = []
    if title:
        lines.append(title)
        lines.append("-" * len(title))
    for key, label, fmt in _METRIC_ORDER:
        mean = aggregated.get(f"{key}_mean")
        std = aggregated.get(f"{key}_std")
        if mean is None:
            continue
        try:
            if std is not None:
                base = fmt.format(mean)
                std_text = fmt.format(std)
                # For percentage formatting, drop the trailing '%' on std so
                # the line reads e.g. "17.7% ± 3.0%" cleanly.
                text = f"{base} ± {std_text}"
            else:
                text = fmt.format(mean)
            lines.append(f"  {label:<32s} {text}")
        except (ValueError, TypeError):
            lines.append(f"  {label:<32s} {mean}")
    return "\n".join(lines)
