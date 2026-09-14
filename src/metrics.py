"""Evaluation helpers."""
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             f1_score, log_loss, roc_auc_score)


def tier_metrics(y_true, proba=None, y_pred=None, reviews_true=None, score=None) -> dict:
    """proba: class probabilities (n, 4). score: continuous score for rank
    correlation, defaults to the expected tier."""
    y_true = np.asarray(y_true)
    if y_pred is None:
        y_pred = proba.argmax(1)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "within_1_tier": float(np.mean(np.abs(y_true - y_pred) <= 1)),
    }
    if proba is not None:
        out["log_loss"] = log_loss(y_true, proba, labels=[0, 1, 2, 3])
        out["auc_ovr"] = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
        # Binary view: Success or Hit (100+ reviews) vs the rest.
        out["auc_success_plus"] = roc_auc_score(y_true >= 2, proba[:, 2:].sum(1))
        if score is None:
            score = proba @ np.arange(proba.shape[1])
    if score is not None and reviews_true is not None:
        out["spearman_reviews"] = spearmanr(score, reviews_true).correlation
    return out
