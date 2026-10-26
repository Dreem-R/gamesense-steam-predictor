import numpy as np
import pytest

from gamesense.metrics import tier_metrics

Y = np.array([0, 0, 1, 1, 2, 2, 3, 3])


def one_hot(labels: np.ndarray, eps: float = 0.01) -> np.ndarray:
    proba = np.full((len(labels), 4), eps)
    proba[np.arange(len(labels)), labels] = 1 - 3 * eps
    return proba


def test_perfect_predictions():
    m = tier_metrics(Y, one_hot(Y), reviews_true=10.0**Y)
    assert m["accuracy"] == 1
    assert m["macro_f1"] == 1
    assert m["qwk"] == pytest.approx(1)
    assert m["within_1_tier"] == 1
    assert m["auc_ovr"] == 1
    assert m["auc_success_plus"] == 1
    assert m["spearman_reviews"] == pytest.approx(1)


def test_off_by_one_predictions_count_as_within_one_tier():
    pred = np.clip(Y + 1, 0, 3)
    m = tier_metrics(Y, y_pred=pred)
    assert m["within_1_tier"] == 1
    assert m["accuracy"] == pytest.approx(2 / 8)
    assert "log_loss" not in m


def test_success_auc_groups_success_and_hit():
    # Swapping Success and Hit does not change the 100+ reviews ranking.
    swapped = one_hot(np.array([0, 0, 1, 1, 3, 3, 2, 2]))
    assert tier_metrics(Y, swapped)["auc_success_plus"] == 1
