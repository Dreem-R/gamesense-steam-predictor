import numpy as np
import pytest

from gamesense.metrics import outcome_metrics, tier_metrics

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


def test_outcome_metrics_rates_and_ranking():
    happened = [False, False, False, True, True]
    m = outcome_metrics(happened, [0.02, 0.04, 0.25, 0.6, 0.9])
    assert m["games"] == 5
    assert m["actual_rate"] == pytest.approx(0.4)
    assert m["predicted_rate"] == pytest.approx(0.362)
    assert m["auc"] == 1


def test_outcome_metrics_calibration_bands():
    happened = [False, True, False, True, True, True]
    m = outcome_metrics(happened, [0.01, 0.03, 0.25, 0.8, 0.9, 1.0])
    bands = {(b["from"], b["to"]): b for b in m["calibration"]}
    assert set(bands) == {(0.0, 0.05), (0.2, 0.3), (0.7, 1.0)}
    assert bands[(0.0, 0.05)]["games"] == 2
    assert bands[(0.0, 0.05)]["actual"] == pytest.approx(0.5)
    assert bands[(0.7, 1.0)]["games"] == 3
    assert bands[(0.7, 1.0)]["predicted"] == pytest.approx(0.9)
