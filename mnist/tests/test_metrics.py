import numpy as np
import pytest

from utils.classification_metrics import calculate


def test_confusion_and_per_class_scores():
    labels = np.array([0, 0, 1, 1])
    scores = np.zeros((4, 10))
    scores[np.arange(4), [0, 1, 1, 1]] = 2
    result = calculate(labels, scores)
    assert result['accuracy'] == .75
    assert np.array(result['confusion_matrix'])[:2, :2].tolist() == [[1, 1], [0, 2]]
    assert result['per_class'][0]['precision'] == 1
    assert result['per_class'][0]['recall'] == .5
    assert result['per_class'][1]['precision'] == pytest.approx(2 / 3)
    assert result['per_class'][1]['f1'] == pytest.approx(.8)
    assert result['macro_auc_ovr'] is None  # classes 2–9 are absent


def test_auc_ties_and_perfect_multiclass_separation():
    labels = np.repeat(np.arange(10), 2)
    perfect = calculate(labels, np.eye(10)[labels])
    assert perfect['macro_auc_ovr'] == 1.0
    tied = calculate(labels, np.zeros((20, 10)))
    assert tied['macro_auc_ovr'] == .5
    assert sum(row['support'] for row in tied['per_class']) == 20
