import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from utils.classification_metrics import calculate
from utils.evaluation import evaluate_batches


def test_confusion_and_per_class_scores():
    labels = np.array([0, 0, 1, 1])
    scores = np.zeros((4, 10))
    scores[np.arange(4), [0, 1, 1, 1]] = 2
    result = calculate(labels, scores)
    assert result["accuracy"] == 0.75
    assert np.array(result["confusion_matrix"])[:2, :2].tolist() == [[1, 1], [0, 2]]
    assert result["per_class"][0]["precision"] == 1
    assert result["per_class"][0]["recall"] == 0.5
    assert result["per_class"][1]["precision"] == pytest.approx(2 / 3)
    assert result["per_class"][1]["f1"] == pytest.approx(0.8)
    assert result["macro_auc_ovr"] is None  # classes 2–9 are absent


def test_auc_ties_and_perfect_multiclass_separation():
    labels = np.repeat(np.arange(10), 2)
    perfect = calculate(labels, np.eye(10)[labels])
    assert perfect["macro_auc_ovr"] == 1.0
    tied = calculate(labels, np.zeros((20, 10)))
    assert tied["macro_auc_ovr"] == 0.5
    assert sum(row["support"] for row in tied["per_class"]) == 20


def test_checkpoint_and_training_use_same_score_evaluation():
    labels = torch.arange(10)
    scores = torch.eye(10)
    loader = DataLoader(TensorDataset(scores, labels), batch_size=3)
    expected = calculate(labels.numpy(), scores.numpy())
    assert evaluate_batches(loader, lambda images: images) == expected
