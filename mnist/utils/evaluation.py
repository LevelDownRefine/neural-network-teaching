"""Collect model scores and calculate the shared classification metrics."""

import torch
from utils.classification_metrics import calculate


def evaluate_batches(loader, predict, max_batches=0, observe=None):
    labels, scores = [], []
    with torch.no_grad():
        for index, (images, target) in enumerate(loader):
            output = predict(images)
            if observe is not None:
                observe(target, output)
            labels.extend(target.cpu().tolist())
            scores.extend(output.cpu().tolist())
            if max_batches and index + 1 >= max_batches:
                break
    return calculate(labels, scores)
