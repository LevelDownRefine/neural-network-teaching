import numpy as np


def calculate(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    predicted = scores.argmax(axis=1)
    cm = np.bincount(labels * 10 + predicted, minlength=100).reshape(10, 10)
    tp = cm.diagonal()
    precision = np.divide(tp, cm.sum(0), out=np.zeros(10), where=cm.sum(0) != 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(10), where=cm.sum(1) != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(10),
        where=precision + recall != 0,
    )
    probability = np.exp(scores - scores.max(axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)
    aucs = []
    for i in range(10):
        s = probability[:, i]
        y = labels == i
        n = y.sum()
        neg = len(y) - n
        if n == 0 or neg == 0:
            aucs.append(None)
            continue
        order = np.argsort(s, kind="mergesort")
        ss = s[order]
        starts = np.r_[0, np.flatnonzero(np.diff(ss) != 0) + 1]
        ends = np.r_[starts[1:], len(s)]
        ranks = np.empty(len(s))
        for a, b in zip(starts, ends):
            ranks[order[a:b]] = (a + 1 + b) / 2
        aucs.append(float((ranks[y].sum() - n * (n + 1) / 2) / (n * neg)))
    return {
        "accuracy": float((predicted == labels).mean()),
        "count": len(labels),
        "confusion_matrix": cm.tolist(),
        "per_class": [
            {
                "digit": i,
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(cm[i].sum()),
                "auc": aucs[i],
            }
            for i in range(10)
        ],
        "macro_auc_ovr": float(np.mean(aucs))
        if all(x is not None for x in aucs)
        else None,
        "auc_score": "softmax of model output scores",
    }
