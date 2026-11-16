"""Logistic regression from scratch (full-batch gradient descent)."""
from __future__ import annotations

import hashlib
import json
import math


def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class LogisticModel:
    def __init__(self, n_features):
        self.n_features = n_features
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.means = [0.0] * n_features
        self.stds = [1.0] * n_features

    def _standardize_row(self, row):
        return [(row[j] - self.means[j]) / self.stds[j] for j in range(self.n_features)]

    def fit(self, X, y, lr=0.1, epochs=400, l2=0.01):
        n = len(X)
        d = self.n_features
        self.means = [sum(row[j] for row in X) / n for j in range(d)]
        self.stds = []
        for j in range(d):
            var = sum((row[j] - self.means[j]) ** 2 for row in X) / n
            s = math.sqrt(var)
            self.stds.append(s if s > 1e-9 else 1.0)
        Xs = [self._standardize_row(r) for r in X]
        w = [0.0] * d
        b = 0.0
        for _ in range(epochs):
            grad_w = [0.0] * d
            grad_b = 0.0
            for i in range(n):
                logit = b + sum(w[j] * Xs[i][j] for j in range(d))
                err = _sigmoid(logit) - y[i]
                for j in range(d):
                    grad_w[j] += err * Xs[i][j]
                grad_b += err
            for j in range(d):
                w[j] -= lr * (grad_w[j] / n + l2 * w[j])
            b -= lr * (grad_b / n)
        self.weights = w
        self.bias = b
        return self

    def predict_proba(self, X):
        out = []
        for row in X:
            z = self._standardize_row(row)
            logit = self.bias + sum(self.weights[j] * z[j] for j in range(self.n_features))
            out.append(_sigmoid(logit))
        return out

    def coefficients_hash(self):
        payload = json.dumps({"w": [round(v, 8) for v in self.weights],
                              "b": round(self.bias, 8)}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self):
        return {"weights": self.weights, "bias": self.bias,
                "means": self.means, "stds": self.stds}

    @classmethod
    def from_dict(cls, d):
        m = cls(len(d["weights"]))
        m.weights = list(d["weights"]); m.bias = d["bias"]
        m.means = list(d["means"]); m.stds = list(d["stds"])
        return m
