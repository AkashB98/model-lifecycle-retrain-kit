"""Deterministic synthetic credit dataset, seed 42.

Two periods:
- period 1: original relationship
- period 2: planted drift (utilization distribution shifts up and the
  true outcome relationship rotates: income matters much more, while
  utilization, delinquency history, and tenure matter much less).
  A model trained on period 1 degrades on period 2.
- control: a later period with NO drift (same distribution and same
  relationship as period 1) -- triggers must stay silent there.

Fictional borrowers only. Stdlib only.
"""
from __future__ import annotations

import math
import random

FEATURES = ["income_k", "utilization", "delinquency_count", "tenure_years"]
SEED = 42

# True (data-generating) weights -- NOT the fitted model weights.
_W_P1 = {"intercept": -1.0, "income_k": -0.028, "utilization": 3.0,
         "delinquency_count": 0.95, "tenure_years": -0.13}
_W_P2_DRIFT = {"intercept": 1.45, "income_k": -0.055, "utilization": 0.8,
               "delinquency_count": 0.15, "tenure_years": 0.0}


def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def _one(rng, weights, drift_dist):
    income_k = _clip(rng.gauss(68.0, 22.0), 18.0, 160.0)
    if drift_dist:
        utilization = _clip(rng.gauss(0.62, 0.20), 0.02, 0.98)
    else:
        utilization = _clip(rng.gauss(0.42, 0.20), 0.02, 0.98)
    # delinquency: mostly 0, tail to 4
    r = rng.random()
    if r < 0.62:
        delinq = 0
    elif r < 0.82:
        delinq = 1
    elif r < 0.93:
        delinq = 2
    elif r < 0.98:
        delinq = 3
    else:
        delinq = 4
    tenure = _clip(rng.gauss(5.5, 3.2), 0.3, 16.0)
    feats = {"income_k": round(income_k, 2), "utilization": round(utilization, 4),
             "delinquency_count": delinq, "tenure_years": round(tenure, 2)}
    logit = (weights["intercept"] + weights["income_k"] * income_k
             + weights["utilization"] * utilization
             + weights["delinquency_count"] * delinq
             + weights["tenure_years"] * tenure)
    p = _sigmoid(logit)
    label = 1 if rng.random() < p else 0
    return {"features": feats, "label": label}


def generate_period(n, period="period1", drift=False, seed=SEED):
    """Generate n records for one period.

    period labels: 'period1', 'period2', 'control'
    drift=True selects the drifted distribution + relationship.
    Seed is derived deterministically from (seed, period) so each
    period is reproducible independently.
    """
    offset = {"period1": 0, "period2": 100000, "control": 200000,
              "period2_holdout": 300000, "period1_holdout": 400000,
              "control_holdout": 500000}.get(period, 900000)
    rng = random.Random(seed + offset + n)
    weights = _W_P2_DRIFT if drift else _W_P1
    return [_one(rng, weights, drift) for _ in range(n)]


def generate_all(seed=SEED):
    """The canonical lifecycle dataset (deterministic).

    Returns dict with train / holdout / recent / fresh_holdout /
    control_recent / control_holdout splits.
    """
    return {
        # period 1
        "train": generate_period(600, "period1", drift=False, seed=seed),
        "holdout": generate_period(400, "period1_holdout", drift=False, seed=seed),
        # period 2 (drifted production)
        "recent": generate_period(600, "period2", drift=True, seed=seed),
        "fresh_holdout": generate_period(400, "period2_holdout", drift=True, seed=seed),
        # control (no drift, later period, same law as period 1)
        "control_recent": generate_period(600, "control", drift=False, seed=seed),
        "control_holdout": generate_period(400, "control_holdout", drift=False, seed=seed),
    }


def records_to_rows(records):
    """Flatten records to (X, y) in FEATURES order."""
    X = [[float(r["features"][f]) for f in FEATURES] for r in records]
    y = [int(r["label"]) for r in records]
    return X, y
