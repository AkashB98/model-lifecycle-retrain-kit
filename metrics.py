"""Metrics from scratch: AUC (Mann-Whitney), Brier, PSI, calibration."""
from __future__ import annotations

import math


def auc(scores, labels):
    """Rank-based AUC (Mann-Whitney U), ties take average ranks."""
    n = len(scores)
    if n == 0:
        return 0.5
    n_pos = sum(1 for y in labels if y == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum_pos = sum(ranks[i] for i in range(n) if labels[i] == 1)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def auc_bruteforce(scores, labels):
    """Pairwise definition, for cross-checking auc() in tests/evals."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for q in neg:
            if p > q:
                wins += 1.0
            elif p == q:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def brier_score(probs, labels):
    if not probs:
        return 0.0
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs)


def psi(reference, current, bins=10, lo=0.0, hi=1.0, eps=0.0001):
    """Population Stability Index over equal-width bins in [lo, hi]."""
    def shares(vals):
        counts = [0] * bins
        for v in vals:
            idx = int((v - lo) / (hi - lo) * bins)
            idx = max(0, min(bins - 1, idx))
            counts[idx] += 1
        total = len(vals) or 1
        return [max(c / total, eps) for c in counts]
    ref = shares(reference)
    cur = shares(current)
    return sum((c - r) * math.log(c / r) for c, r in zip(cur, ref))


def calibration_by_decile(probs, labels, n_bins=10):
    """Mean predicted vs observed rate per probability decile bin."""
    pairs = sorted(zip(probs, labels))
    out = []
    n = len(pairs)
    for b in range(n_bins):
        chunk = pairs[b * n // n_bins:(b + 1) * n // n_bins]
        if not chunk:
            continue
        mp = sum(p for p, _ in chunk) / len(chunk)
        obs = sum(y for _, y in chunk) / len(chunk)
        out.append({"bin": b, "n": len(chunk),
                    "mean_predicted": round(mp, 6), "observed": round(obs, 6)})
    return out
