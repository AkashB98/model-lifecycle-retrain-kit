"""JSON model registry: versions, status, metrics, coefficient hash."""
from __future__ import annotations

import json
import os

from generator import FEATURES
from metrics import auc, brier_score, calibration_by_decile


def evaluate(model, X, y):
    probs = model.predict_proba(X)
    cal = calibration_by_decile(probs, y)
    return {"auc": round(auc(probs, y), 6),
            "brier": round(brier_score(probs, y), 6),
            "calibration_deciles": cal}


def make_entry(version, trained_on, model, metrics_dict, status):
    return {"version": version, "trained_on": trained_on,
            "features": list(FEATURES),
            "coefficients_hash": model.coefficients_hash(),
            "coefficients_summary": [round(w, 4) for w in model.weights],
            "bias": round(model.bias, 4),
            "holdout_auc": metrics_dict["auc"],
            "holdout_brier": metrics_dict["brier"],
            "calibration_deciles": metrics_dict["calibration_deciles"],
            "status": status,
            "model": model.to_dict()}


def new_registry():
    return {"model_name": "synthetic-credit-logit", "versions": []}


def save_registry(reg, path):
    with open(path, "w") as f:
        json.dump(reg, f, indent=2, sort_keys=True)
        f.write("\n")


def load_registry(path):
    with open(path) as f:
        return json.load(f)


def champion_entry(reg):
    for e in reg["versions"]:
        if e["status"] == "champion":
            return e
    return None


def add_version(reg, entry):
    reg["versions"].append(entry)
    return reg


def promote(reg, version):
    """Archive the old champion, promote the named challenger."""
    for e in reg["versions"]:
        if e["status"] == "champion":
            e["status"] = "archived"
    for e in reg["versions"]:
        if e["version"] == version:
            e["status"] = "champion"
    return reg


def default_registry_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "registry.json")
