"""Monitoring (narrow) + gated retrain/promotion loop.

Monitoring is exactly two checks (per design):
  1. PSI on the shifted feature (utilization), threshold 0.25.
  2. AUC-drop vs the champion's reference (period-1 holdout) AUC,
     threshold 0.03.
Anything broader (a full governance evidence pack) lives in the
sibling model-governance-evidence-kit, not here.

Promotion gate (pre-declared): a challenger trained on recent data is
promoted ONLY if, on a fresh holdout, its AUC >= champion AUC + 0.01
AND its Brier score is not worse than the champion's.
"""
from __future__ import annotations

from generator import records_to_rows
from metrics import auc, psi
from model import LogisticModel
from registry import add_version, evaluate, make_entry, promote

PSI_THRESHOLD = 0.25
AUC_DROP_THRESHOLD = 0.03
PROMOTION_MARGIN = 0.01
DRIFT_FEATURE = "utilization"


def monitor(champion_model, reference_records, reference_auc, current_records, current_labels_records):
    """Run the two monitoring checks.

    reference_records: period-1 records (drift-feature reference).
    current_records: production-period records to score for AUC.
    """
    ref_vals = [r["features"][DRIFT_FEATURE] for r in reference_records]
    cur_vals = [r["features"][DRIFT_FEATURE] for r in current_records]
    psi_val = psi(ref_vals, cur_vals)
    X, y = records_to_rows(current_labels_records)
    probs = champion_model.predict_proba(X)
    cur_auc = auc(probs, y)
    drop = reference_auc - cur_auc
    psi_fired = psi_val > PSI_THRESHOLD
    auc_fired = drop > AUC_DROP_THRESHOLD
    return {"drift_feature": DRIFT_FEATURE,
            "psi": round(psi_val, 6), "psi_threshold": PSI_THRESHOLD,
            "psi_trigger": psi_fired,
            "reference_auc": round(reference_auc, 6),
            "current_auc": round(cur_auc, 6),
            "auc_drop": round(drop, 6),
            "auc_drop_threshold": AUC_DROP_THRESHOLD,
            "auc_trigger": auc_fired,
            "trigger_fired": bool(psi_fired or auc_fired)}


def gate_decision(champion_auc_fresh, champion_brier_fresh,
                  challenger_auc_fresh, challenger_brier_fresh,
                  margin=PROMOTION_MARGIN):
    """Pure promotion gate -- unit-testable without any training."""
    auc_ok = challenger_auc_fresh >= champion_auc_fresh + margin
    brier_ok = challenger_brier_fresh <= champion_brier_fresh
    return {"promote": bool(auc_ok and brier_ok),
            "margin": margin,
            "auc_ok": bool(auc_ok), "brier_ok": bool(brier_ok),
            "champion_auc": round(champion_auc_fresh, 6),
            "challenger_auc": round(challenger_auc_fresh, 6),
            "champion_brier": round(champion_brier_fresh, 6),
            "challenger_brier": round(challenger_brier_fresh, 6)}


def run_lifecycle(data, scenario="drift"):
    """Full loop for one scenario ('drift' or 'control').

    Returns a deterministic report dict (rounded numbers only).
    """
    if scenario == "drift":
        recent, fresh = data["recent"], data["fresh_holdout"]
    else:
        recent, fresh = data["control_recent"], data["control_holdout"]

    Xtr, ytr = records_to_rows(data["train"])
    champion = LogisticModel(len(Xtr[0])).fit(Xtr, ytr)
    Xh, yh = records_to_rows(data["holdout"])
    champ_holdout = evaluate(champion, Xh, yh)

    from registry import new_registry
    reg = new_registry()
    add_version(reg, make_entry("v1", "period1", champion, champ_holdout, "champion"))

    mon = monitor(champion, data["train"], champ_holdout["auc"], recent, fresh)

    report = {"scenario": scenario,
              "champion": {"version": "v1", "holdout_auc": champ_holdout["auc"],
                           "holdout_brier": champ_holdout["brier"]},
              "monitoring": mon,
              "retrained": False, "challenger": None,
              "gate": None, "promoted": False,
              "final_champion": "v1"}

    if not mon["trigger_fired"]:
        return report, reg

    # Trigger fired -> train challenger on recent data only.
    Xr, yr = records_to_rows(recent)
    challenger = LogisticModel(len(Xr[0])).fit(Xr, yr)
    Xf, yf = records_to_rows(fresh)
    champ_fresh = evaluate(champion, Xf, yf)
    chall_fresh = evaluate(challenger, Xf, yf)
    gate = gate_decision(champ_fresh["auc"], champ_fresh["brier"],
                         chall_fresh["auc"], chall_fresh["brier"])
    status = "challenger"
    add_version(reg, make_entry("v2", "period2_recent" if scenario == "drift" else "control_recent",
                                challenger, chall_fresh, status))
    report["retrained"] = True
    report["challenger"] = {"version": "v2", "fresh_auc": chall_fresh["auc"],
                            "fresh_brier": chall_fresh["brier"],
                            "champion_fresh_auc": champ_fresh["auc"],
                            "champion_fresh_brier": champ_fresh["brier"]}
    report["gate"] = gate
    if gate["promote"]:
        promote(reg, "v2")
        report["promoted"] = True
        report["final_champion"] = "v2"
    return report, reg
