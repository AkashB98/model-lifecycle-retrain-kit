"""End-to-end demo: drift scenario promotes, control stays put."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generator import generate_all  # noqa: E402
from lifecycle import run_lifecycle  # noqa: E402


def run():
    data = generate_all()
    print("== model-lifecycle-retrain-kit demo ==")
    for scenario in ("drift", "control"):
        report, _reg = run_lifecycle(data, scenario=scenario)
        mon = report["monitoring"]
        print(f"\n[{scenario}] champion v1 holdout AUC {report['champion']['holdout_auc']}")
        print(f"[{scenario}] PSI(utilization) {mon['psi']} (threshold {mon['psi_threshold']}) "
              f"trigger={mon['psi_trigger']}")
        print(f"[{scenario}] AUC {mon['reference_auc']} -> {mon['current_auc']} "
              f"(drop {mon['auc_drop']}, threshold {mon['auc_drop_threshold']}) "
              f"trigger={mon['auc_trigger']}")
        if not report["retrained"]:
            print(f"[{scenario}] no trigger -> no retrain; champion stays v1")
            assert report["final_champion"] == "v1"
        else:
            g = report["gate"]
            print(f"[{scenario}] challenger v2 fresh AUC {report['challenger']['fresh_auc']} "
                  f"vs champion {report['challenger']['champion_fresh_auc']}; "
                  f"gate promote={g['promote']}")
            print(f"[{scenario}] final champion: {report['final_champion']}")
    drift_report, _ = run_lifecycle(data, scenario="drift")
    assert drift_report["promoted"] is True, "drift scenario must promote"
    print("\ndemo OK: drift -> retrained + promoted; control -> no trigger, no retrain")
    return 0


if __name__ == "__main__":
    sys.exit(run())
