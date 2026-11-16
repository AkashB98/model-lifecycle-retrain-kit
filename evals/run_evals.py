"""Golden evals for model-lifecycle-retrain-kit.

Deterministic: eval_report.json is byte-identical across runs
(no timestamps, no temp paths). Run: python3 evals/run_evals.py
"""
from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli import EXIT_ERROR, EXIT_OK, EXIT_TRIGGER, main as cli_main  # noqa: E402
from generator import generate_all  # noqa: E402
from lifecycle import gate_decision, run_lifecycle  # noqa: E402
from metrics import auc, auc_bruteforce, psi  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = []


def check(name, passed, detail=""):
    RESULTS.append({"name": name, "passed": bool(passed), "detail": detail})
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    return passed


def _run_cli(argv):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return cli_main(argv)


def main():
    print("== golden evals: model-lifecycle-retrain-kit ==")
    data = generate_all()
    drift, _reg = run_lifecycle(data, scenario="drift")
    control, _reg2 = run_lifecycle(data, scenario="control")
    mon = drift["monitoring"]

    check("G1 champion AUC on period-1 holdout pinned",
          drift["champion"]["holdout_auc"] == 0.783209
          and drift["champion"]["holdout_brier"] == 0.135441,
          f"auc={drift['champion']['holdout_auc']} brier={drift['champion']['holdout_brier']}")

    check("G2 drift PSI on utilization fires (PSI > 0.25)",
          mon["psi"] == 1.11443 and mon["psi_trigger"] is True,
          f"psi={mon['psi']}")

    check("G3 period-2 AUC drop fires (drop > 0.03)",
          mon["current_auc"] == 0.725005 and mon["auc_drop"] == 0.058204
          and mon["auc_trigger"] is True and mon["trigger_fired"] is True,
          f"drop={mon['auc_drop']}")

    check("G4 challenger promoted in drift scenario (wins by >= 0.01, Brier not worse)",
          drift["retrained"] is True and drift["promoted"] is True
          and drift["challenger"]["fresh_auc"] == 0.786853
          and drift["challenger"]["fresh_brier"] == 0.132616
          and drift["final_champion"] == "v2",
          f"challenger_auc={drift['challenger']['fresh_auc']}")

    check("G5 control: no trigger, no retrain (PSI pinned)",
          control["monitoring"]["psi"] == 0.044039
          and control["monitoring"]["trigger_fired"] is False
          and control["retrained"] is False and control["promoted"] is False
          and control["final_champion"] == "v1",
          f"control_psi={control['monitoring']['psi']}")

    g_margin = gate_decision(0.80, 0.15, 0.805, 0.14)
    g_brier = gate_decision(0.72, 0.13, 0.80, 0.16)
    check("G6 gate rejects challenger failing margin / failing Brier",
          g_margin["promote"] is False and g_margin["auc_ok"] is False
          and g_brier["promote"] is False and g_brier["brier_ok"] is False,
          "margin and brier rejections both pinned")

    hand = auc([0.9, 0.4, 0.5, 0.1], [1, 1, 0, 0])
    small_scores = [0.3, 0.7, 0.7, 0.2, 0.9, 0.7, 0.1]
    small_labels = [1, 0, 1, 0, 1, 0, 0]
    psi_hand = psi([0.1, 0.1, 0.9, 0.9], [0.1, 0.9, 0.9, 0.9], bins=2)
    check("G7 AUC/PSI match hand-computed examples + brute force",
          hand == 0.75
          and abs(auc(small_scores, small_labels) - auc_bruteforce(small_scores, small_labels)) < 1e-9
          and round(psi_hand, 6) == 0.274653,
          f"hand_auc={hand} psi_hand={round(psi_hand, 6)}")

    codes = (_run_cli(["check", "--scenario", "drift"]),
             _run_cli(["check", "--scenario", "control"]),
             _run_cli(["check", "--scenario", "drift", "--registry", "/nonexistent-registry.json"]))
    drift2, _ = run_lifecycle(generate_all(), scenario="drift")
    check("G8 CLI check exit codes 1/0/2 + lifecycle determinism",
          codes == (EXIT_TRIGGER, EXIT_OK, EXIT_ERROR)
          and json.dumps(drift, sort_keys=True) == json.dumps(drift2, sort_keys=True),
          f"codes={codes}")

    passed = sum(1 for r in RESULTS if r["passed"])
    report = {"kit": "model-lifecycle-retrain-kit",
              "passed": passed, "total": len(RESULTS), "results": RESULTS}
    out = os.path.join(HERE, "eval_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"evals: {passed}/{len(RESULTS)} passed -> {out}")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
