import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generator import generate_all  # noqa: E402
from lifecycle import gate_decision, run_lifecycle  # noqa: E402
from registry import champion_entry  # noqa: E402


class TestGate(unittest.TestCase):
    def test_promote_when_margin_and_brier_ok(self):
        g = gate_decision(0.72, 0.16, 0.78, 0.13)
        self.assertTrue(g["promote"])

    def test_reject_when_margin_fails(self):
        # challenger better, but by less than the 0.01 pre-declared margin
        g = gate_decision(0.80, 0.15, 0.805, 0.14)
        self.assertFalse(g["promote"])
        self.assertFalse(g["auc_ok"])
        self.assertTrue(g["brier_ok"])

    def test_reject_when_brier_worse(self):
        g = gate_decision(0.72, 0.13, 0.80, 0.16)
        self.assertFalse(g["promote"])
        self.assertTrue(g["auc_ok"])
        self.assertFalse(g["brier_ok"])

    def test_exact_margin_boundary_promotes(self):
        g = gate_decision(0.70, 0.15, 0.71, 0.15)
        self.assertTrue(g["promote"])


class TestLifecycleScenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = generate_all()

    def test_drift_promotes(self):
        report, reg = run_lifecycle(self.data, scenario="drift")
        self.assertTrue(report["monitoring"]["trigger_fired"])
        self.assertTrue(report["monitoring"]["psi_trigger"])
        self.assertTrue(report["monitoring"]["auc_trigger"])
        self.assertTrue(report["retrained"])
        self.assertTrue(report["promoted"])
        self.assertEqual(report["final_champion"], "v2")
        self.assertEqual(champion_entry(reg)["version"], "v2")
        v1 = next(v for v in reg["versions"] if v["version"] == "v1")
        self.assertEqual(v1["status"], "archived")

    def test_control_no_trigger_no_retrain(self):
        report, reg = run_lifecycle(self.data, scenario="control")
        self.assertFalse(report["monitoring"]["trigger_fired"])
        self.assertFalse(report["retrained"])
        self.assertFalse(report["promoted"])
        self.assertEqual(report["final_champion"], "v1")
        self.assertEqual(len(reg["versions"]), 1)

    def test_headline_numbers_pinned(self):
        report, _ = run_lifecycle(self.data, scenario="drift")
        self.assertEqual(report["champion"]["holdout_auc"], 0.783209)
        self.assertEqual(report["monitoring"]["psi"], 1.11443)
        self.assertEqual(report["monitoring"]["current_auc"], 0.725005)
        self.assertEqual(report["challenger"]["fresh_auc"], 0.786853)

    def test_determinism(self):
        r1, reg1 = run_lifecycle(self.data, scenario="drift")
        r2, reg2 = run_lifecycle(self.data, scenario="drift")
        self.assertEqual(json.dumps(r1, sort_keys=True), json.dumps(r2, sort_keys=True))
        self.assertEqual(json.dumps(reg1, sort_keys=True), json.dumps(reg2, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
