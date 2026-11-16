import io
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cli import EXIT_ERROR, EXIT_OK, EXIT_TRIGGER, main  # noqa: E402
from generator import generate_all, records_to_rows  # noqa: E402
from model import LogisticModel  # noqa: E402
from registry import add_version, evaluate, make_entry, new_registry, promote  # noqa: E402


def _run(argv):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return main(argv)


class TestRegistry(unittest.TestCase):
    def test_entry_fields(self):
        data = generate_all()
        X, y = records_to_rows(data["train"])
        m = LogisticModel(4).fit(X, y)
        Xh, yh = records_to_rows(data["holdout"])
        e = make_entry("v1", "period1", m, evaluate(m, Xh, yh), "champion")
        for field in ("version", "trained_on", "features", "coefficients_hash",
                      "holdout_auc", "holdout_brier", "calibration_deciles", "status"):
            self.assertIn(field, e)
        self.assertEqual(len(e["calibration_deciles"]), 10)

    def test_promote_archives_old(self):
        reg = new_registry()
        reg["versions"] = [{"version": "v1", "status": "champion"},
                           {"version": "v2", "status": "challenger"}]
        promote(reg, "v2")
        self.assertEqual(reg["versions"][0]["status"], "archived")
        self.assertEqual(reg["versions"][1]["status"], "champion")


class TestCLI(unittest.TestCase):
    def test_check_exit_codes(self):
        self.assertEqual(_run(["check", "--scenario", "drift"]), EXIT_TRIGGER)
        self.assertEqual(_run(["check", "--scenario", "control"]), EXIT_OK)
        self.assertEqual(_run(["check", "--scenario", "drift",
                               "--registry", "/nonexistent-registry.json"]), EXIT_ERROR)

    def test_generate_and_train_ok(self):
        self.assertEqual(_run(["generate"]), EXIT_OK)
        self.assertEqual(_run(["train"]), EXIT_OK)

    def test_monitor_ok(self):
        self.assertEqual(_run(["monitor", "--scenario", "control"]), EXIT_OK)


if __name__ == "__main__":
    unittest.main()
