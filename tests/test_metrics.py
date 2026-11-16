import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics import auc, auc_bruteforce, brier_score, calibration_by_decile, psi  # noqa: E402


class TestAUC(unittest.TestCase):
    def test_hand_computed_075(self):
        # pos scores 0.9, 0.4 vs neg 0.5, 0.1 -> 3 of 4 pairs correct
        self.assertAlmostEqual(auc([0.9, 0.4, 0.5, 0.1], [1, 1, 0, 0]), 0.75, places=9)

    def test_perfect_and_reversed(self):
        self.assertEqual(auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]), 1.0)
        self.assertEqual(auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]), 0.0)

    def test_all_tied_is_half(self):
        self.assertEqual(auc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]), 0.5)

    def test_matches_bruteforce_with_ties(self):
        scores = [0.3, 0.7, 0.7, 0.2, 0.9, 0.7, 0.1]
        labels = [1, 0, 1, 0, 1, 0, 0]
        self.assertAlmostEqual(auc(scores, labels), auc_bruteforce(scores, labels), places=9)

    def test_single_class_is_half(self):
        self.assertEqual(auc([0.3, 0.9], [1, 1]), 0.5)


class TestPSI(unittest.TestCase):
    def test_identical_distributions_zero(self):
        vals = [0.1, 0.3, 0.5, 0.7, 0.9] * 4
        self.assertAlmostEqual(psi(vals, list(vals)), 0.0, places=9)

    def test_hand_computed(self):
        ref = [0.1, 0.1, 0.9, 0.9]
        cur = [0.1, 0.9, 0.9, 0.9]
        expected = (0.25 - 0.5) * math.log(0.25 / 0.5) + (0.75 - 0.5) * math.log(0.75 / 0.5)
        self.assertAlmostEqual(psi(ref, cur, bins=2), expected, places=9)
        self.assertAlmostEqual(expected, 0.274653, places=5)

    def test_shifted_distribution_large_psi(self):
        ref = [0.2] * 50 + [0.4] * 50
        cur = [0.8] * 50 + [0.95] * 50
        self.assertGreater(psi(ref, cur), 0.25)


class TestBrierCalibration(unittest.TestCase):
    def test_brier_hand(self):
        self.assertAlmostEqual(brier_score([0.9, 0.1], [1, 0]), 0.01, places=9)
        self.assertAlmostEqual(brier_score([0.5, 0.5], [1, 0]), 0.25, places=9)

    def test_calibration_bins(self):
        probs = [0.05] * 10 + [0.95] * 10
        labels = [0] * 10 + [1] * 10
        cal = calibration_by_decile(probs, labels)
        self.assertEqual(len(cal), 10)
        self.assertEqual(cal[0]["observed"], 0.0)
        self.assertEqual(cal[-1]["observed"], 1.0)


if __name__ == "__main__":
    unittest.main()
