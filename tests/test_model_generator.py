import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generator import FEATURES, generate_all, generate_period, records_to_rows  # noqa: E402
from metrics import auc  # noqa: E402
from model import LogisticModel  # noqa: E402


class TestGenerator(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(generate_all(), generate_all())

    def test_sizes_and_schema(self):
        data = generate_all()
        self.assertEqual(len(data["train"]), 600)
        self.assertEqual(len(data["holdout"]), 400)
        self.assertEqual(len(data["recent"]), 600)
        self.assertEqual(len(data["fresh_holdout"]), 400)
        rec = data["train"][0]
        self.assertEqual(sorted(rec["features"].keys()), sorted(FEATURES))
        self.assertIn(rec["label"], (0, 1))

    def test_label_rates_sane(self):
        for split in generate_all().values():
            rate = sum(r["label"] for r in split) / len(split)
            self.assertTrue(0.05 < rate < 0.45, rate)

    def test_period_independence(self):
        a = generate_period(50, "period1")
        b = generate_period(50, "period2", drift=True)
        self.assertNotEqual(a, b)


class TestModel(unittest.TestCase):
    def test_training_learns(self):
        data = generate_all()
        X, y = records_to_rows(data["train"])
        m = LogisticModel(4).fit(X, y)
        Xh, yh = records_to_rows(data["holdout"])
        self.assertGreater(auc(m.predict_proba(Xh), yh), 0.70)

    def test_fit_deterministic(self):
        data = generate_all()
        X, y = records_to_rows(data["train"])
        m1 = LogisticModel(4).fit(X, y)
        m2 = LogisticModel(4).fit(X, y)
        self.assertEqual(m1.weights, m2.weights)
        self.assertEqual(m1.coefficients_hash(), m2.coefficients_hash())

    def test_predict_in_unit_interval(self):
        data = generate_all()
        X, y = records_to_rows(data["train"])
        m = LogisticModel(4).fit(X, y)
        for p in m.predict_proba(X[:20]):
            self.assertTrue(0.0 < p < 1.0)

    def test_roundtrip_serialization(self):
        data = generate_all()
        X, y = records_to_rows(data["train"])
        m = LogisticModel(4).fit(X, y)
        m2 = LogisticModel.from_dict(m.to_dict())
        self.assertEqual(m.predict_proba(X[:10]), m2.predict_proba(X[:10]))


if __name__ == "__main__":
    unittest.main()
