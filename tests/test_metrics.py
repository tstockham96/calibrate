import os, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from calibrate import metrics as M
from calibrate.monitor import Monitor
from calibrate.providers import MockProvider
from calibrate import store


def row(p, actual, ts="2026-09-01", seg="s"):
    return {"qtype": "noul", "prob": p, "answer": "yes" if p >= 0.5 else "no", "actual": actual, "ts": ts, "segment": seg, "model": "m"}


class T(unittest.TestCase):
    def test_accept_rates(self):
        rows = [row(0.9, "no"), row(0.2, "no"), row(0.8, "yes"), row(0.3, "yes")]
        ar = M.accept_rates(rows, 0.5)
        self.assertEqual((ar["fa"], ar["neg"], ar["fr"], ar["pos"]), (1, 2, 1, 2))

    def test_ece_perfect(self):
        rows = [row(0.95, "yes")] * 19 + [row(0.95, "no")] + [row(0.05, "no")] * 19 + [row(0.05, "yes")]
        _, ece = M.reliability(rows)
        self.assertAlmostEqual(ece, 0.0, places=6)

    def test_threshold_target(self):
        rows = [row(0.6, "no")] * 10 + [row(0.9, "yes")] * 10 + [row(0.3, "no")] * 10
        rec = M.recommend_threshold(rows, max_far=0.0)
        self.assertGreater(rec["target"][0], 0.6)

    def test_monitor_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            m = Monitor(os.path.join(d, "x.db"), provider=MockProvider())
            r = m.decide({"text": "hi"}, {"q": {"type": "noul", "instructions": "?"}}, thresholds={"q": 0.5})
            m.record_outcome(r["decision_id"], "q", "yes")
            self.assertEqual(len(store.joined(m.con, "q")), 1)


if __name__ == "__main__":
    unittest.main()
