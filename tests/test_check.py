import contextlib
import io
import os
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from calibrate import check as C  # noqa: E402
from calibrate import cli  # noqa: E402

SAMPLE = os.path.join(ROOT, "examples", "sample_decisions.csv")


def write(d, name, text, encoding="utf-8"):
    p = os.path.join(d, name)
    with open(p, "w", encoding=encoding, newline="") as f:
        f.write(text)
    return p


def R(p, y, thr=0.5, accept=None, seg="-", model="-"):
    acc = p >= thr if accept is None else accept
    return C.Row(prob=p, y=y, accept=acc, correct=(acc == bool(y)), threshold=thr, segment=seg, model_version=model)


class ParseAliases(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def test_aliases_case_insensitive_and_boolean_spellings(self):
        p = write(self.d, "a.csv", "Score,ACTUAL,Model,Group,Date\n"
                                   "0.9,yes,v1,email,2026-09-01\n0.2,No,v1,chat,2026-09-02\n"
                                   "0.7,TRUE,v2,chat,2026-09-03\n0.4,0,v2,email,2026-09-04\n0.6,false,v2,email,\n")
        rows, info = C.parse_csv(p)
        self.assertEqual(info.mode, "binary")
        self.assertEqual(info.columns["probability"], "Score")
        self.assertEqual(info.columns["outcome"], "ACTUAL")
        self.assertEqual(info.columns["model_version"], "Model")
        self.assertEqual(info.columns["segment"], "Group")
        self.assertEqual(info.columns["timestamp"], "Date")
        self.assertEqual([r.y for r in rows], [1, 0, 1, 0, 0])
        # no decision column -> derived from probability >= threshold (default 0.5)
        self.assertEqual([r.accept for r in rows], [True, False, True, False, True])
        self.assertEqual(rows[2].model_version, "v2")
        self.assertEqual(rows[1].segment, "chat")
        self.assertIsNotNone(rows[0].ts)
        self.assertIsNone(rows[4].ts)

    def test_confidence_label_threshold_column_and_decision_column(self):
        p = write(self.d, "b.csv", "Confidence,Label,Threshold,Decision\n"
                                   "0.8,1,0.9,\n0.95,1,0.9,\n0.3,0,,yes\n")
        rows, info = C.parse_csv(p, default_threshold=0.25)
        self.assertEqual([r.threshold for r in rows], [0.9, 0.9, 0.25])
        # row 1: no decision, 0.8 < 0.9 -> reject; row 3: explicit decision 'yes' wins over probability
        self.assertEqual([r.accept for r in rows], [False, True, True])
        self.assertEqual([r.correct for r in rows], [False, True, False])

    def test_semicolon_bom_percent_comments_and_skips(self):
        text = "# SYNTHETIC test file\n\nprobability;outcome\n85%;y\n0.1;n\n1.5;yes\n0.4;\n;\n"
        p = write(self.d, "c.csv", text, encoding="utf-8-sig")
        rows, info = C.parse_csv(p)
        self.assertEqual([(r.prob, r.y) for r in rows], [(0.85, 1), (0.1, 0)])
        self.assertEqual(info.comments, ["SYNTHETIC test file"])
        self.assertEqual(info.skipped["probability missing or not in [0, 1]"], 1)
        self.assertEqual(info.skipped["no outcome yet (unlabelled)"], 1)

    def test_missing_required_column(self):
        p = write(self.d, "d.csv", "prob,decision\n0.4,yes\n")
        with self.assertRaises(C.CheckError) as cm:
            C.parse_csv(p)
        self.assertIn("outcome", str(cm.exception))

    def test_choice_rows(self):
        p = write(self.d, "e.csv", "confidence,decision,actual\n0.9,billing,Billing\n0.6,sales,technical\n"
                                   "0.8,technical,technical\n0.4,sales,sales\n")
        rows, info = C.parse_csv(p, default_threshold=0.5)
        self.assertEqual(info.mode, "choice")
        self.assertEqual([r.correct for r in rows], [True, False, True, True])
        self.assertEqual([r.y for r in rows], [1, 0, 1, 1])
        self.assertEqual([r.accept for r in rows], [True, True, True, False])  # confidence gate
        ar = C.rates(rows)
        self.assertEqual((ar["fa"], ar["neg"], ar["fr"], ar["pos"]), (1, 1, 1, 3))
        self.assertAlmostEqual(C.accuracy(rows), 0.75)

    def test_categorical_outcome_without_decision_is_an_error(self):
        p = write(self.d, "f.csv", "probability,outcome\n0.7,billing\n")
        with self.assertRaises(C.CheckError):
            C.parse_csv(p)

    def test_timestamp_formats(self):
        for s in ["2026-09-30T14:05:00Z", "2026-09-30 14:05:00", "2026-09-30", "09/30/2026",
                  "2026-09-30T08:05:00-06:00", "1790777100", "1790777100000"]:
            self.assertIsNotNone(C.parse_ts(s), s)
        self.assertEqual(C.parse_ts("2026-09-30T08:05:00-06:00").hour, 14)
        self.assertIsNone(C.parse_ts("yesterday"))

    def test_over_time_bucketing(self):
        from datetime import datetime, timedelta
        base = datetime(2026, 9, 1)
        rows = []
        for d in range(28):
            for k in range(10):
                r = R(0.9, 1 if k else 0)
                r.ts = base + timedelta(days=d, hours=k)
                rows.append(r)
        tl, unit = C.over_time(rows)
        self.assertEqual(unit, "week")  # 10/day is too thin for daily points
        self.assertEqual(sum(n for *_, n in tl), 280)
        tl, unit = C.over_time(rows, min_median_n=5)
        self.assertEqual((unit, len(tl)), ("day", 28))
        self.assertAlmostEqual(tl[0][2], 0.9)


class Metrics(unittest.TestCase):
    def test_ece_perfect_is_zero(self):
        rows = [R(0.95, 1)] * 19 + [R(0.95, 0)] + [R(0.05, 0)] * 19 + [R(0.05, 1)]
        bins, ece = C.ece_bins(rows)
        self.assertAlmostEqual(ece, 0.0, places=9)
        self.assertEqual(len(bins), 10)
        self.assertEqual([b["n"] for b in bins], [20, 0, 0, 0, 0, 0, 0, 0, 0, 20])

    def test_ece_hand_computed(self):
        # bin 0.8-0.9: 4 rows at 0.8, 2 positive -> gap 0.3; bin 0.2-0.3: 4 rows at 0.2, 0 positive -> gap 0.2
        rows = [R(0.8, 1), R(0.8, 1), R(0.8, 0), R(0.8, 0), R(0.2, 0), R(0.2, 0), R(0.2, 0), R(0.2, 0)]
        bins, ece = C.ece_bins(rows)
        self.assertAlmostEqual(ece, 0.5 * 0.3 + 0.5 * 0.2)
        self.assertEqual(bins[8]["n"], 4)
        self.assertAlmostEqual(bins[8]["observed"], 0.5)
        self.assertEqual(C.ece_bins([R(1.0, 1)])[0][9]["n"], 1)  # p=1.0 lands in the last bin

    def test_brier_and_auroc(self):
        rows = [R(0.9, 1), R(0.2, 0)]
        self.assertAlmostEqual(C.brier(rows), (0.01 + 0.04) / 2)
        self.assertAlmostEqual(C.auroc([R(0.9, 1), R(0.8, 1), R(0.3, 0), R(0.1, 0)]), 1.0)
        self.assertAlmostEqual(C.auroc([R(0.1, 1), R(0.9, 0)]), 0.0)
        self.assertAlmostEqual(C.auroc([R(0.5, 1), R(0.5, 0)]), 0.5)
        self.assertAlmostEqual(C.auroc([R(0.9, 1), R(0.4, 1), R(0.6, 0), R(0.1, 0)]), 0.75)
        self.assertTrue(C.isnan(C.auroc([R(0.9, 1)])))


class Threshold(unittest.TestCase):
    def test_recommendation_meets_target_with_lowest_threshold(self):
        rows = [R(0.6, 0)] * 10 + [R(0.9, 1)] * 10 + [R(0.3, 0)] * 10 + [R(0.55, 1)] * 5
        rec = C.recommend_threshold(rows, target_far=0.0)
        self.assertAlmostEqual(rec["threshold"], 0.61)
        self.assertEqual(rec["fa"], 0)
        self.assertEqual(rec["fr"], 5)  # the cost: the 0.55 positives are now rejected
        loose = C.recommend_threshold(rows, target_far=0.5)
        self.assertAlmostEqual(loose["threshold"], 0.31)  # 10/20 negatives accepted = 50%

    def test_unreachable_and_no_negatives(self):
        rows = [R(1.0, 0)] * 3 + [R(1.0, 1)] * 3
        self.assertIsNone(C.recommend_threshold(rows, target_far=0.05))
        self.assertIsNone(C.recommend_threshold([R(0.7, 1)] * 5, target_far=0.05))

    def test_current_rates_use_recorded_decisions(self):
        rows = [R(0.9, 0, accept=False), R(0.2, 0, accept=True), R(0.8, 1), R(0.1, 1)]
        self.assertEqual(C.rates(rows)["fa"], 1)
        self.assertEqual(C.rates(rows, 0.5)["fa"], 1)
        self.assertEqual(C.rates(rows, 0.95)["fr"], 2)


class CLIEndToEnd(unittest.TestCase):
    def test_sample_report(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "sub", "report.html")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                cli.main(["check", SAMPLE, "--threshold", "0.5", "--target-far", "0.05", "--out", out])
            txt = buf.getvalue()
            self.assertIn("SYNTHETIC", txt)
            self.assertIn("recommended", txt)
            self.assertIn("by segment", txt)
            self.assertIn("small n", txt)
            with open(out, encoding="utf-8") as f:
                doc = f.read()
        for needle in ["Verdict", "Reliability bins", "Brier", "AUROC", "False-accept", "False-reject",
                       "By segment", "By model version", "Accuracy over time", "Threshold recommendation",
                       "small n", "SYNTHETIC DATA",
                       'Generated locally by Calibrate (open source). Hosted monitoring: <a href="https://calibrate.tools">https://calibrate.tools</a>']:
            self.assertIn(needle, doc)
        # self-contained: no scripts, stylesheets, images or fonts pulled from anywhere
        self.assertNotIn("<script", doc.lower())
        self.assertNotIn("<link", doc.lower())
        self.assertNotIn("<img", doc.lower())
        self.assertIsNone(re.search(r"(src|href)=\"(?!https://calibrate\.tools)", doc))
        self.assertNotIn("@import", doc)

    def test_subprocess_choice_and_errors(self):
        with tempfile.TemporaryDirectory() as d:
            p = write(d, "choice.csv", "timestamp,confidence,decision,outcome\n" + "".join(
                f"2026-09-{1 + i % 20:02d},{0.5 + (i % 5) / 10:.2f},{'a' if i % 3 else 'b'},{'a' if i % 4 else 'b'}\n"
                for i in range(200)))
            out = os.path.join(d, "r.html")
            env = {**os.environ, "PYTHONPATH": ROOT}
            env.pop("TYPESAFE_API_KEY", None)
            r = subprocess.run([sys.executable, "-m", "calibrate", "check", p, "--out", out],
                               capture_output=True, text=True, env=env, cwd=d)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("choice decisions", r.stdout)
            self.assertTrue(os.path.exists(out))
            bad = write(d, "bad.csv", "foo,bar\n1,2\n")
            r = subprocess.run([sys.executable, "-m", "calibrate", "check", bad, "--out", out],
                               capture_output=True, text=True, env=env, cwd=d)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("Missing required column", r.stderr)
            r = subprocess.run([sys.executable, "-m", "calibrate", "check", os.path.join(d, "nope.csv")],
                               capture_output=True, text=True, env=env, cwd=d)
            self.assertIn("file not found", r.stderr)


if __name__ == "__main__":
    unittest.main()
