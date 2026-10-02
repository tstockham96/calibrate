"""calibrate CLI.

  calibrate demo [--out demo_out]                 synthetic end-to-end run (mock provider)
  calibrate decide --db X --state '{"..."}' --questions q.json [--model jev-1.13.0] [--segment s]
                                                  one real/mocked call, logged (real if TYPESAFE_API_KEY set)
  calibrate ingest --db X outcomes.csv            CSV: decision_id,question_key,actual[,observed_at]
  calibrate report --db X --question KEY --out dashboard.html [--target-far 0.05]
  calibrate alerts --db X --question KEY          print drift alerts (exit 2 if any: cron/CI friendly)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import dashboard, metrics, store
from .monitor import Monitor
from .providers import get_default_provider


def cmd_demo(a):
    from . import synth
    os.makedirs(a.out, exist_ok=True)
    db = os.path.join(a.out, "calibrate_demo.db")
    csvp = os.path.join(a.out, "outcomes_synthetic.csv")
    html = os.path.join(a.out, "dashboard.html")
    if os.path.exists(db):
        os.remove(db)
    print("[demo] generating SYNTHETIC decisions with the mock provider ...")
    n = synth.generate(db, csvp, seed=a.seed)
    con = store.connect(db)
    k = store.ingest_outcomes_csv(con, csvp)
    print(f"[demo] wrote {n} synthetic outcome rows to {csvp}; ingested {k}")
    res = dashboard.render(db, html, question="refund_eligible", target_far=a.target_far)
    print(f"[demo] dashboard: {html}")
    for al in res["alerts"]:
        print(f"[alert] {al['scope']}: {al['detail']} (n={al['n']})")


def cmd_decide(a):
    provider = get_default_provider()
    mon = Monitor(a.db, provider=provider)
    state = json.loads(a.state)
    with open(a.questions) as f:
        questions = json.load(f)
    thresholds = json.loads(a.thresholds) if a.thresholds else None
    res = mon.decide(state, questions, model=a.model, segment=a.segment, thresholds=thresholds)
    print(f"# provider={provider.name}" + (" (MOCK - set TYPESAFE_API_KEY for real calls)" if provider.name == "mock" else ""))
    res.pop("_mock", None)
    print(json.dumps(res, indent=2))


def cmd_ingest(a):
    con = store.connect(a.db)
    print(f"ingested {store.ingest_outcomes_csv(con, a.csv)} outcome rows")


def cmd_report(a):
    res = dashboard.render(a.db, a.out, question=a.question, target_far=a.target_far,
                           threshold=a.threshold, synthetic=a.synthetic)
    print(f"dashboard: {res['out']}")


def cmd_alerts(a):
    con = store.connect(a.db)
    rows = store.joined(con, a.question)
    thr = a.threshold if a.threshold is not None else ((rows[-1]["threshold"] if rows else None) or 0.5)
    rep = metrics.drift_report(rows, thr)
    for al in rep["alerts"]:
        print(f"ALERT {al['scope']}: {al['detail']} (n={al['n']})")
    if not rep["alerts"]:
        print("no alerts")
    sys.exit(2 if rep["alerts"] else 0)


def main(argv=None):
    p = argparse.ArgumentParser(prog="calibrate", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo"); d.add_argument("--out", default="demo_out"); d.add_argument("--seed", type=int, default=7)
    d.add_argument("--target-far", type=float, default=0.05); d.set_defaults(fn=cmd_demo)
    c = sub.add_parser("decide"); c.add_argument("--db", default="calibrate.db"); c.add_argument("--state", required=True)
    c.add_argument("--questions", required=True); c.add_argument("--model", default="jev-latest")
    c.add_argument("--segment"); c.add_argument("--thresholds", help='JSON, e.g. {"q": 0.7}'); c.set_defaults(fn=cmd_decide)
    i = sub.add_parser("ingest"); i.add_argument("--db", default="calibrate.db"); i.add_argument("csv"); i.set_defaults(fn=cmd_ingest)
    r = sub.add_parser("report"); r.add_argument("--db", default="calibrate.db"); r.add_argument("--question", required=True)
    r.add_argument("--out", default="dashboard.html"); r.add_argument("--target-far", type=float, default=0.05)
    r.add_argument("--threshold", type=float); r.add_argument("--synthetic", action="store_true",
                                                              help="show the SYNTHETIC DATA banner"); r.set_defaults(fn=cmd_report)
    al = sub.add_parser("alerts"); al.add_argument("--db", default="calibrate.db"); al.add_argument("--question", required=True)
    al.add_argument("--threshold", type=float); al.set_defaults(fn=cmd_alerts)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
