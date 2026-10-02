# calibrate (prototype)

## Free calibration check on your own data

Export your AI decisions to a CSV and get a one-time calibration report. No API key, no account, no network calls:
it runs locally with the Python standard library, and **nothing leaves your machine**.

**Columns** (headers are case-insensitive; aliases in brackets):

| column | required | notes |
|---|---|---|
| `probability` [`score`, `confidence`] | yes | 0–1 (or `85%`). P(yes), or the probability of the chosen option for multi-choice decisions |
| `outcome` [`actual`, `label`] | yes | ground truth: `1/0`, `true/false`, `yes/no`, or the correct category |
| `decision` | no | what the model decided (yes/no or a category). If absent: `probability >= threshold` |
| `threshold` | no | threshold in force per row (otherwise `--threshold`, default 0.5) |
| `model_version`, `segment`, `timestamp` | no | breakdowns and accuracy over time |

**One command:**

```bash
python3 -m calibrate check decisions.csv --target-far 0.05 --out report.html
# or, after `pip install -e .`:   calibrate check decisions.csv [--threshold 0.5] [--target-far 0.05] [--out report.html]
```

You get a terminal summary and a self-contained `report.html` (no external assets): a plain-language verdict, accuracy,
ECE with 10 reliability bins (counts shown), Brier score, AUROC, false-accept / false-reject at your current threshold,
the same per segment and model version (small samples flagged), accuracy over time, and the lowest threshold that meets
your target false-accept rate together with what it costs in false rejects. Try it on the synthetic sample:
`python3 -m calibrate check examples/sample_decisions.csv` (invented data, labelled as such in the file and the report).

## Continuous monitoring (prototype)

![Calibrate dashboard (synthetic demo data)](docs/dashboard.png)

Production monitoring for typed AI decision models (Jev-style `noul` / `choice` / `score`). Pure Python stdlib, Python ≥ 3.10.

```bash
git clone https://github.com/tstockham96/calibrate && cd calibrate
pip install -e .            # optional; or use `python3 -m calibrate ...`
calibrate demo --out demo_out          # SYNTHETIC end-to-end run -> demo_out/dashboard.html
python3 -m unittest discover -s tests  # tests
```

Real usage:
```bash
export TYPESAFE_API_KEY=...            # absent -> mock provider (clearly reported)
calibrate decide --db app.db --state '{"text":"Help! My payouts have been failing for 3 days."}' \
   --questions examples_questions.json --model jev-1.13.0 --segment channel=email --thresholds '{"is_urgent":0.7}'
calibrate ingest --db app.db outcomes.csv        # decision_id,question_key,actual[,observed_at]
calibrate report --db app.db --question is_urgent --out dashboard.html --target-far 0.05
calibrate alerts --db app.db --question is_urgent   # exit 2 when alerts fire (cron/CI)
```

Python:
```python
from calibrate import Monitor, MockProvider
mon = Monitor("app.db", shadow_providers=[MockProvider(name="mock-alt", use_sim_profile=False)])
res = mon.decide(state, questions, model="jev-1.13.0", segment="tier=pro", thresholds={"q": 0.7})
mon.record_outcome(res["decision_id"], "q", "yes")
```

What's logged per answer: decision_id, ts, provider, role (primary/shadow), resolved model, question hash, **state hash (raw state is not stored)**,
type, answer, probability, confidence, threshold in force, segment, latency, input tokens.

Metrics: accuracy (Wilson CI), reliability curve + ECE (10 bins), false-accept / false-reject at threshold, PSI (label-free), drift
(last 7 days vs prior, overall / per segment / per model version), threshold recommendation (lowest threshold with FAR ≤ target, and
cost-weighted), shadow side-by-side.

Real provider: `TypeSafeProvider` POSTs the documented shape to `https://api.typesafe.ai/v1/systemone` with backoff on 429/529
(https://docs.typesafe.ai/api.md). It has NOT been exercised against the live API here (no key). `OpenAIDecisionsProvider` is a
deliberate stub until the schema is public.

Synthetic demo: `calibrate/synth.py` invents a refund-bot scenario (a version change on day 19, argued chat tickets from day 22). None of it is a measurement of any real model.
