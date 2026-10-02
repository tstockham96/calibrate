"""Synthetic demo scenario. EVERYTHING HERE IS FAKE DATA, generated to exercise the dashboard.

Scenario (invented): a support tool asks a decision model "Is this refund request eligible under the
written policy?" (noul) and "Which team should handle this?" (choice) for ~110 tickets/day over 28 days.
- Day 19: the (mock) provider version changes from mock-decider-v1 to mock-decider-v2.
- Day 22 onward, chat tickets start carrying customer-written arguments ("I'm sure this qualifies...")
  that push the mock toward 'yes' - modelled loosely on the failure mode JevAdvBench reports
  (https://arxiv.org/abs/2609.31142). The numbers below are NOT measurements of any real model.
- Ground truth arrives with a 2-day lag for ~85% of tickets (CSV export from the "refund system").
"""
from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta, timezone

from .monitor import Monitor
from .providers import MockProvider

QUESTIONS = {
    "refund_eligible": {
        "type": "noul",
        "instructions": "Is this refund request eligible under the written refund policy in `policy`?",
        "criteria": {"true": "Meets every condition in the policy", "false": "Fails at least one condition"},
    },
    "route": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages, integrations",
                     "sales": "Pricing, upgrades, new accounts"},
    },
}

END = datetime(2026, 10, 1, 18, 0, tzinfo=timezone.utc)
DAYS = 28


def generate(db_path: str, outcomes_csv: str, seed: int = 7, per_day: int = 110, with_shadow: bool = True):
    rng = random.Random(seed)
    primary = MockProvider(model_name="mock-decider-v1")
    shadow = [MockProvider(model_name="mock-alt-provider", skill=2.0, overconfidence=0.9,
                           use_sim_profile=False, name="mock-alt")] if with_shadow else []
    mon = Monitor(db_path, provider=primary, shadow_providers=shadow)
    start = END - timedelta(days=DAYS - 1)
    outcome_rows = []
    for d in range(DAYS):
        day = start + timedelta(days=d)
        n = per_day + rng.randint(-15, 15)
        for i in range(n):
            ts = (day.replace(hour=0) + timedelta(seconds=rng.randint(0, 86399))).isoformat()
            channel = "chat" if rng.random() < 0.45 else "email"
            eligible = rng.random() < 0.38
            route = rng.choices(["billing", "technical", "sales"], [0.55, 0.3, 0.15])[0]
            version = "mock-decider-v1" if d < 18 else "mock-decider-v2"
            profile = {"model": version, "skill": 2.5 if d < 18 else 2.35, "bias": -0.2, "overconfidence": 1.0}
            if channel == "chat" and d >= 21:
                profile.update(bias=1.25, overconfidence=1.25)  # argued input pushes toward yes
            state = {"ticket_id": f"SYN-{d:02d}-{i:04d}", "channel": channel,
                     "text": "[synthetic ticket text]", "policy": "[synthetic policy]",
                     "_sim": {**profile, "truth": {"refund_eligible": eligible, "route": route}}}
            res = mon.decide(state, QUESTIONS, model="jev-1.13.0-pinned (mock)", segment=f"channel={channel}",
                             thresholds={"refund_eligible": 0.5}, ts=ts)
            # ground truth lags 2 days and covers ~85% of tickets
            if d <= DAYS - 3 and rng.random() < 0.85:
                outcome_rows.append((res["decision_id"], "refund_eligible", "yes" if eligible else "no"))
                outcome_rows.append((res["decision_id"], "route", route))
    with open(outcomes_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["decision_id", "question_key", "actual"])
        w.writerows(outcome_rows)
    return len(outcome_rows)
