"""Metrics over joined (decision, outcome) rows. Pure stdlib."""
from __future__ import annotations

import math
from collections import defaultdict

YES = {"1", "yes", "true", "y", "t"}


def actual_yes(v) -> bool:
    return str(v).strip().lower() in YES


def is_correct(r) -> bool:
    if r["qtype"] == "noul":
        return (r["answer"] == "yes") == actual_yes(r["actual"])
    return str(r["answer"]) == str(r["actual"])


def calib_pair(r):
    """(predicted probability, observed 0/1) used for calibration.
    noul: P(yes) vs actual yes. choice/score: top-option probability vs answer correct."""
    if r["qtype"] == "noul":
        return r["prob"], 1.0 if actual_yes(r["actual"]) else 0.0
    return r["prob"], 1.0 if is_correct(r) else 0.0


def reliability(rows, bins: int = 10):
    """Returns list of (bin_mid, mean_pred, observed_rate, n) and the ECE."""
    buckets = defaultdict(list)
    for r in rows:
        p, y = calib_pair(r)
        b = min(int(p * bins), bins - 1)
        buckets[b].append((p, y))
    out, ece, n_total = [], 0.0, len(rows)
    for b in range(bins):
        pts = buckets.get(b)
        if not pts:
            continue
        mp = sum(p for p, _ in pts) / len(pts)
        oy = sum(y for _, y in pts) / len(pts)
        out.append(((b + 0.5) / bins, mp, oy, len(pts)))
        ece += len(pts) / n_total * abs(mp - oy)
    return out, (ece if n_total else float("nan"))


def accept_rates(rows, threshold: float):
    """noul only. false-accept rate = accepted negatives / all negatives (cf. 47/148 in the SaaStr post);
    false-reject rate = rejected positives / all positives; acceptance = share accepted."""
    neg = [r for r in rows if not actual_yes(r["actual"])]
    pos = [r for r in rows if actual_yes(r["actual"])]
    fa = sum(1 for r in neg if r["prob"] >= threshold)
    fr = sum(1 for r in pos if r["prob"] < threshold)
    acc = sum(1 for r in rows if r["prob"] >= threshold)
    return {
        "far": fa / len(neg) if neg else float("nan"),
        "frr": fr / len(pos) if pos else float("nan"),
        "acceptance": acc / len(rows) if rows else float("nan"),
        "fa": fa, "neg": len(neg), "fr": fr, "pos": len(pos), "n": len(rows),
    }


def accuracy(rows) -> float:
    return sum(is_correct(r) for r in rows) / len(rows) if rows else float("nan")


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def daily(rows, fn):
    by = defaultdict(list)
    for r in rows:
        by[r["ts"][:10]].append(r)
    return [(d, fn(by[d]), len(by[d])) for d in sorted(by)]


def psi(base_probs, recent_probs, bins: int = 10) -> float:
    """Population stability index on the probability distribution. Needs no labels."""
    def hist(xs):
        h = [0] * bins
        for x in xs:
            h[min(int(x * bins), bins - 1)] += 1
        n = max(len(xs), 1)
        return [max(c / n, 1e-4) for c in h]
    b, r = hist(base_probs), hist(recent_probs)
    return sum((ri - bi) * math.log(ri / bi) for bi, ri in zip(b, r))


def recommend_threshold(rows, max_far: float | None = 0.05, cost_fa: float = 5.0, cost_fr: float = 1.0):
    """noul only. Two recommendations on the given (recent) labelled rows:
    - 'target': lowest threshold whose false-accept rate <= max_far (maximises automation within the bar)
    - 'cost':   threshold minimising cost_fa*FA + cost_fr*FR
    Both are point estimates; the dashboard shows n so users can judge reliability."""
    grid = [i / 100 for i in range(5, 100)]
    target = None
    for t in grid:
        ar = accept_rates(rows, t)
        if ar["neg"] and ar["far"] <= max_far:
            target = (t, ar)
            break
    best = None
    for t in grid:
        ar = accept_rates(rows, t)
        cost = cost_fa * ar["fa"] + cost_fr * ar["fr"]
        if best is None or cost < best[2]:
            best = (t, ar, cost)
    return {"target": target, "cost": best[:2] if best else None,
            "max_far": max_far, "cost_fa": cost_fa, "cost_fr": cost_fr}


def split_windows(rows, recent_days: int = 7):
    days = sorted({r["ts"][:10] for r in rows})
    if len(days) <= recent_days:
        return rows, []
    cut = days[-recent_days]
    return [r for r in rows if r["ts"][:10] < cut], [r for r in rows if r["ts"][:10] >= cut]


def drift_report(rows, threshold: float, recent_days: int = 7, min_n: int = 40,
                 acc_drop_pp: float = 5.0, far_rise_pp: float = 5.0, ece_rise: float = 0.05):
    """Compare recent window vs baseline overall, per segment and per model version. Returns alerts."""
    base, recent = split_windows(rows, recent_days)
    groups = {"overall": lambda r: "all", "segment": lambda r: r.get("segment") or "-",
              "model": lambda r: r.get("model") or "-"}
    results, alerts = [], []
    for gname, keyf in groups.items():
        keys = sorted({keyf(r) for r in recent})
        for k in keys:
            b = [r for r in base if keyf(r) == k]
            rc = [r for r in recent if keyf(r) == k]
            if gname == "model" and not b:
                b = base  # new model version: compare to whole baseline
            if len(rc) < min_n or len(b) < min_n:
                continue
            row = {"group": gname, "key": k, "n_base": len(b), "n_recent": len(rc),
                   "acc_base": accuracy(b), "acc_recent": accuracy(rc),
                   "ece_base": reliability(b)[1], "ece_recent": reliability(rc)[1],
                   "psi": psi([r["prob"] for r in b], [r["prob"] for r in rc])}
            if rc and rc[0]["qtype"] == "noul":
                row["far_base"] = accept_rates(b, threshold)["far"]
                row["far_recent"] = accept_rates(rc, threshold)["far"]
            results.append(row)
            msgs = []
            if (row["acc_base"] - row["acc_recent"]) * 100 >= acc_drop_pp:
                msgs.append(f"accuracy {row['acc_base']:.1%} -> {row['acc_recent']:.1%}")
            if "far_base" in row and (row["far_recent"] - row["far_base"]) * 100 >= far_rise_pp:
                msgs.append(f"false-accept {row['far_base']:.1%} -> {row['far_recent']:.1%}")
            if row["ece_recent"] - row["ece_base"] >= ece_rise:
                msgs.append(f"ECE {row['ece_base']:.3f} -> {row['ece_recent']:.3f}")
            if msgs:
                alerts.append({"scope": f"{gname}={k}", "detail": "; ".join(msgs), "n": len(rc)})
    return {"results": results, "alerts": alerts, "n_base": len(base), "n_recent": len(recent)}
