"""`calibrate check`: a one-time calibration report on decisions you exported yourself.

Reads a CSV, computes everything locally (pure stdlib, no network, no API key) and writes a
self-contained HTML report (inline CSS + SVG, no scripts, no external assets).

Required columns (header names are case-insensitive; aliases accepted):
  probability   probability | prob | p | score | confidence | p_yes | predicted_probability
  outcome       outcome | actual | label | ground_truth | truth | target | y
Optional:
  decision      decision | prediction | predicted | answer | choice
  threshold     threshold | cutoff | thr
  model_version model_version | model | version | model_name
  segment       segment | group | cohort | slice
  timestamp     timestamp | ts | time | date | datetime | created_at | decided_at

Two kinds of data:
  yes/no   outcome is 1/0, true/false, yes/no (also accept/reject, approved/denied ...).
           probability = P(yes). decision = predicted yes/no; if absent, probability >= threshold.
           "accepted" = decision yes. false-accept rate = accepted negatives / all negatives.
  choice   decision and outcome are category strings ("billing", "technical", ...).
           probability = probability of the top (chosen) option; correct = decision == outcome.
           "accepted" = confidence >= threshold (auto-accepted instead of sent to review), so the
           false-accept rate = wrong answers auto-accepted / all wrong answers.
"""
from __future__ import annotations

import csv
import html
import io
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .metrics import wilson

BINS = 10
SMALL_N = 100          # groups below this are flagged "small n"
TINY_N = 30            # groups below this get no threshold recommendation
FOOTER_URL = "https://calibrate.tools"

ALIASES = {
    "probability": ["probability", "prob", "p", "score", "confidence", "p_yes", "prob_yes",
                    "predicted_probability", "pred_prob", "probability_yes"],
    "outcome": ["outcome", "actual", "label", "ground_truth", "truth", "target", "y", "true_label", "actual_label"],
    "decision": ["decision", "prediction", "predicted", "answer", "choice", "pred", "predicted_label"],
    "threshold": ["threshold", "cutoff", "thr"],
    "model_version": ["model_version", "model", "version", "model_name", "modelversion"],
    "segment": ["segment", "group", "cohort", "slice"],
    "timestamp": ["timestamp", "ts", "time", "date", "datetime", "created_at", "decided_at", "decision_time"],
}
POSITIVE = {"1", "1.0", "yes", "y", "true", "t", "accept", "accepted", "approve", "approved",
            "pass", "passed", "positive", "pos"}
NEGATIVE = {"0", "0.0", "no", "n", "false", "f", "reject", "rejected", "deny", "denied", "decline",
            "declined", "fail", "failed", "negative", "neg"}
BOOLEAN = POSITIVE | NEGATIVE


class CheckError(ValueError):
    """Input problem the user can fix (bad header, no usable rows, ...)."""


@dataclass
class Row:
    prob: float           # P(yes) for yes/no data; top-choice probability for choice data
    y: int                # yes/no: outcome is yes; choice: decision was correct
    accept: bool          # yes/no: decided yes; choice: confidence >= threshold (auto-accepted)
    correct: bool
    threshold: float
    segment: str = "-"
    model_version: str = "-"
    ts: datetime | None = None


@dataclass
class ParseInfo:
    path: str
    mode: str                                   # "binary" | "choice"
    columns: dict                               # canonical -> header actually used
    n_read: int = 0
    skipped: Counter = field(default_factory=Counter)
    notes: list = field(default_factory=list)
    comments: list = field(default_factory=list)
    threshold_source: str = "--threshold"
    decision_source: str = "derived"
    bad_timestamps: int = 0


def _norm_header(h: str) -> str:
    h = (h or "").strip().strip('"').strip().lower()
    for ch in " -./()":
        h = h.replace(ch, "_")
    while "__" in h:
        h = h.replace("__", "_")
    return h.strip("_")


def map_columns(headers):
    """Map canonical column names to the actual header strings. Exact alias match wins over order."""
    normed = {_norm_header(h): h for h in headers if h is not None}
    found = {}
    for canon, aliases in ALIASES.items():
        for a in aliases:
            if a in normed and normed[a] not in found.values():
                found[canon] = normed[a]
                break
    missing = [c for c in ("probability", "outcome") if c not in found]
    if missing:
        hint = {"probability": "probability (or score / confidence)",
                "outcome": "outcome (or actual / label)"}
        raise CheckError("Missing required column(s): " + ", ".join(hint[m] for m in missing)
                         + f". Found headers: {', '.join(h for h in headers if h)}")
    return found


def parse_prob(v: str) -> float | None:
    s = (v or "").strip()
    if not s:
        return None
    pct = s.endswith("%")
    try:
        x = float(s.rstrip("%").strip())
    except ValueError:
        return None
    if pct:
        x /= 100.0
    if math.isnan(x) or x < 0 or x > 1:
        return None
    return x


def parse_bool(v: str) -> bool | None:
    s = (v or "").strip().lower()
    if s in POSITIVE:
        return True
    if s in NEGATIVE:
        return False
    return None


_TS_FORMATS = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d",
               "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%Y", "%d.%m.%Y", "%d.%m.%Y %H:%M"]


def parse_ts(v: str) -> datetime | None:
    s = (v or "").strip()
    if not s:
        return None
    try:  # epoch seconds / milliseconds
        x = float(s)
        if x > 1e11:
            x /= 1000.0
        if 1e8 < x < 1e10:
            return datetime.fromtimestamp(x, tz=timezone.utc).replace(tzinfo=None)
    except ValueError:
        pass
    iso = s[:-1] + "+00:00" if s.endswith(("Z", "z")) else s
    try:
        d = datetime.fromisoformat(iso)
        return d.astimezone(timezone.utc).replace(tzinfo=None) if d.tzinfo else d
    except ValueError:
        pass
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _read_text(path: str):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        text = f.read()
    lines = text.splitlines(keepends=True)
    comments = []
    while lines and (lines[0].lstrip().startswith("#") or not lines[0].strip()):
        c = lines.pop(0).strip()
        if c:
            comments.append(c.lstrip("#").strip())
    body = "".join(lines)
    first = body.splitlines()[0] if body.strip() else ""
    delim = max([",", ";", "\t", "|"], key=first.count) if first else ","
    if first.count(delim) == 0:
        delim = ","
    return body, delim, comments


def parse_csv(path: str, default_threshold: float = 0.5, threshold_explicit: bool = False):
    """Parse an exported decisions CSV into Rows. Returns (rows, ParseInfo)."""
    body, delim, comments = _read_text(path)
    reader = csv.DictReader(io.StringIO(body), delimiter=delim)
    if not reader.fieldnames:
        raise CheckError(f"{path}: no header row found")
    cols = map_columns(reader.fieldnames)
    raw = list(reader)
    get = lambda r, c: (r.get(cols[c]) or "").strip() if c in cols else ""

    outcomes = [get(r, "outcome") for r in raw if get(r, "outcome")]
    decisions = [get(r, "decision") for r in raw if get(r, "decision")]
    out_bool = bool(outcomes) and all(o.lower() in BOOLEAN for o in outcomes)
    dec_bool = all(d.lower() in BOOLEAN for d in decisions)
    if out_bool and dec_bool:
        mode, outcome_is_correctness = "binary", False
    elif decisions and not out_bool:
        mode, outcome_is_correctness = "choice", False
    elif decisions and out_bool and not dec_bool:
        mode, outcome_is_correctness = "choice", True
    else:
        bad = sorted({o for o in outcomes if o.lower() not in BOOLEAN})[:5]
        raise CheckError("The outcome column has category values (e.g. " + ", ".join(repr(b) for b in bad)
                         + ") but there is no decision column with the model's chosen option. Add a "
                         "'decision' column, or use 1/0, true/false, yes/no outcomes for yes/no decisions.")

    info = ParseInfo(path=path, mode=mode, columns=cols, n_read=len(raw), comments=comments)
    if "threshold" in cols:
        info.threshold_source = f"'{cols['threshold']}' column (falls back to --threshold {default_threshold:g})"
        if threshold_explicit:
            info.notes.append(f"--threshold {default_threshold:g} is only used for rows with an empty "
                              f"'{cols['threshold']}' value; the CSV's own thresholds take precedence.")
    else:
        info.threshold_source = f"--threshold {default_threshold:g}"
    if mode == "binary":
        info.decision_source = (f"'{cols['decision']}' column (rows without one: probability >= threshold)"
                                if "decision" in cols else "derived: probability >= threshold")
    else:
        info.decision_source = f"'{cols['decision']}' column (the chosen option)"
        info.notes.append("Choice data: calibration, Brier and AUROC use the probability of the chosen option "
                          "versus whether it was correct. 'Accepted' means confidence >= threshold "
                          "(auto-accepted rather than sent to review).")
        if outcome_is_correctness:
            info.notes.append(f"The '{cols['outcome']}' column holds yes/no values while decisions are categories, "
                              "so it was read as 'the decision was correct'.")

    rows = []
    for r in raw:
        p = parse_prob(get(r, "probability"))
        o = get(r, "outcome")
        if not get(r, "probability") and not o:
            info.skipped["empty row"] += 1
            continue
        if p is None:
            info.skipped["probability missing or not in [0, 1]"] += 1
            continue
        if not o:
            info.skipped["no outcome yet (unlabelled)"] += 1
            continue
        t = parse_prob(get(r, "threshold")) if "threshold" in cols else None
        thr = t if t is not None else default_threshold
        d = get(r, "decision")
        if mode == "binary":
            y = parse_bool(o)
            dec = parse_bool(d) if d else None
            accept = dec if dec is not None else p >= thr
            correct = accept == y
        else:
            if outcome_is_correctness:
                correct = bool(parse_bool(o))
            else:
                if not d:
                    info.skipped["no decision (choice data)"] += 1
                    continue
                correct = d.strip().lower() == o.strip().lower()
            y = correct
            accept = p >= thr
        ts = None
        if "timestamp" in cols:
            ts = parse_ts(get(r, "timestamp"))
            if ts is None and get(r, "timestamp"):
                info.bad_timestamps += 1
        rows.append(Row(prob=p, y=int(bool(y)), accept=bool(accept), correct=bool(correct), threshold=thr,
                        segment=get(r, "segment") or "-", model_version=get(r, "model_version") or "-", ts=ts))
    if info.bad_timestamps:
        info.notes.append(f"{info.bad_timestamps} timestamp value(s) could not be parsed and were left out of the "
                          "over-time chart (use ISO 8601, e.g. 2026-09-30T14:05:00Z).")
    if not rows:
        raise CheckError(f"{path}: no usable rows (" + ", ".join(f"{k}: {v}" for k, v in info.skipped.items()) + ")")
    return rows, info


# ----------------------------------------------------------------------------------------- metrics

def _nan():
    return float("nan")


def isnan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def ece_bins(rows, bins: int = BINS):
    """All `bins` equal-width bins (empty ones included) and the expected calibration error."""
    acc = [[0, 0.0, 0.0] for _ in range(bins)]
    for r in rows:
        b = min(int(r.prob * bins), bins - 1)
        acc[b][0] += 1
        acc[b][1] += r.prob
        acc[b][2] += r.y
    n = len(rows)
    out, ece = [], 0.0
    for b, (k, sp, sy) in enumerate(acc):
        mp = sp / k if k else _nan()
        oy = sy / k if k else _nan()
        out.append({"lo": b / bins, "hi": (b + 1) / bins, "n": k, "mean_p": mp, "observed": oy})
        if k:
            ece += k / n * abs(mp - oy)
    return out, (ece if n else _nan())


def brier(rows) -> float:
    return sum((r.prob - r.y) ** 2 for r in rows) / len(rows) if rows else _nan()


def auroc(rows) -> float:
    """Probability a random positive outranks a random negative (ties count half). NaN if one class."""
    pos = sum(r.y for r in rows)
    neg = len(rows) - pos
    if not pos or not neg:
        return _nan()
    srt = sorted(rows, key=lambda r: r.prob)
    rank_sum, i = 0.0, 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1].prob == srt[i].prob:
            j += 1
        avg = (i + j) / 2 + 1
        rank_sum += avg * sum(srt[k].y for k in range(i, j + 1))
        i = j + 1
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def rates(rows, threshold: float | None = None):
    """False-accept / false-reject. threshold=None: use each row's actual accept decision;
    otherwise accept = probability >= threshold."""
    acc = (lambda r: r.accept) if threshold is None else (lambda r: r.prob >= threshold)
    neg = [r for r in rows if not r.y]
    pos = [r for r in rows if r.y]
    fa = sum(1 for r in neg if acc(r))
    fr = sum(1 for r in pos if not acc(r))
    na = sum(1 for r in rows if acc(r))
    return {"far": fa / len(neg) if neg else _nan(), "frr": fr / len(pos) if pos else _nan(),
            "acceptance": na / len(rows) if rows else _nan(),
            "fa": fa, "neg": len(neg), "fr": fr, "pos": len(pos), "n": len(rows)}


THRESHOLD_GRID = [round(i / 100, 2) for i in range(1, 101)]


def recommend_threshold(rows, target_far: float = 0.05, grid=THRESHOLD_GRID):
    """Lowest threshold (accept = probability >= t) whose false-accept rate <= target_far.
    FAR only falls as t rises, so the lowest such t keeps the most automation. None if unreachable
    or if there are no negatives to measure against."""
    if not any(not r.y for r in rows):
        return None
    for t in grid:
        ar = rates(rows, t)
        if ar["far"] <= target_far + 1e-12:
            return {"threshold": t, **ar}
    return None


def accuracy(rows) -> float:
    return sum(r.correct for r in rows) / len(rows) if rows else _nan()


def current_threshold(rows):
    c = Counter(r.threshold for r in rows)
    common = c.most_common(1)[0][0]
    lo, hi = min(c), max(c)
    return common, lo, hi


def summarize(rows, target_far: float, mode: str = "binary"):
    k = sum(r.correct for r in rows)
    lo, hi = wilson(k, len(rows))
    bins, ece = ece_bins(rows)
    thr, tlo, thi = current_threshold(rows)
    cur = rates(rows)
    rec = recommend_threshold(rows, target_far) if len(rows) >= TINY_N else None
    return {"n": len(rows), "accuracy": k / len(rows), "acc_ci": (lo, hi), "ece": ece, "bins": bins,
            "brier": brier(rows), "auroc": auroc(rows), "current": cur, "threshold": thr,
            "threshold_range": (tlo, thi), "at_threshold": rates(rows, thr),
            "recommended": rec, "small": len(rows) < SMALL_N, "base_rate": sum(r.y for r in rows) / len(rows),
            "conf_bias": confidence_bias(bins, len(rows), mode)}


def confidence_bias(bins, n, mode="binary"):
    """>0: overconfident (probabilities more extreme / higher than reality), <0: underconfident."""
    if not n:
        return _nan()
    s = 0.0
    for b in bins:
        if not b["n"]:
            continue
        direction = 1 if mode == "choice" or b["mean_p"] >= 0.5 else -1
        s += b["n"] / n * direction * (b["mean_p"] - b["observed"])
    return s


def group_summaries(rows, key, target_far, mode="binary"):
    by = defaultdict(list)
    for r in rows:
        by[key(r)].append(r)
    return [(k, summarize(v, target_far, mode)) for k, v in sorted(by.items(), key=lambda kv: (-len(kv[1]), kv[0]))]


def over_time(rows, min_median_n: int = 50):
    """Accuracy per day / ISO week / month: the finest unit whose typical bucket has >= min_median_n rows
    (and that still gives at least 4 points). Returns ([(label, start_date, accuracy, n)], unit)."""
    ts_rows = [r for r in rows if r.ts is not None]
    if len(ts_rows) < 2 or (max(r.ts for r in ts_rows) - min(r.ts for r in ts_rows)).days < 1:
        return [], None
    units = [("day", lambda d: d.date()),
             ("week", lambda d: (d - timedelta(days=d.weekday())).date()),
             ("month", lambda d: d.date().replace(day=1))]
    candidates = []
    for unit, keyf in units:
        by = defaultdict(list)
        for r in ts_rows:
            by[keyf(r.ts)].append(r)
        if len(by) > 400:
            continue
        sizes = sorted(len(v) for v in by.values())
        candidates.append((unit, by, sizes[len(sizes) // 2]))
    usable = [c for c in candidates if len(c[1]) >= 4]
    if not usable:
        usable = candidates[:1]
    if not usable:
        return [], None
    unit, by, _ = next((c for c in usable if c[2] >= min_median_n), usable[-1])
    out = []
    for k in sorted(by):
        label = k.strftime("%Y-%m") if unit == "month" else k.isoformat()
        out.append((label, k, accuracy(by[k]), len(by[k])))
    return out, unit


def version_changes(rows):
    ts_rows = sorted((r for r in rows if r.ts is not None), key=lambda r: r.ts)
    seen, out = set(), []
    for r in ts_rows:
        if r.model_version not in seen:
            seen.add(r.model_version)
            out.append((r.ts, r.model_version))
    return out[1:] if len(out) > 1 else []


def analyze(rows, info: ParseInfo, target_far: float = 0.05):
    overall = summarize(rows, target_far, info.mode)
    segs = group_summaries(rows, lambda r: r.segment, target_far, info.mode) if "segment" in info.columns else []
    models = group_summaries(rows, lambda r: r.model_version, target_far, info.mode) if "model_version" in info.columns else []
    timeline, unit = over_time(rows) if "timestamp" in info.columns else ([], None)
    sweep = [(t, rates(rows, t)) for t in THRESHOLD_GRID]
    ts = [r.ts for r in rows if r.ts is not None]
    res = {"overall": overall, "segments": segs, "models": models, "timeline": timeline, "unit": unit,
           "sweep": sweep, "target_far": target_far, "info": info, "changes": version_changes(rows),
           "period": (min(ts), max(ts)) if ts else None}
    res["verdict"] = verdict(res)
    return res


# ----------------------------------------------------------------------------------------- wording

def pct(x, d=1):
    return "–" if isnan(x) else f"{x * 100:.{d}f}%"


def num(x, d=3):
    return "–" if isnan(x) else f"{x:.{d}f}"


def _words(mode):
    if mode == "binary":
        return {"neg": "cases that should have been rejected", "pos": "good cases", "accept": "accepts",
                "unit": "decisions", "acc_noun": "right"}
    return {"neg": "wrong answers", "pos": "right answers", "accept": "auto-accepts",
            "unit": "decisions", "acc_noun": "right"}


def verdict(res) -> str:
    o, info, tf = res["overall"], res["info"], res["target_far"]
    w = _words(info.mode)
    parts = []
    when = ""
    if res["period"]:
        a, b = res["period"]
        when = f" between {a:%Y-%m-%d} and {b:%Y-%m-%d}"
    lo, hi = o["acc_ci"]
    parts.append(f"Across {o['n']:,} labelled {w['unit']}{when}, the model was right {pct(o['accuracy'])} of the time "
                 f"(95% CI {pct(lo)}–{pct(hi)}).")
    ece = o["ece"]
    # the most telling bin: largest gap among bins with a meaningful share of rows
    big = [b for b in o["bins"] if b["n"] >= max(20, 0.02 * o["n"])]
    worst = max(big, key=lambda b: abs(b["mean_p"] - b["observed"]), default=None)
    what = "turned out yes" if info.mode == "binary" else "were right"
    example = ""
    if worst is not None and abs(worst["mean_p"] - worst["observed"]) >= 0.03:
        example = (f" For example, {w['unit']} it scored around {worst['mean_p']:.0%} {what} "
                   f"{worst['observed']:.0%} of the time (n={worst['n']:,}).")
    bias = o["conf_bias"]
    lean = "overconfident" if bias > 0 else "underconfident"
    if ece < 0.03:
        parts.append(f"Its probabilities are well calibrated (ECE {ece:.3f}): a stated 80% means roughly 80%.")
    elif ece < 0.08:
        parts.append(f"Its probabilities are somewhat off (ECE {ece:.3f}, mostly {lean}).{example}")
    else:
        parts.append(f"Its probabilities are poorly calibrated (ECE {ece:.3f}, {lean}), so they should not be "
                     f"read at face value.{example}")
    cur, thr = o["current"], o["threshold"]
    tlo, thi = o["threshold_range"]
    thr_txt = f"threshold of {thr:.2f}" if tlo == thi else f"thresholds ({tlo:.2f}–{thi:.2f})"
    if isnan(cur["far"]):
        parts.append("There are no negative outcomes in this file, so the false-accept rate cannot be measured.")
    else:
        status = "above" if cur["far"] > tf else "within"
        parts.append(f"At your current {thr_txt}, it {w['accept']} {pct(cur['far'])} of the {w['neg']} "
                     f"({cur['fa']:,} of {cur['neg']:,}), {status} your {pct(tf, 0)} target, and wrongly rejects "
                     f"{pct(cur['frr'])} of the {w['pos']}.")
        rec = o["recommended"]
        if rec is None:
            parts.append(f"No threshold meets a {pct(tf, 0)} false-accept rate on this data; the model needs "
                         "improvement or a human review step for this decision.")
        elif rec["threshold"] > thr + 1e-9 and cur["far"] > tf:
            parts.append(f"Raising the threshold to {rec['threshold']:.2f} would bring false accepts down to "
                         f"{pct(rec['far'])}, at the cost of false rejects rising to {pct(rec['frr'])} "
                         f"(auto-accepted share {pct(cur['acceptance'])} → {pct(rec['acceptance'])}).")
        elif rec["threshold"] < thr - 1e-9 and cur["far"] <= tf:
            parts.append(f"You have headroom: a threshold of {rec['threshold']:.2f} would still meet the target "
                         f"and auto-accept {pct(rec['acceptance'])} instead of {pct(cur['acceptance'])}.")
        elif cur["far"] > tf:
            parts.append(f"A threshold of {rec['threshold']:.2f} applied to the probability would meet the target "
                         f"(false rejects {pct(rec['frr'])}); your recorded decisions don't follow the probability "
                         "cut-off exactly.")
    slices = [("segment", k, s) for k, s in res["segments"]] + [("model", k, s) for k, s in res["models"]]
    solid = [x for x in slices if not x[2]["small"] and len({k for g, k, _ in slices if g == x[0]}) > 1]
    weak = min(solid, key=lambda x: x[2]["accuracy"], default=None)
    if weak and o["accuracy"] - weak[2]["accuracy"] >= 0.03:
        g, k, s = weak
        extra = "" if isnan(s["current"]["far"]) else f", false-accept {pct(s['current']['far'])}"
        parts.append(f"Weakest slice: {g} {k} at {pct(s['accuracy'])} accuracy{extra} (n={s['n']:,}).")
    small = [x for x in slices if x[2]["small"]]
    if small:
        parts.append(f"{len(small)} slice{'s' if len(small) > 1 else ''} ha{'ve' if len(small) > 1 else 's'} fewer "
                     f"than {SMALL_N} rows; treat {'their' if len(small) > 1 else 'its'} numbers as rough.")
    return " ".join(parts)


def terminal_summary(res, out_path: str) -> str:
    o, info = res["overall"], res["info"]
    cur, rec = o["current"], o["recommended"]
    L = [f"Calibrate check: {os.path.basename(info.path)} ({info.mode} decisions, {o['n']:,} labelled rows"
         + (f", {sum(info.skipped.values())} skipped" if info.skipped else "") + ")", ""]
    if any("synthetic" in c.lower() for c in info.comments):
        L += ["  ** SYNTHETIC DATA (per the file header) **", ""]
    L += [_wrap(res["verdict"], 96, ""), ""]
    lo, hi = o["acc_ci"]
    L += [f"  accuracy        {pct(o['accuracy'])}  (95% CI {pct(lo)}–{pct(hi)})",
          f"  ECE (10 bins)   {num(o['ece'])}",
          f"  Brier score     {num(o['brier'])}",
          f"  AUROC           {num(o['auroc'])}",
          f"  false-accept    {pct(cur['far'])}  ({cur['fa']}/{cur['neg']})  @ current threshold {o['threshold']:.2f}",
          f"  false-reject    {pct(cur['frr'])}  ({cur['fr']}/{cur['pos']})"]
    if rec:
        L.append(f"  recommended     {rec['threshold']:.2f} for false-accept <= {pct(res['target_far'], 0)}: "
                 f"false-accept {pct(rec['far'])}, false-reject {pct(rec['frr'])}, auto-accept {pct(rec['acceptance'])}")
    else:
        L.append(f"  recommended     none: no threshold reaches false-accept <= {pct(res['target_far'], 0)}")
    for title, groups in (("segment", res["segments"]), ("model_version", res["models"])):
        if not groups:
            continue
        L += ["", f"  by {title}:", f"    {'':24} {'n':>7} {'accuracy':>9} {'ECE':>7} {'FAR':>7} {'FRR':>7}"]
        for k, s in groups:
            flag = "  small n" if s["small"] else ""
            L.append(f"    {k[:24]:24} {s['n']:>7,} {pct(s['accuracy']):>9} {num(s['ece']):>7} "
                     f"{pct(s['current']['far']):>7} {pct(s['current']['frr']):>7}{flag}")
    for note in info.notes:
        L += ["", _wrap("note: " + note, 96, "  ")]
    L += ["", f"Report: {out_path}  (generated locally; nothing left this machine)"]
    return "\n".join(L)


def _wrap(text, width, indent):
    words, lines, cur = text.split(), [], indent
    for wd in words:
        if len(cur) + len(wd) + 1 > width and cur.strip():
            lines.append(cur.rstrip())
            cur = indent
        cur += wd + " "
    if cur.strip():
        lines.append(cur.rstrip())
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------- HTML

def _e(s):
    return html.escape(str(s))


SVG_HEAD = ('<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img" '
            'font-family="system-ui,-apple-system,Segoe UI,Roboto,sans-serif" font-size="11">')


def reliability_svg(bins, w=460, h=330):
    pl, pr, pt, pb = 44, 14, 14, 92
    iw, ih = w - pl - pr, h - pt - pb
    X = lambda v: pl + iw * v
    Y = lambda v: pt + ih * (1 - v)
    o = [SVG_HEAD.format(w=w, h=h)]
    for k in range(6):
        v = k / 5
        o.append(f'<line x1="{X(0):.1f}" x2="{X(1):.1f}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#e5e7eb"/>')
        o.append(f'<text x="{pl - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" fill="#6b7280">{v:.1f}</text>')
        o.append(f'<text x="{X(v):.1f}" y="{pt + ih + 14}" text-anchor="middle" fill="#6b7280">{v:.1f}</text>')
    o.append(f'<line x1="{X(0)}" y1="{Y(0)}" x2="{X(1)}" y2="{Y(1)}" stroke="#9ca3af" stroke-dasharray="4 3"/>')
    bw = iw / len(bins)
    for i, b in enumerate(bins):  # observed-rate bars per bin
        if b["n"]:
            o.append(f'<rect x="{X(b["lo"]) + 2:.1f}" y="{Y(b["observed"]):.1f}" width="{bw - 4:.1f}" '
                     f'height="{ih * b["observed"]:.1f}" fill="#bfdbfe"/>')
    pts = [(b["mean_p"], b["observed"]) for b in bins if b["n"]]
    if pts:
        d = " ".join(f"{'M' if j == 0 else 'L'}{X(p):.1f},{Y(q):.1f}" for j, (p, q) in enumerate(pts))
        o.append(f'<path d="{d}" fill="none" stroke="#1d4ed8" stroke-width="2"/>')
        for p, q in pts:
            o.append(f'<circle cx="{X(p):.1f}" cy="{Y(q):.1f}" r="3.5" fill="#1d4ed8"/>')
    o.append(f'<text x="{pl + iw / 2}" y="{pt + ih + 30}" text-anchor="middle" fill="#374151">predicted probability</text>')
    o.append(f'<text x="12" y="{pt + ih / 2}" transform="rotate(-90 12 {pt + ih / 2})" text-anchor="middle" '
             f'fill="#374151">observed rate</text>')
    # count strip
    maxn = max((b["n"] for b in bins), default=1) or 1
    cy = pt + ih + 42
    o.append(f'<text x="{pl - 6}" y="{cy + 22}" text-anchor="end" fill="#6b7280">n</text>')
    for b in bins:
        hh = 26 * b["n"] / maxn
        o.append(f'<rect x="{X(b["lo"]) + 2:.1f}" y="{cy + 26 - hh:.1f}" width="{bw - 4:.1f}" height="{hh:.1f}" fill="#9ca3af"/>')
        o.append(f'<text x="{X(b["lo"]) + bw / 2:.1f}" y="{cy + 40}" text-anchor="middle" fill="#374151">{b["n"]:,}</text>')
    o.append("</svg>")
    return "".join(o)


def tradeoff_svg(sweep, current, recommended, target, w=460, h=310):
    pl, pr, pt, pb = 44, 14, 14, 62
    iw, ih = w - pl - pr, h - pt - pb
    X = lambda v: pl + iw * v
    Y = lambda v: pt + ih * (1 - v)
    o = [SVG_HEAD.format(w=w, h=h)]
    for k in range(6):
        v = k / 5
        o.append(f'<line x1="{X(0):.1f}" x2="{X(1):.1f}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#e5e7eb"/>')
        o.append(f'<text x="{pl - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" fill="#6b7280">{v:.0%}</text>')
        o.append(f'<text x="{X(v):.1f}" y="{pt + ih + 14}" text-anchor="middle" fill="#6b7280">{v:.1f}</text>')
    o.append(f'<line x1="{X(0)}" x2="{X(1)}" y1="{Y(target):.1f}" y2="{Y(target):.1f}" stroke="#16a34a" stroke-dasharray="5 4"/>')
    o.append(f'<text x="{X(1) - 2}" y="{Y(target) - 4:.1f}" text-anchor="end" fill="#16a34a">target {target:.0%}</text>')
    for key, col in (("far", "#dc2626"), ("frr", "#2563eb"), ("acceptance", "#6b7280")):
        pts = [(t, a[key]) for t, a in sweep if not isnan(a[key])]
        if pts:
            d = " ".join(f"{'M' if j == 0 else 'L'}{X(t):.1f},{Y(v):.1f}" for j, (t, v) in enumerate(pts))
            dash = ' stroke-dasharray="2 3"' if key == "acceptance" else ""
            o.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2"{dash}/>')
    for t, lab, col in ((current, "current", "#111827"), (recommended, "recommended", "#16a34a")):
        if t is None:
            continue
        o.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{pt}" y2="{pt + ih}" stroke="{col}" stroke-width="1.5"/>')
        right = t > 0.72
        o.append(f'<text x="{X(t) + (-4 if right else 4):.1f}" y="{pt + 12 + (14 if lab == "recommended" else 0)}" '
                 f'text-anchor="{"end" if right else "start"}" fill="{col}">{lab} {t:.2f}</text>')
    lx, ly = pl, h - 10
    for lab, col in (("false-accept rate", "#dc2626"), ("false-reject rate", "#2563eb"), ("auto-accepted share", "#6b7280")):
        o.append(f'<rect x="{lx}" y="{ly - 4}" width="12" height="3" fill="{col}"/><text x="{lx + 16}" y="{ly}" fill="#374151">{lab}</text>')
        lx += 22 + 6.0 * len(lab)
    o.append(f'<text x="{pl + iw / 2}" y="{pt + ih + 30}" text-anchor="middle" fill="#374151">threshold (accept if probability ≥ threshold)</text>')
    o.append("</svg>")
    return "".join(o)


def timeline_svg(timeline, unit, changes, w=940, h=260, min_n=20):
    pl, pr, pt, pb = 46, 14, 14, 40
    iw, ih = w - pl - pr, h - pt - pb
    accs = [a for _, _, a, n in timeline if n >= min_n] or [a for _, _, a, _ in timeline]
    ymax = 1.0
    ymin = max(0.0, ymax - 0.2 * math.ceil((ymax - min(accs) + 0.03) / 0.2))
    m = max(len(timeline) - 1, 1)
    X = lambda i: pl + iw * i / m
    Y = lambda v: pt + ih * (1 - (v - ymin) / (ymax - ymin))
    o = [SVG_HEAD.format(w=w, h=h)]
    maxn = max(n for *_, n in timeline)
    bw = max(2.0, iw / (len(timeline) + 1) * 0.7)
    for i, (_, _, _, n) in enumerate(timeline):  # volume bars in the lower third
        hh = ih * 0.3 * n / maxn
        o.append(f'<rect x="{X(i) - bw / 2:.1f}" y="{pt + ih - hh:.1f}" width="{bw:.1f}" height="{hh:.1f}" fill="#e5e7eb"/>')
    for k in range(5):
        v = ymin + (ymax - ymin) * k / 4
        o.append(f'<line x1="{pl}" x2="{w - pr}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#eef0f3"/>')
        o.append(f'<text x="{pl - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" fill="#6b7280">{v:.0%}</text>')
    step = max(1, len(timeline) // 8)
    for i, (lab, *_r) in enumerate(timeline):
        if i % step == 0 or i == len(timeline) - 1:
            if i != len(timeline) - 1 and len(timeline) - 1 - i < step / 2:
                continue
            anchor = "end" if i == len(timeline) - 1 and len(timeline) > 1 else ("start" if i == 0 else "middle")
            o.append(f'<text x="{X(i):.1f}" y="{h - 22}" text-anchor="{anchor}" fill="#6b7280">{_e(lab)}</text>')
    starts = [s for _, s, _, _ in timeline]
    for ts, ver in changes[:6]:
        d = ts.date()
        idx = max([i for i, s in enumerate(starts) if s <= d], default=0)
        o.append(f'<line x1="{X(idx):.1f}" x2="{X(idx):.1f}" y1="{pt}" y2="{pt + ih}" stroke="#a855f7" stroke-dasharray="3 3"/>')
        o.append(f'<text x="{X(idx) + 4:.1f}" y="{pt + 11}" fill="#7e22ce">{_e(ver)} starts</text>')
    d = " ".join(f"{'M' if j == 0 else 'L'}{X(i):.1f},{Y(a):.1f}" for j, (i, (_, _, a, _)) in enumerate(enumerate(timeline)))
    o.append(f'<path d="{d}" fill="none" stroke="#111827" stroke-width="2"/>')
    for i, (_, _, a, n) in enumerate(timeline):
        fill = "#111827" if n >= min_n else "#ffffff"
        o.append(f'<circle cx="{X(i):.1f}" cy="{Y(a):.1f}" r="3" fill="{fill}" stroke="#111827"><title>{a:.1%} (n={n})</title></circle>')
    o.append(f'<text x="{pl}" y="{h - 4}" fill="#6b7280">accuracy per {unit} (line) · volume (grey bars) · hollow points: n &lt; {min_n} (treat as noise)</text>')
    o.append("</svg>")
    return "".join(o)


CSS = """
*{box-sizing:border-box}body{margin:0;background:#f8fafc;color:#0f172a;font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:28px 24px}h1{font-size:22px;margin:0}h2{font-size:15px;margin:0 0 8px}
.sub{color:#64748b;font-size:12px;margin-top:2px}.card{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:16px;margin-bottom:14px}
.verdict{background:#eff6ff;border-color:#bfdbfe;font-size:15px}.banner{background:#fef3c7;border:1px solid #fcd34d;color:#78350f;border-radius:10px;padding:8px 14px;margin:14px 0;font-size:13px}
.tiles{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:14px}.tile{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:12px}
.tile .k{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b}.tile .v{font-size:22px;font-weight:600;margin-top:2px}.tile .s{font-size:11px;color:#64748b}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}.grid2>.card{margin-bottom:14px}
table{border-collapse:collapse;width:100%;font-size:13px}th{text-align:left;font-size:11px;text-transform:uppercase;color:#64748b;font-weight:600;padding:4px 8px 4px 0}
td{border-top:1px solid #eef0f3;padding:5px 8px 5px 0}td.r,th.r{text-align:right}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.flag{display:inline-block;font-size:11px;background:#fef3c7;color:#92400e;border-radius:6px;padding:0 6px}.bad{color:#b91c1c;font-weight:600}.good{color:#15803d;font-weight:600}
.muted{color:#64748b;font-size:12px}svg{width:100%;height:auto;display:block}footer{color:#64748b;font-size:12px;margin-top:20px;text-align:center}
footer a{color:#1d4ed8}@media(max-width:800px){.tiles{grid-template-columns:repeat(2,1fr)}.grid2{grid-template-columns:1fr}}
@media print{body{background:#fff}.card,.tile{break-inside:avoid}}
"""


def _group_table(groups, target_far, label):
    if not groups:
        return ""
    head = (f"<table><thead><tr><th>{_e(label)}</th><th class='r'>n</th><th class='r'>accuracy</th><th class='r'>ECE</th>"
            "<th class='r'>Brier</th><th class='r'>AUROC</th><th class='r'>false-accept</th><th class='r'>false-reject</th>"
            f"<th class='r'>threshold for ≤{target_far:.0%} FA</th><th></th></tr></thead><tbody>")
    trs = []
    for k, s in groups:
        cur, rec = s["current"], s["recommended"]
        far_cls = " class='r bad'" if not isnan(cur["far"]) and cur["far"] > target_far else " class='r'"
        if s["n"] < TINY_N:
            rec_txt = "n too small"
        elif rec is None:
            rec_txt = "unreachable" if cur["neg"] else "no negatives"
        else:
            rec_txt = f"{rec['threshold']:.2f} <span class='muted'>(FR {pct(rec['frr'])})</span>"
        lo, hi = s["acc_ci"]
        flag = f"<span class='flag'>small n</span>" if s["small"] else ""
        trs.append(f"<tr><td class='mono'>{_e(k)}</td><td class='r'>{s['n']:,}</td>"
                   f"<td class='r' title='95% CI {pct(lo)}–{pct(hi)}'>{pct(s['accuracy'])} <span class='muted'>±{(hi - lo) / 2 * 100:.1f}</span></td>"
                   f"<td class='r'>{num(s['ece'])}</td><td class='r'>{num(s['brier'])}</td><td class='r'>{num(s['auroc'])}</td>"
                   f"<td{far_cls}>{pct(cur['far'])} <span class='muted'>{cur['fa']}/{cur['neg']}</span></td>"
                   f"<td class='r'>{pct(cur['frr'])} <span class='muted'>{cur['fr']}/{cur['pos']}</span></td>"
                   f"<td class='r'>{rec_txt}</td><td>{flag}</td></tr>")
    return head + "".join(trs) + "</tbody></table>"


def _gap(b):
    return "–" if not b["n"] else f"{(b['observed'] - b['mean_p']) * 100:+.1f} pp"


def render_html(res) -> str:
    o, info, tf = res["overall"], res["info"], res["target_far"]
    w = _words(info.mode)
    cur, rec = o["current"], o["recommended"]
    lo, hi = o["acc_ci"]
    gen = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z").strip()
    synthetic = any("synthetic" in c.lower() for c in info.comments)
    banner = ""
    if info.comments:
        banner = (f"<div class='banner'>{'<b>SYNTHETIC DATA.</b> ' if synthetic else ''}"
                  f"{'<br>'.join(_e(c) for c in info.comments)}</div>")
    tlo, thi = o["threshold_range"]
    thr_txt = f"{o['threshold']:.2f}" if tlo == thi else f"{tlo:.2f}–{thi:.2f} (most common {o['threshold']:.2f})"
    tile = lambda k, v, s, cls="": f"<div class='tile'><div class='k'>{k}</div><div class='v {cls}'>{v}</div><div class='s'>{s}</div></div>"
    far_cls = "bad" if not isnan(cur["far"]) and cur["far"] > tf else ("good" if not isnan(cur["far"]) else "")
    disc = "P(yes) vs outcome" if info.mode == "binary" else "confidence vs correct"
    tiles = "".join([
        tile("Accuracy", pct(o["accuracy"]), f"95% CI {pct(lo)}–{pct(hi)} · n={o['n']:,}"),
        tile("Calibration (ECE)", num(o["ece"]), "10 bins · 0 is perfect"),
        tile("Brier score", num(o["brier"]), "0 is perfect · lower is better"),
        tile("AUROC", num(o["auroc"]), f"{disc} · 0.5 = chance"),
        tile("False-accept", pct(cur["far"]), f"{cur['fa']:,}/{cur['neg']:,} {w['neg']} @ {o['threshold']:.2f}", far_cls),
        tile("False-reject", pct(cur["frr"]), f"{cur['fr']:,}/{cur['pos']:,} {w['pos']}"),
    ])
    bin_rows = "".join(
        f"<tr><td class='mono'>{b['lo']:.1f}–{b['hi']:.1f}</td><td class='r'>{b['n']:,}</td><td class='r'>{num(b['mean_p'], 2)}</td>"
        f"<td class='r'>{num(b['observed'], 2)}</td><td class='r'>{_gap(b)}</td></tr>"
        for b in o["bins"])
    obs_lab = "observed yes rate" if info.mode == "binary" else "observed accuracy"
    bins_table = (f"<table><thead><tr><th>bin</th><th class='r'>n</th><th class='r'>mean predicted</th>"
                  f"<th class='r'>{obs_lab}</th><th class='r'>gap</th></tr></thead><tbody>{bin_rows}</tbody></table>")
    at = o["at_threshold"]
    rec_row = (f"<tr><td><b>recommended</b></td><td class='r'><b>{rec['threshold']:.2f}</b></td><td class='r good'>{pct(rec['far'])}</td>"
               f"<td class='r'>{pct(rec['frr'])}</td><td class='r'>{pct(rec['acceptance'])}</td></tr>") if rec else (
        f"<tr><td><b>recommended</b></td><td class='r' colspan='4'>No threshold reaches false-accept ≤ {tf:.0%} on this data.</td></tr>")
    asdecided = ""
    differs = any(cur[k] != at[k] for k in ("fa", "fr"))
    if differs:
        asdecided = (f"<tr><td>as decided <span class='muted'>(your recorded decisions)</span></td><td class='r'>{thr_txt}</td>"
                     f"<td class='r'>{pct(cur['far'])}</td><td class='r'>{pct(cur['frr'])}</td><td class='r'>{pct(cur['acceptance'])}</td></tr>")
    cost = ""
    if rec:
        d_frr = rec["frr"] - at["frr"]
        d_acc = rec["acceptance"] - at["acceptance"]
        if not isnan(d_frr):
            extra_fr = rec["fr"] - at["fr"]
            cost = (f"<p>Moving from {o['threshold']:.2f} to <b>{rec['threshold']:.2f}</b> changes false accepts "
                    f"{pct(at['far'])} → {pct(rec['far'])} and false rejects {pct(at['frr'])} → {pct(rec['frr'])} "
                    f"({'+' if extra_fr >= 0 else ''}{extra_fr:,} {w['pos']} in this file sent to review or rejected), "
                    f"auto-accepted share {pct(at['acceptance'])} → {pct(rec['acceptance'])} ({d_acc * 100:+.1f} pp).</p>")
    thr_table = (f"<table><thead><tr><th></th><th class='r'>threshold</th><th class='r'>false-accept</th><th class='r'>false-reject</th>"
                 f"<th class='r'>auto-accepted</th></tr></thead><tbody>{asdecided}"
                 f"<tr><td>current</td><td class='r'>{o['threshold']:.2f}</td><td class='r'>{pct(at['far'])}</td><td class='r'>{pct(at['frr'])}</td>"
                 f"<td class='r'>{pct(at['acceptance'])}</td></tr>{rec_row}</tbody></table>")
    trade = tradeoff_svg(res["sweep"], o["threshold"], rec["threshold"] if rec else None, tf)
    timeline_html = ""
    if res["timeline"]:
        timeline_html = (f"<div class='card'><h2>Accuracy over time</h2>"
                         f"{timeline_svg(res['timeline'], res['unit'], res['changes'])}</div>")
    elif "timestamp" in info.columns:
        timeline_html = "<div class='card'><h2>Accuracy over time</h2><p class='muted'>Not enough distinct dates to plot.</p></div>"
    seg_html = (f"<div class='card'><h2>By segment</h2>{_group_table(res['segments'], tf, 'segment')}</div>"
                if res["segments"] else "")
    mod_html = (f"<div class='card'><h2>By model version</h2>{_group_table(res['models'], tf, 'model_version')}</div>"
                if res["models"] else "")
    small_note = (f"<p class='muted'>Slices with fewer than {SMALL_N} rows are flagged <span class='flag'>small n</span>; "
                  f"no threshold is recommended below {TINY_N}. ± is the half-width of the 95% Wilson interval in points.</p>"
                  if res["segments"] or res["models"] else "")
    colmap = ", ".join(f"{k} ← <span class='mono'>{_e(v)}</span>" for k, v in info.columns.items())
    skipped = "; ".join(f"{_e(k)}: {v:,}" for k, v in info.skipped.items()) or "none"
    notes = "".join(f"<li>{_e(n)}</li>" for n in info.notes)
    kind = "yes/no decisions" if info.mode == "binary" else "choice decisions (top-choice probability)"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Calibration check · {_e(os.path.basename(info.path))}</title><style>{CSS}</style></head><body><div class="wrap">
<h1>Calibration check <span style="color:#94a3b8;font-weight:400">/ {_e(os.path.basename(info.path))}</span></h1>
<div class="sub">Generated {_e(gen)} · {o['n']:,} labelled rows · {kind} · threshold {thr_txt} · target false-accept ≤ {tf:.0%}</div>
{banner}
<div class="card verdict"><h2>Verdict</h2><p style="margin:0">{_e(res['verdict'])}</p></div>
<div class="tiles">{tiles}</div>
<div class="grid2">
<div class="card"><h2>Reliability diagram</h2>{reliability_svg(o['bins'])}
<p class="muted">Points on the dashed diagonal are perfectly calibrated. Grey bars underneath show how many rows fall in each bin.</p></div>
<div class="card"><h2>Reliability bins <span class="muted">(ECE {num(o['ece'])})</span></h2>{bins_table}</div>
</div>
<div class="grid2">
<div class="card"><h2>Threshold recommendation <span class="muted">(lowest threshold with false-accept ≤ {tf:.0%})</span></h2>{thr_table}{cost}
<p class="muted">Accept = probability ≥ threshold. Point estimate on this file; check it on fresh data before applying.</p></div>
<div class="card"><h2>Threshold trade-off</h2>{trade}</div>
</div>
{seg_html}{mod_html}{small_note}
{timeline_html}
<div class="card"><h2>About this data</h2><ul class="muted" style="margin:0;padding-left:18px">
<li>File: <span class="mono">{_e(info.path)}</span> · {info.n_read:,} rows read · {o['n']:,} used · skipped: {skipped}</li>
<li>Columns: {colmap}</li><li>Threshold: {_e(info.threshold_source)} · Decision: {_e(info.decision_source)}</li>
<li>Base rate ({'outcome yes' if info.mode == 'binary' else 'decision correct'}): {pct(o['base_rate'])}</li>{notes}
<li>ECE = Σ (bin share × |mean predicted − observed|) over 10 equal-width bins. Brier = mean (p − outcome)². AUROC by rank (ties count half).</li>
</ul></div>
<footer>Generated locally by Calibrate (open source). Hosted monitoring: <a href="{FOOTER_URL}">{FOOTER_URL}</a></footer>
</div></body></html>"""


def run_check(path: str, threshold: float = 0.5, target_far: float = 0.05, out: str = "report.html",
              threshold_explicit: bool = False):
    if not 0 <= target_far <= 1:
        raise CheckError("--target-far must be between 0 and 1 (e.g. 0.05 for 5%)")
    if not 0 <= threshold <= 1:
        raise CheckError("--threshold must be between 0 and 1")
    rows, info = parse_csv(path, default_threshold=threshold, threshold_explicit=threshold_explicit)
    res = analyze(rows, info, target_far)
    d = os.path.dirname(os.path.abspath(out))
    os.makedirs(d, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(render_html(res))
    res["out"] = out
    res["summary"] = terminal_summary(res, out)
    return res
