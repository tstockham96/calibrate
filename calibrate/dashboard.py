"""Render a static HTML dashboard (inline SVG, no JS, no external assets except Tailwind CDN)."""
from __future__ import annotations

import html
import math
from datetime import datetime, timezone

from . import metrics as M
from . import store

COLORS = {"overall": "#0f172a", "channel=email": "#2563eb", "channel=chat": "#dc2626",
          "baseline": "#64748b", "recent": "#dc2626", "target": "#16a34a"}
PALETTE = ["#2563eb", "#dc2626", "#0f172a", "#7c3aed", "#ea580c"]


def _esc(s):
    return html.escape(str(s))


def _pct(x, d=1):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.{d}f}%"


def line_chart(series: dict, labels: list, ymin=0.0, ymax=1.0, w=560, h=240, marker_x=None,
               marker_label="", hline=None, hline_label="", yfmt=lambda v: f"{v:.0%}"):
    pl, pr, pt, pb = 46, 12, 14, 34
    iw, ih = w - pl - pr, h - pt - pb
    n = max(len(labels) - 1, 1)
    X = lambda i: pl + iw * i / n
    Y = lambda v: pt + ih * (1 - (v - ymin) / (ymax - ymin))
    out = [f'<svg viewBox="0 0 {w} {h}" class="w-full h-auto" xmlns="http://www.w3.org/2000/svg" font-family="ui-sans-serif,system-ui" font-size="10">']
    for k in range(5):
        v = ymin + (ymax - ymin) * k / 4
        out.append(f'<line x1="{pl}" x2="{w - pr}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#e2e8f0"/>')
        out.append(f'<text x="{pl - 6}" y="{Y(v) + 3:.1f}" text-anchor="end" fill="#64748b">{yfmt(v)}</text>')
    step = max(1, len(labels) // 7)
    for i, lab in enumerate(labels):
        if (i % step == 0 and len(labels) - 1 - i >= step / 2) or i == len(labels) - 1:
            out.append(f'<text x="{X(i):.1f}" y="{h - 14}" text-anchor="middle" fill="#64748b">{_esc(lab[5:])}</text>')
    if hline is not None:
        out.append(f'<line x1="{pl}" x2="{w - pr}" y1="{Y(hline):.1f}" y2="{Y(hline):.1f}" stroke="{COLORS["target"]}" stroke-dasharray="5 4"/>')
        out.append(f'<text x="{w - pr - 2}" y="{Y(hline) - 4:.1f}" text-anchor="end" fill="{COLORS["target"]}">{_esc(hline_label)}</text>')
    if marker_x is not None:
        out.append(f'<line x1="{X(marker_x):.1f}" x2="{X(marker_x):.1f}" y1="{pt}" y2="{pt + ih}" stroke="#a855f7" stroke-dasharray="3 3"/>')
        out.append(f'<text x="{X(marker_x) + 4:.1f}" y="{pt + 10}" fill="#a855f7">{_esc(marker_label)}</text>')
    for idx, (name, pts) in enumerate(series.items()):
        col = COLORS.get(name, PALETTE[idx % len(PALETTE)])
        pts = [(i, v) for i, v in pts if v is not None and not math.isnan(v)]
        if not pts:
            continue
        d = " ".join(f"{'M' if j == 0 else 'L'}{X(i):.1f},{Y(min(max(v, ymin), ymax)):.1f}" for j, (i, v) in enumerate(pts))
        sw = 2.4 if name == "overall" else 1.8
        out.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{sw}"/>')
    lx = pl
    for idx, name in enumerate(series):
        col = COLORS.get(name, PALETTE[idx % len(PALETTE)])
        out.append(f'<rect x="{lx}" y="{h - 9}" width="10" height="3" fill="{col}"/><text x="{lx + 14}" y="{h - 5}" fill="#334155">{_esc(name)}</text>')
        lx += 18 + 6.2 * len(name)
    out.append("</svg>")
    return "".join(out)


def reliability_chart(curves: dict, w=300, h=260):
    pl, pr, pt, pb = 40, 10, 12, 40
    iw, ih = w - pl - pr, h - pt - pb
    X = lambda v: pl + iw * v
    Y = lambda v: pt + ih * (1 - v)
    out = [f'<svg viewBox="0 0 {w} {h}" class="w-full h-auto" xmlns="http://www.w3.org/2000/svg" font-family="ui-sans-serif,system-ui" font-size="10">']
    for k in range(6):
        v = k / 5
        out.append(f'<line x1="{X(0)}" x2="{X(1)}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#e2e8f0"/>')
        out.append(f'<text x="{pl - 5}" y="{Y(v) + 3:.1f}" text-anchor="end" fill="#64748b">{v:.1f}</text>')
        out.append(f'<text x="{X(v):.1f}" y="{pt + ih + 13}" text-anchor="middle" fill="#64748b">{v:.1f}</text>')
    out.append(f'<line x1="{X(0)}" y1="{Y(0)}" x2="{X(1)}" y2="{Y(1)}" stroke="#94a3b8" stroke-dasharray="4 3"/>')
    out.append(f'<text x="{pl + iw / 2}" y="{h - 14}" text-anchor="middle" fill="#475569">predicted P(yes)</text>')
    lx = pl
    for name, (pts, ece) in curves.items():
        col = COLORS.get(name, "#0f172a")
        d = " ".join(f"{'M' if j == 0 else 'L'}{X(mp):.1f},{Y(oy):.1f}" for j, (_, mp, oy, n) in enumerate(pts))
        out.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2"/>')
        for _, mp, oy, n in pts:
            r = 2 + min(5, math.sqrt(n) / 4)
            out.append(f'<circle cx="{X(mp):.1f}" cy="{Y(oy):.1f}" r="{r:.1f}" fill="{col}" fill-opacity="0.75"/>')
        label = f"{name} (ECE {ece:.3f})"
        out.append(f'<rect x="{lx}" y="{h - 7}" width="10" height="3" fill="{col}"/><text x="{lx + 14}" y="{h - 3}" fill="#334155">{_esc(label)}</text>')
        lx += 20 + 5.6 * len(label)
    out.append("</svg>")
    return "".join(out)


def render(db_path: str, out_path: str, question: str = "refund_eligible", target_far: float = 0.05,
           threshold: float | None = None, synthetic: bool = True):
    con = store.connect(db_path)
    rows = store.joined(con, question)
    if not rows:
        raise SystemExit(f"No labelled rows for question {question!r}. Ingest outcomes first.")
    qtype = rows[0]["qtype"]
    thr = threshold if threshold is not None else (rows[-1]["threshold"] or 0.5)
    all_dec = store.all_decisions(con)
    n_logged = len({r["decision_id"] for r in all_dec})
    base, recent = M.split_windows(rows)
    segments = sorted({r["segment"] or "-" for r in rows})
    days = sorted({r["ts"][:10] for r in rows})
    di = {d: i for i, d in enumerate(days)}

    # model version change marker
    first_model = rows[0]["model"]
    marker = None
    marker_label = ""
    for r in rows:
        if r["model"] != first_model:
            marker = di[r["ts"][:10]]
            marker_label = f"version → {r['model']}"
            break

    def series(fn, min_n=10):
        s = {"overall": [(di[d], v) for d, v, n in M.daily(rows, fn) if n >= min_n]}
        for seg in segments:
            sr = [r for r in rows if (r["segment"] or "-") == seg]
            s[seg] = [(di[d], v) for d, v, n in M.daily(sr, fn) if n >= min_n]
        return s

    acc_chart = line_chart(series(M.accuracy), days, ymin=0.6, ymax=1.0, marker_x=marker, marker_label=marker_label)
    far_chart = ""
    if qtype == "noul":
        far_chart = line_chart(series(lambda rs: M.accept_rates(rs, thr)["far"]), days, ymin=0.0, ymax=0.4,
                               marker_x=marker, marker_label=marker_label, hline=target_far,
                               hline_label=f"target ≤ {target_far:.0%}")
    rb, eb = M.reliability(base)
    rr, er = M.reliability(recent)
    calib = reliability_chart({"baseline": (rb, eb), "recent": (rr, er)})

    k_acc = M.accuracy(recent)
    k_correct = sum(M.is_correct(r) for r in recent)
    lo, hi = M.wilson(k_correct, len(recent))
    ar_recent = M.accept_rates(recent, thr) if qtype == "noul" else None
    drift = M.drift_report(rows, thr)

    # threshold recommendations per segment (recent window)
    rec_rows = []
    if qtype == "noul":
        for seg in ["all"] + segments:
            rs = recent if seg == "all" else [r for r in recent if (r["segment"] or "-") == seg]
            if len(rs) < 40:
                continue
            cur = M.accept_rates(rs, thr)
            rec = M.recommend_threshold(rs, max_far=target_far)
            t = rec["target"]
            rec_rows.append((seg, len(rs), cur, t))

    # shadow comparison
    shadow_html = ""
    prov = con.execute("SELECT DISTINCT provider, role FROM decisions").fetchall()
    if any(p["role"] == "shadow" for p in prov):
        trs = []
        for q in sorted({r["question_key"] for r in all_dec}):
            for role in ("primary", "shadow"):
                rs = store.joined(con, q, role=role)
                _, rs_recent = M.split_windows(rs)
                if not rs_recent:
                    continue
                name = f"{rs_recent[0]['provider']} · {rs_recent[-1]['model']}"
                far = _pct(M.accept_rates(rs_recent, thr)["far"]) if rs_recent[0]["qtype"] == "noul" else "n/a"
                trs.append(f"<tr class='border-t'><td class='py-1.5 pr-3 font-mono text-xs'>{_esc(q)}</td><td class='pr-3'>{_esc(role)}</td><td class='pr-3 text-xs'>{_esc(name)}</td><td class='pr-3'>{_pct(M.accuracy(rs_recent))}</td><td class='pr-3'>{M.reliability(rs_recent)[1]:.3f}</td><td>{far}</td></tr>")
        shadow_html = ("<table class='w-full text-sm'><thead class='text-left text-slate-500 text-xs uppercase'><tr><th class='py-1'>question</th><th>role</th><th>provider · model</th><th class='pr-3'>accuracy</th><th class='pr-3'>ECE</th><th>false-accept</th></tr></thead><tbody>"
                       + "".join(trs) + "</tbody></table>")

    alerts_html = "".join(
        f"<li class='flex gap-2 items-start'><span class='mt-1 h-2 w-2 rounded-full bg-red-600 shrink-0'></span><span><b class='font-mono text-xs'>{_esc(a['scope'])}</b> — {_esc(a['detail'])} <span class='text-slate-400'>(n={a['n']}, last 7 days vs prior)</span></span></li>"
        for a in drift["alerts"]) or "<li class='text-slate-500'>No alerts.</li>"

    rec_html = "".join(
        f"<tr class='border-t'><td class='py-1.5 pr-3 font-mono text-xs'>{_esc(seg)}</td><td class='pr-3'>{n}</td>"
        f"<td class='pr-3'>{thr:.2f} → <b>{(t[0] if t else float('nan')):.2f}</b></td>"
        f"<td class='pr-3'>{_pct(cur['far'])} → <b>{_pct(t[1]['far']) if t else 'unreachable'}</b></td>"
        f"<td class='pr-3'>{_pct(cur['frr'])} → {_pct(t[1]['frr']) if t else '–'}</td>"
        f"<td>{_pct(cur['acceptance'])} → {_pct(t[1]['acceptance']) if t else '–'}</td></tr>"
        for seg, n, cur, t in rec_rows)

    banner = ("<div class='bg-amber-100 border border-amber-300 text-amber-900 text-sm rounded-lg px-4 py-2 mb-5'>"
              "<b>SYNTHETIC DATA.</b> Mock provider + invented scenario generated by <code>calibrate demo</code>. "
              "These numbers are not measurements of Jev or any real model.</div>") if synthetic else ""
    gen = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    tile = lambda label, val, sub: (f"<div class='bg-white rounded-xl border p-4'><div class='text-xs uppercase tracking-wide text-slate-500'>{label}</div>"
                                    f"<div class='text-2xl font-semibold mt-1'>{val}</div><div class='text-xs text-slate-500 mt-1'>{sub}</div></div>")
    tiles = [
        tile("Accuracy · last 7d", _pct(k_acc), f"95% CI {_pct(lo)}–{_pct(hi)} · n={len(recent)}"),
        tile("False-accept rate · last 7d", _pct(ar_recent["far"]) if ar_recent else "n/a",
             f"{ar_recent['fa']}/{ar_recent['neg']} negatives accepted @ {thr:.2f}" if ar_recent else ""),
        tile("Calibration (ECE)", f"{er:.3f}", f"baseline {eb:.3f} · 10 bins"),
        tile("Decisions logged", f"{n_logged:,}", f"{len({r['decision_id'] for r in rows}):,} with ground truth"),
    ]
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Calibrate · demo dashboard</title><script src="https://cdn.tailwindcss.com"></script></head>
<body class="bg-slate-50 text-slate-900"><div class="max-w-6xl mx-auto p-6">
<div class="flex items-baseline justify-between mb-3"><div><h1 class="text-xl font-semibold">Calibrate <span class="text-slate-400 font-normal">/ refund-bot / <span class="font-mono text-base">{_esc(question)}</span> ({_esc(qtype)})</span></h1>
<p class="text-xs text-slate-500">Generated {gen} · threshold in force {thr:.2f} · baseline = prior days, recent = last 7 days with ground truth</p></div></div>
{banner}
<div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">{''.join(tiles)}</div>
<div class="grid md:grid-cols-3 gap-3 mb-4">
<div class="bg-white rounded-xl border p-4 md:col-span-2"><h2 class="text-sm font-semibold mb-1">Accuracy over time, by segment</h2>{acc_chart}</div>
<div class="bg-white rounded-xl border p-4"><h2 class="text-sm font-semibold mb-1">Calibration curve</h2>{calib}</div></div>
<div class="grid md:grid-cols-3 gap-3 mb-4">
<div class="bg-white rounded-xl border p-4 md:col-span-2"><h2 class="text-sm font-semibold mb-1">False-accept rate @ {thr:.2f}</h2>{far_chart}</div>
<div class="bg-white rounded-xl border p-4"><h2 class="text-sm font-semibold mb-2 text-red-700">Drift alerts</h2><ul class="space-y-2 text-sm">{alerts_html}</ul></div></div>
<div class="grid md:grid-cols-2 gap-3">
<div class="bg-white rounded-xl border p-4"><h2 class="text-sm font-semibold mb-2">Threshold recommendation <span class="font-normal text-slate-500">(lowest threshold with false-accept ≤ {target_far:.0%}, last 7d)</span></h2>
<table class="w-full text-sm"><thead class="text-left text-slate-500 text-xs uppercase"><tr><th class="py-1">segment</th><th>n</th><th>threshold</th><th>false-accept</th><th>false-reject</th><th>auto-accept</th></tr></thead><tbody>{rec_html}</tbody></table>
<p class="text-xs text-slate-500 mt-2">Point estimates on recent labelled data; review before applying. Never auto-applied.</p></div>
<div class="bg-white rounded-xl border p-4"><h2 class="text-sm font-semibold mb-2">Shadow mode: provider side-by-side <span class="font-normal text-slate-500">(last 7d, your labels)</span></h2>{shadow_html or '<p class="text-sm text-slate-500">No shadow provider configured.</p>'}</div>
</div>
<p class="text-xs text-slate-400 mt-6">Calibrate prototype v0.0.1 · private monitoring of your own decisions and outcomes · not affiliated with TypeSafe AI or OpenAI.</p>
</div></body></html>"""
    with open(out_path, "w") as f:
        f.write(doc)
    return {"out": out_path, "alerts": drift["alerts"], "recent_n": len(recent), "ece_recent": er,
            "accuracy_recent": k_acc, "recommendations": rec_rows}
