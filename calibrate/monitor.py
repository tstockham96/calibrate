"""The Monitor wraps decision calls and logs every typed answer."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone

from . import store
from .providers import MockProvider, get_default_provider


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _public_state(state):
    # Strip simulator hints so they never influence hashes or leave the process.
    if isinstance(state, dict) and "_sim" in state:
        return {k: v for k, v in state.items() if k != "_sim"}
    return state


class Monitor:
    """
    m = Monitor("calibrate.db")                      # mock provider unless TYPESAFE_API_KEY is set
    res = m.decide(state, {"eligible": {"type": "noul", "instructions": "..."}},
                   model="jev-1.13.0", segment="channel=chat", thresholds={"eligible": 0.7})
    res["decision_id"]  -> keep it; send the real outcome later with m.record_outcome(...)
    """

    def __init__(self, db_path: str = "calibrate.db", provider=None, shadow_providers=None):
        self.con = store.connect(db_path)
        self.provider = provider or get_default_provider()
        self.shadow_providers = shadow_providers or []

    def _log(self, decision_id, ts, provider, role, requested_model, resp, questions, state,
             segment, latency_ms, thresholds):
        state_hash = _hash(_public_state(state))
        for key, q in questions.items():
            a = resp["answers"].get(key)
            if a is None:
                continue
            thr = None
            if a["type"] == "noul":
                prob = a["noul"]
                thr = (thresholds or {}).get(key, 0.5)
                answer = "yes" if prob >= thr else "no"
                conf = None
            elif a["type"] == "choice":
                answer = a["choice"]
                prob = a["probabilities"][answer]
                conf = a.get("confidence")
            else:  # score
                top = max(a["probabilities"], key=a["probabilities"].get)
                answer = top
                prob = a["probabilities"][top]
                conf = a.get("confidence")
            store.insert_decision(self.con, {
                "decision_id": decision_id, "question_key": key, "ts": ts,
                "provider": getattr(provider, "name", "custom"), "role": role,
                "model": resp.get("model"), "requested_model": requested_model,
                "question_hash": _hash({k: q.get(k) for k in ("type", "instructions", "criteria")}),
                "state_hash": state_hash, "qtype": a["type"], "answer": answer, "prob": prob,
                "confidence": conf, "threshold": thr, "segment": segment, "latency_ms": latency_ms,
                "input_tokens": (resp.get("usage") or {}).get("input_tokens"),
            })

    def decide(self, state, questions: dict, model: str = "jev-latest", segment: str | None = None,
               thresholds: dict | None = None, ts: str | None = None, decision_id: str | None = None):
        decision_id = decision_id or uuid.uuid4().hex
        ts = ts or datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()
        call_state = state if isinstance(self.provider, MockProvider) else _public_state(state)
        resp = self.provider.decide(call_state, questions, model=model)
        latency = (time.perf_counter() - t0) * 1000
        self._log(decision_id, ts, self.provider, "primary", model, resp, questions, state, segment,
                  latency, thresholds)
        # Shadow mode: same request to other providers; logged, never returned to the caller.
        for sp in self.shadow_providers:
            try:
                s_state = state if isinstance(sp, MockProvider) else _public_state(state)
                t1 = time.perf_counter()
                sresp = sp.decide(s_state, questions)
                self._log(decision_id, ts, sp, "shadow", None, sresp, questions, state, segment,
                          (time.perf_counter() - t1) * 1000, thresholds)
            except Exception as e:  # shadow failures must never break the primary path
                print(f"[calibrate] shadow provider {getattr(sp, 'name', sp)} failed: {e}")
        self.con.commit()
        resp = dict(resp)
        resp["decision_id"] = decision_id
        return resp

    def record_outcome(self, decision_id: str, question_key: str, actual):
        store.ingest_outcome(self.con, decision_id, question_key, actual)
