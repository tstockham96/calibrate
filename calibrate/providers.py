"""Decision providers.

TypeSafeProvider follows the documented HTTP shape at https://docs.typesafe.ai/api.md:
    POST https://api.typesafe.ai/v1/systemone
    Authorization: Bearer <API_KEY>
    {"state": ..., "model": "jev-latest" | pinned, "questions": {key: {type, instructions, criteria}}}
Response: {"model": "jev-1.13.0", "answers": {key: {...}}, "usage": {"input_tokens": N, ...}}

The API key stays in YOUR process (env var TYPESAFE_API_KEY). Calibrate never needs it: only the
logged decision metadata is stored. This matters for TypeSafe MCA s.2.4 (credentials must not be shared).

MockProvider returns answers in the same shape, deterministically, for demos and tests. It is not a model.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
import urllib.error
import urllib.request

TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"


class ProviderError(RuntimeError):
    pass


class TypeSafeProvider:
    name = "typesafe"

    def __init__(self, api_key: str | None = None, url: str = TYPESAFE_URL, timeout: float = 30.0,
                 max_retries: int = 3):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise ProviderError("TYPESAFE_API_KEY is not set")
        self.url = url
        self.timeout = timeout
        self.max_retries = max_retries

    def decide(self, state, questions: dict, model: str = "jev-latest") -> dict:
        body = json.dumps({"state": state, "model": model, "questions": questions}).encode()
        delay = 0.5
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(self.url, data=body, method="POST", headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            })
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                # Docs: back off on 429 Too Many Requests and 529 Overloaded.
                if e.code in (429, 529) and attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise ProviderError(f"TypeSafe HTTP {e.code}: {e.read()[:300]!r}") from e
        raise ProviderError("unreachable")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class MockProvider:
    """Deterministic fake provider producing Jev-shaped answers. SYNTHETIC - not a real model.

    If the state contains a "_sim" dict (only the bundled synthetic generator adds it), the mock uses
    it to simulate a model with a given skill / bias / overconfidence so the dashboard has something to
    show. Otherwise answers are pseudo-random but stable per (state, question).
    """
    name = "mock"

    def __init__(self, model_name: str = "mock-jev-1.13", skill: float = 2.2, bias: float = 0.0,
                 overconfidence: float = 1.0, use_sim_profile: bool = True, name: str = "mock"):
        self.model_name = model_name
        self.use_sim_profile = use_sim_profile
        self.name = name
        self.skill = skill
        self.bias = bias
        self.overconfidence = overconfidence

    def _rng(self, state, key):
        h = hashlib.sha256((json.dumps(state, sort_keys=True, default=str) + key + self.model_name).encode())
        return random.Random(int(h.hexdigest()[:16], 16))

    def decide(self, state, questions: dict, model: str | None = None) -> dict:
        sim = state.get("_sim", {}) if isinstance(state, dict) else {}
        truth_map = sim.get("truth", {})
        if not self.use_sim_profile:
            sim = {}
        resolved = sim.get("model") or self.model_name
        answers = {}
        for key, q in questions.items():
            rng = self._rng(state, key)
            qtype = q.get("type")
            skill = sim.get("skill", self.skill)
            bias = sim.get("bias", self.bias)
            oc = sim.get("overconfidence", self.overconfidence)
            truth = truth_map.get(key)
            if qtype == "noul":
                sign = (1 if truth else -1) if truth is not None else rng.choice([1, -1])
                logit = sign * skill + bias + rng.gauss(0, 1.35)
                p = _sigmoid(logit * oc)
                answers[key] = {"type": "noul", "noul": round(p, 4)}
            elif qtype in ("choice", "score"):
                opts = list(q["criteria"].keys()) if qtype == "choice" else [str(i) for i in range(len(q["criteria"]))]
                logits = [rng.gauss(0, 1.0) for _ in opts]
                if truth is not None and str(truth) in opts:
                    logits[opts.index(str(truth))] += skill
                m = max(l * oc for l in logits)
                exps = [math.exp(l * oc - m) for l in logits]
                s = sum(exps)
                probs = {o: round(e / s, 4) for o, e in zip(opts, exps)}
                top = max(probs, key=probs.get)
                conf = round(max(0.0, (3 * probs[top] - 1) / 2), 4)  # illustrative only
                if qtype == "choice":
                    answers[key] = {"type": "choice", "choice": top, "probabilities": probs, "confidence": conf}
                else:
                    exp_score = sum(int(o) * p for o, p in probs.items())
                    legend = {str(i): lvl for i, lvl in enumerate(q["criteria"])}
                    answers[key] = {"type": "score", "score": round(exp_score, 3), "legend": legend,
                                    "probabilities": probs, "confidence": conf}
            else:
                raise ProviderError(f"unknown question type {qtype!r}")
        return {"model": resolved, "answers": answers,
                "usage": {"input_tokens": len(json.dumps(state, default=str)) // 4, "output_tokens": 0},
                "_mock": True}


class OpenAIDecisionsProvider:
    """Placeholder. OpenAI announced a Decisions API in limited preview on 2026-09-29, but no public
    request schema was available when this prototype was written, so this adapter is intentionally
    unimplemented rather than guessed."""
    name = "openai-decisions"

    def __init__(self, *a, **kw):
        raise NotImplementedError("OpenAI Decisions API schema not public yet; adapter pending.")


def get_default_provider():
    """Real TypeSafe provider if TYPESAFE_API_KEY is set, otherwise the mock."""
    if os.environ.get("TYPESAFE_API_KEY"):
        return TypeSafeProvider()
    return MockProvider()
