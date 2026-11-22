"""Context Engine: local-LLM 'vibe check' on an unstructured news feed.

Produces a Regime Context Vector each minute:
    macro_shock      in [0, 1]   (surprise / tail-risk intensity)
    directional_bias in [-1, 1]  (bullish .. bearish)
    news_driven      in {0, 1}   (is price currently headline-driven?)

The LLM is a locally hosted quantized model reached over a localhost HTTP
API (Ollama / llama.cpp) so there is no external network hop. If no local
server is reachable, a neutral MOCK vector is returned so the sandbox still
runs — in this environment there is no Ollama and no live squawk feed, so
the mock is what actually drives training. The interface is production-shaped
so a real feed can be dropped in unchanged.
"""
from __future__ import annotations

import functools
import json
from dataclasses import dataclass

import numpy as np

try:
    import requests
except Exception:  # requests optional for the mock path
    requests = None

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "llama3:8b-instruct-q4_0"

_SYSTEM = (
    "You are a macro squawk-box analyst. Read the latest market headlines and "
    "return ONLY compact JSON with keys macro_shock (0..1), directional_bias "
    "(-1..1), news_driven (0 or 1). No prose."
)


@dataclass
class RegimeContext:
    macro_shock: float = 0.0
    directional_bias: float = 0.0
    news_driven: int = 0

    def as_vector(self) -> np.ndarray:
        return np.array([self.macro_shock, self.directional_bias,
                         float(self.news_driven)], dtype=np.float32)


class ContextEngine:
    def __init__(self, url: str = OLLAMA_URL, model: str = MODEL,
                 timeout: float = 0.5):
        self.url, self.model, self.timeout = url, model, timeout
        self._live = self._probe()
        self._rng = np.random.default_rng(0)

    def _probe(self) -> bool:
        if requests is None:
            return False
        try:
            requests.get(self.url.replace("/api/generate", "/"),
                         timeout=self.timeout)
            return True
        except Exception:
            return False

    def vibe_check(self, headlines: list[str]) -> RegimeContext:
        """Return the RegimeContext for the current minute's headlines."""
        if not self._live:
            return self._mock(headlines)
        prompt = _SYSTEM + "\nHEADLINES:\n" + "\n".join(headlines[-20:])
        try:
            r = requests.post(self.url, timeout=self.timeout, json={
                "model": self.model, "prompt": prompt, "stream": False,
                "format": "json", "options": {"temperature": 0.0}})
            d = json.loads(r.json()["response"])
            return RegimeContext(
                float(np.clip(d.get("macro_shock", 0), 0, 1)),
                float(np.clip(d.get("directional_bias", 0), -1, 1)),
                int(bool(d.get("news_driven", 0))))
        except Exception:
            return self._mock(headlines)

    def _mock(self, headlines: list[str]) -> RegimeContext:
        """Neutral-noisy stand-in when no local LLM is reachable. Occasionally
        flags news_driven so the veto path is exercised during training."""
        shock = float(self._rng.beta(1.2, 8))          # usually calm
        bias = float(self._rng.normal(0, 0.2))
        news = int(shock > 0.6)
        return RegimeContext(shock, float(np.clip(bias, -1, 1)), news)


def news_veto(entry_fn):
    """Hard structural veto: block MEAN-REVERSION entries when the regime is
    headline-driven (don't catch a falling knife into fresh news). Wraps an
    entry function taking (..., ctx: RegimeContext, is_mean_reversion: bool)."""
    @functools.wraps(entry_fn)
    def wrapped(*args, ctx: RegimeContext, is_mean_reversion: bool = False,
                **kwargs):
        if is_mean_reversion and ctx.news_driven == 1:
            return 0  # forced Hold
        return entry_fn(*args, ctx=ctx, is_mean_reversion=is_mean_reversion,
                        **kwargs)
    return wrapped
