"""LLM provider contract.

A provider is a thin transport: given a prompt and a model id, return text. It
knows nothing about review axes, findings or verification. Retry and failover
policy lives in :mod:`pr_review.providers.dispatch`; a provider's only duty is
to classify its transport failures with :func:`http_error`, so the dispatcher
can tell "wait and retry" (429) from "this model is gone, move on" (404).
"""

from __future__ import annotations

import contextlib
import json
import re
from typing import Protocol, runtime_checkable

from pr_review.config import ProviderConfig
from pr_review.errors import ProviderError, ProviderUnavailable, RateLimited
from pr_review.registry import Registry


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def complete(
        self,
        prompt: str,
        *,
        model: str,
        max_tokens: int = 1400,
        temperature: float = 0.2,
        system: str | None = None,
    ) -> str: ...


# Registered by *kind* (e.g. "openai_compat"); the selector instantiates with a
# ProviderConfig.
PROVIDERS: Registry[LLMProvider] = Registry("provider")


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_JSON_ARR_RE = re.compile(r"\[.*\]", re.DOTALL)


def extract_json(text: str) -> object:
    """Best-effort: pull the first JSON value out of a model response.

    Tolerates ```json fences and leading/trailing prose.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for rx in (_JSON_OBJ_RE, _JSON_ARR_RE):
        m = rx.search(cleaned)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON value found in model response")


# HTTP status -> failure class. Transport modules share this so every provider
# fails over identically.
#   throttle  : same model, retry after a wait
#   permanent : this (provider, model) will not work -- fail over immediately
#   other     : transient (5xx, proxy hiccup) -- short retry, then fail over
_THROTTLE_CODES = frozenset({429})
_PERMANENT_CODES = frozenset({400, 401, 403, 404, 405, 413, 422, 451})


def http_error(name: str, code: int, detail: str, retry_after: str = "") -> ProviderError:
    """Map an HTTP status from a provider onto the error hierarchy."""
    label = f"{code} from {name}: {detail}"
    if code in _THROTTLE_CODES:
        wait = 60.0
        with contextlib.suppress(TypeError, ValueError):
            wait = max(1.0, float(retry_after))
        return RateLimited(label, retry_after=wait)
    if code in _PERMANENT_CODES:
        # 413 lands here deliberately: the prompt does not shrink on retry, but
        # a provider with a larger context window may still accept it.
        return ProviderUnavailable(label)
    return ProviderError(label)


def build(kind: str, cfg: ProviderConfig) -> LLMProvider:
    return PROVIDERS.create(kind, cfg)
