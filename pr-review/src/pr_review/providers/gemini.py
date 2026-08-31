"""Google Gemini provider via the AI Studio REST API (API-key mode, stdlib only).

Vertex AI (service-account / WIF) is intentionally out of scope for the
prototype -- AI Studio keys are free-tier and keep the runtime dependency-free.
A Vertex transport can be added as its own provider kind later without touching
this module.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from pr_review.config import ProviderConfig
from pr_review.errors import ProviderError, ProviderUnavailable
from pr_review.providers.base import PROVIDERS, http_error

_BASE = "https://generativelanguage.googleapis.com/v1beta"
_USER_AGENT = "pr-review/1.0"


@PROVIDERS.register("gemini")
class GeminiProvider:
    name = "gemini"

    def __init__(self, cfg: ProviderConfig) -> None:
        self.cfg = cfg

    def complete(
        self,
        prompt: str,
        *,
        model: str,
        max_tokens: int = 1400,
        temperature: float = 0.2,
        system: str | None = None,
    ) -> str:
        key = self.cfg.api_key()
        if not key:
            raise ProviderUnavailable(
                f"no API key: set ${self.cfg.api_key_env} for gemini"
            )
        payload: dict = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        url = f"{_BASE}/models/{model}:generateContent?key={key}"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            method="POST",
            headers={"Content-Type": "application/json", "User-Agent": _USER_AGENT},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            raise http_error(
                "gemini", exc.code, detail, exc.headers.get("Retry-After", "")
            ) from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"network error calling gemini: {exc.reason}") from exc

        try:
            parts = body["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected gemini response: {body}") from exc
