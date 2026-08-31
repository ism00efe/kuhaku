"""OpenAI-compatible chat-completions provider (stdlib ``urllib`` only).

Covers Groq, OpenRouter, Cerebras, NVIDIA NIM, Mistral, DeepInfra, OpenAI and
any other ``/chat/completions`` endpoint. Cloudflare fronts some of these
(notably Groq) and 403s the default ``Python-urllib`` User-Agent, so a custom
UA is always sent.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from pr_review.config import ProviderConfig
from pr_review.errors import ProviderError, ProviderUnavailable
from pr_review.providers.base import PROVIDERS, http_error

_USER_AGENT = "pr-review/1.0 (+https://github.com/ism00efe/kuhaku)"


@PROVIDERS.register("openai_compat")
class OpenAICompatProvider:
    def __init__(self, cfg: ProviderConfig) -> None:
        self.cfg = cfg
        self.name = cfg.kind
        self._base = cfg.base_url.rstrip("/")
        if not self._base:
            raise ProviderError("openai_compat provider requires base_url")

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
                f"no API key: set ${self.cfg.api_key_env} for provider {self.name!r}"
            )
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        req = urllib.request.Request(
            f"{self._base}/chat/completions",
            data=json.dumps(payload).encode(),
            method="POST",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "User-Agent": _USER_AGENT,
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            raise http_error(
                self.name, exc.code, detail, exc.headers.get("Retry-After", "")
            ) from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"network error calling {self.name}: {exc.reason}") from exc

        try:
            return body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected response shape from {self.name}: {body}") from exc
