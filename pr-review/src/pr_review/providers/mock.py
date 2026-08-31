"""Deterministic, offline provider -- for tests and demos only.

Inspects the prompt for the markers the engine emits (``AXIS: <name>``,
``TASK: plan`` / ``TASK: verify``) and returns schema-valid JSON derived from
the prompt text, so end-to-end tests can assert on real structure without a
network call.

It emits a labelled placeholder finding, which is exactly what a real review
must never do. That is why it is no longer the CI fallback: with no API key the
pipeline reports a structural-only review instead (see
:meth:`pr_review.config.Config.has_llm`). Reach for this provider only via an
explicit ``force_provider = "mock"`` / ``--provider mock``.
"""

from __future__ import annotations

import hashlib
import json
import re

from pr_review.config import ProviderConfig
from pr_review.providers.base import PROVIDERS


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


@PROVIDERS.register("mock")
class MockProvider:
    name = "mock"

    def __init__(self, cfg: ProviderConfig | None = None) -> None:
        self.cfg = cfg or ProviderConfig(kind="mock")

    def complete(
        self,
        prompt: str,
        *,
        model: str = "mock",
        max_tokens: int = 1400,
        temperature: float = 0.2,
        system: str | None = None,
    ) -> str:
        low = prompt.lower()
        if "task: plan" in low:
            return self._plan(prompt)
        if "task: verify" in low:
            return self._verify(prompt)
        return self._review(prompt)

    # ------------------------------------------------------------------ #

    def _plan(self, prompt: str) -> str:
        axes = ["correctness", "method", "scope", "structure"]
        deps_changed = "dependency_changes" in prompt and "[]" not in _section(
            prompt, "dependency_changes"
        )
        entries = [
            {"axis": "correctness", "depth": "basic", "reason": "always inspect logic"},
            {"axis": "scope", "depth": "basic", "reason": "check intent match"},
        ]
        if deps_changed:
            entries.append(
                {"axis": "structure", "depth": "normal", "reason": "dependency files changed"}
            )
        for a in axes:
            if not any(e["axis"] == a for e in entries):
                entries.append({"axis": a, "depth": "off", "reason": "not indicated"})
        return json.dumps(
            {
                "change_type": "mock-classified",
                "risk_areas": ["mock"],
                "reviews": entries,
            }
        )

    def _review(self, prompt: str) -> str:
        axis_m = re.search(r"AXIS:\s*([a-z_]+)", prompt)
        axis = axis_m.group(1) if axis_m else "correctness"
        file_m = re.search(r"^\+\+\+ (?:b/)?(\S+)", prompt, re.MULTILINE)
        target = file_m.group(1) if file_m else "UNKNOWN"
        # One low-confidence, clearly-labelled finding so downstream stages have
        # something to act on; deterministic verification will mark it uncertain.
        return json.dumps(
            {
                "axis": axis,
                "findings": [
                    {
                        "severity": "info",
                        "file": target,
                        "line": None,
                        "issue": f"[mock:{axis}] deterministic placeholder finding "
                        f"(digest {_digest(prompt)})",
                        "evidence": "generated offline by MockProvider",
                        "reasoning": "no LLM configured; run with a real provider for "
                        "substantive findings",
                        "confidence": 0.2,
                    }
                ],
            }
        )

    def _verify(self, prompt: str) -> str:
        return json.dumps(
            {
                "verdict": "uncertain",
                "evidence": "MockProvider cannot verify against repository evidence",
            }
        )


def _section(prompt: str, label: str) -> str:
    idx = prompt.find(label)
    return prompt[idx : idx + 120] if idx >= 0 else ""
