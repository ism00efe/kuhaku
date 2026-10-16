"""The input checks core offers every tool, behind one object.

A tool runs untrusted text through :class:`InputPolicy` instead of calling each check
itself, so a check added here reaches every tool that uses the policy without a change
to the tool. :class:`InputSecurity` holds which checks are on; a tool nests it in its
own settings, so a new option here appears in every tool's settings too.

Every check is listed with its cost. Those that cost nothing are on by default; those
that get in the way of ordinary use are off until asked for -- a default may cost CPU
and memory, never a download, and must not stop a quick experiment.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from pydantic import BaseModel

from ..observability.metrics import record_redactions
from ..sanitization import Redaction, sanitize_text
from .guard import inspect_query


class InputSecurity(BaseModel):
    """Which input checks run."""

    sanitize: bool = False
    """Mask PII (e-mails, tokens, IPs, card and national-ID numbers, phone numbers).
    Off by default: masked text cannot be extracted -- a schema asking for an e-mail or
    a phone number cannot be filled with it on."""

    input_guard: bool = False
    """Refuse text matching a known prompt-injection pattern. Off by default: it also
    refuses chat logs with ``user:`` lines and text that only quotes such a phrase."""


@dataclass
class InputPolicy:
    """Applies :class:`InputSecurity` to text. Keeps the redaction counts of every
    :meth:`mask` call so a tool can report them once."""

    options: InputSecurity = field(default_factory=InputSecurity)
    _counts: Counter[str] = field(default_factory=Counter, repr=False)

    def screen(self, text: str) -> str | None:
        """Why ``text`` must be refused, or ``None`` when it may go on."""

        if not self.options.input_guard:
            return None
        safe, reason = inspect_query(text)
        return None if safe else reason

    def mask(self, text: str) -> str:
        """``text`` with every enabled masking applied."""

        if not self.options.sanitize:
            return text
        masked, redactions = sanitize_text(text)
        for redaction in redactions:
            self._counts[redaction.label] += redaction.count
        return masked

    def redactions(self) -> list[Redaction]:
        """What :meth:`mask` has masked so far -- labels and counts, never values --
        recorded once on the redaction metric."""

        found = [Redaction(label, count) for label, count in self._counts.items()]
        record_redactions(found)
        return found
