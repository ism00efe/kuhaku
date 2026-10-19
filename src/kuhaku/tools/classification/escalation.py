"""Provider escalation and the structured-completion call classification makes.

Classification tries a cheap provider first and a stronger one only when the cheap
one's answer is not good enough. :func:`escalate` is that walk:

- Each provider in order gets one *attempt* -- a function of the provider that returns
  an :class:`Attempt`: the tool's value, whether it is accepted (stop here), and a rank
  used to pick the best unaccepted value.
- A provider that cannot be reached (:class:`LLMError`) or cannot take the content
  (:class:`UnsupportedContentError`) is skipped with a warning.
- When no attempt is accepted, the best one is returned (highest ``rank``; on a tie the
  later, presumably stronger, provider wins) with ``accepted=False``.
- Only when **no** provider produced any reply does it raise: the last provider's
  ``UnsupportedContentError`` as-is (a configuration problem), otherwise
  :class:`LLMUnavailableError`. An earlier provider's imperfect answer still says
  something about the input; a dropped connection on the last provider does not erase it.

:func:`complete_json` is one structured completion: the provider's native
structured-output mode when it has one and the caller allows it, else the prompt alone.
It returns the raw text (parsing and validation stay with the tool, which owns the
repair loop) and the call's token usage read off the response, not ``last_usage``.

:func:`capable_providers` filters a provider list down to those that accept a set of
content types (images, documents), raising when none does.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from kuhaku.core.llm.base import (
    LLMError,
    LLMProvider,
    LLMUnavailableError,
    StructuredOutputError,
    TokenUsage,
    UnsupportedContentError,
)
from kuhaku.core.llm.provider import as_chat_provider, capabilities_of
from kuhaku.core.media import MediaPart
from kuhaku.core.models import Message, TextPart

T = TypeVar("T")

logger = logging.getLogger(__name__)

__all__ = [
    "Attempt",
    "Completion",
    "Escalation",
    "capable_providers",
    "complete_json",
    "escalate",
]


@dataclass(frozen=True)
class Attempt(Generic[T]):
    """One provider's outcome. ``accepted`` stops the walk; ``rank`` orders the
    unaccepted ones (higher is better)."""

    value: T
    accepted: bool
    rank: int = 0


@dataclass(frozen=True)
class Escalation(Generic[T]):
    """What :func:`escalate` returns: the chosen value and the provider that produced
    it, whether it was accepted, and the providers that were skipped and why."""

    value: T
    llm: LLMProvider
    accepted: bool
    skipped: tuple[tuple[str, str], ...] = field(default_factory=tuple)


def escalate(
    providers: Sequence[LLMProvider],
    attempt: Callable[[LLMProvider], Attempt[T]],
    *,
    log: logging.Logger | None = None,
    what: str = "llm",
) -> Escalation[T]:
    """Walk ``providers`` calling ``attempt`` until one is accepted. See the module
    docstring for the policy. ``what`` prefixes the log messages ("extraction ...")."""

    log = log or logger
    if not providers:
        raise LLMUnavailableError(f"{what}: no provider to call.")
    best: tuple[Attempt[T], LLMProvider] | None = None
    skipped: list[tuple[str, str]] = []
    last_error: Exception | None = None
    for position, llm in enumerate(providers):
        is_last = position == len(providers) - 1
        try:
            outcome = attempt(llm)
        except UnsupportedContentError as exc:
            last_error = exc
            skipped.append((llm.name, str(exc)))
            log.warning(
                "%s provider rejected the content, %s",
                what,
                "keeping the earlier answer" if is_last and best else "falling back",
                extra={"llm": llm.name, "error": str(exc)},
            )
            continue
        except LLMError as exc:
            last_error = exc
            skipped.append((llm.name, str(exc)))
            log.warning(
                "%s provider unavailable, %s",
                what,
                "keeping the earlier answer" if is_last and best else "falling back",
                extra={"llm": llm.name, "error": str(exc)},
            )
            continue
        if outcome.accepted:
            return Escalation(outcome.value, llm, True, tuple(skipped))
        if best is None or outcome.rank >= best[0].rank:
            best = (outcome, llm)
        if not is_last:
            log.info(
                "%s answer not accepted, escalating",
                what,
                extra={"llm": llm.name, "next_llm": providers[position + 1].name},
            )
    if best is None:
        if isinstance(last_error, UnsupportedContentError):
            raise last_error
        raise LLMUnavailableError(str(last_error)) from last_error
    return Escalation(best[0].value, best[1], False, tuple(skipped))


def capable_providers(
    providers: Iterable[LLMProvider], content_types: Iterable[str], *, what: str = "llm"
) -> list[LLMProvider]:
    """The providers whose ``capabilities.content_types`` include every type in
    ``content_types``; raises :class:`UnsupportedContentError` when none does."""

    providers = list(providers)
    needed = set(content_types)
    if not needed:
        return providers
    capable = [llm for llm in providers if needed <= capabilities_of(llm).content_types]
    if not capable:
        declared = "; ".join(
            f"{llm.name}: {', '.join(sorted(capabilities_of(llm).content_types))}"
            for llm in providers
        )
        raise UnsupportedContentError(
            f"No {what} provider accepts {', '.join(sorted(needed))} input "
            f"({declared}). Use a multimodal model, or declare its content types "
            f"with the provider's content_types= / KUHAKU_LLM_CONTENT_TYPES."
        )
    for llm in providers:
        if llm not in capable:
            logger.info(
                "%s provider skipped: cannot take the content", what, extra={"llm": llm.name}
            )
    return capable


@dataclass(frozen=True)
class Completion:
    """One completion's raw text, its token usage, and whether the provider's native
    structured-output mode produced it."""

    text: str
    usage: TokenUsage | None = None
    native: bool = False


def uses_native(llm: LLMProvider, allowed: bool = True) -> bool:
    """True when ``llm`` has a native structured-output mode and ``allowed``."""

    return allowed and capabilities_of(llm).native_structured_output


def complete_json(
    llm: LLMProvider,
    system: str,
    prompt: str,
    *,
    json_schema: dict[str, Any] | None = None,
    name: str = "response",
    attachments: Sequence[MediaPart] = (),
    native: bool = True,
) -> Completion:
    """Ask ``llm`` for a JSON reply to ``system`` + ``prompt``.

    With ``json_schema`` and a provider that has a native structured-output mode (and
    ``native`` left on), the schema is passed natively as well; otherwise the prompt is
    sent alone and must describe the reply itself. ``attachments`` go in the user turn
    before the prompt (images before the words that refer to them, as vendors
    recommend). Neither path repairs: a native reply that still is not JSON (cut off at
    ``max_tokens``, say) comes back as its raw text, for the caller's own repair loop.
    """

    content: str | list[Any] = [*attachments, TextPart(prompt)] if attachments else prompt
    messages = [Message.system(system), Message.user(content)]
    chat = as_chat_provider(llm)
    # A v1 provider lifted by emulation reports its ``last_usage`` side channel as the
    # response's usage, and that attribute keeps the previous call's value when a reply
    # carries none -- so a usage object that was already there before this call is not
    # this call's.
    before = getattr(llm, "last_usage", None)
    if json_schema is not None and uses_native(llm, native):
        try:
            response = chat.generate_structured(
                messages, json_schema, name=name, max_repair_attempts=0
            )
        except StructuredOutputError as exc:
            return Completion(exc.raw, None, True)
        return Completion(response.text, _fresh(response.usage, before), True)
    reply = chat.chat(messages)
    return Completion(reply.text, _fresh(reply.usage, before), False)


def _fresh(usage: TokenUsage | None, before: Any) -> TokenUsage | None:
    if usage is None or (before is not None and usage is before):
        return None
    return usage
