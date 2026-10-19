"""Choosing which messages of a conversation go into the next LLM call.

All helpers work on whole *turns*: a turn starts at a ``user`` message and runs up to
the next one, so an assistant reply (or, later, a tool call and its result) is never
separated from the message it answers. Leading ``system`` messages are not part of any
turn and are kept by default.

- :func:`last_turns` -- the newest ``n`` turns.
- :func:`within_token_budget` -- the newest turns that fit a token budget, by a
  pluggable :data:`TokenEstimator` (:func:`approximate_tokens` by default).
- :class:`ContextWindow` -- both limits together.

The newest turn is always kept, even when it alone exceeds the budget: a window that
drops the message being answered is never what a caller wants, and truncating inside
a message would change its meaning. Check the budget yourself if a hard cap matters.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..models import Message

TokenEstimator = Callable[[Message], int]


def approximate_tokens(message: Message) -> int:
    """About four characters per token plus a small per-message overhead -- a
    provider-independent estimate. Non-text parts count as zero; pass a real tokenizer
    as the estimator when the budget must be exact."""

    return 4 + math.ceil(len(message.text) / 4)


def split_turns(messages: Sequence[Message]) -> tuple[list[Message], list[list[Message]]]:
    """``(leading system messages, turns)``.

    Messages before the first ``user`` message that are not ``system`` messages form a
    turn of their own, so nothing is lost.
    """

    system: list[Message] = []
    turns: list[list[Message]] = []
    index = 0
    while index < len(messages) and messages[index].role == "system":
        system.append(messages[index])
        index += 1
    for message in messages[index:]:
        if message.role == "user" or not turns:
            turns.append([message])
        else:
            turns[-1].append(message)
    return system, turns


def last_turns(
    messages: Sequence[Message], n: int, *, keep_system: bool = True
) -> list[Message]:
    if n < 1:
        raise ValueError(f"n must be at least 1; got {n}.")
    system, turns = split_turns(messages)
    return _join(system if keep_system else [], turns[-n:])


def within_token_budget(
    messages: Sequence[Message],
    max_tokens: int,
    *,
    estimator: TokenEstimator = approximate_tokens,
    keep_system: bool = True,
) -> list[Message]:
    """The newest whole turns whose estimated size, with the kept system messages,
    fits ``max_tokens``. The newest turn is kept regardless (see the module docstring)."""

    system, turns = split_turns(messages)
    head = system if keep_system else []
    return _join(head, _fit(turns, max_tokens - _size(head, estimator), estimator))


@dataclass
class ContextWindow:
    """A reusable windowing policy: ``apply(messages)`` returns the messages to send.

    ``max_turns`` and ``max_tokens`` are both optional; with neither, every message is
    kept.
    """

    max_turns: int | None = None
    max_tokens: int | None = None
    estimator: TokenEstimator = approximate_tokens
    keep_system: bool = True

    def __post_init__(self) -> None:
        if self.max_turns is not None and self.max_turns < 1:
            raise ValueError(f"max_turns must be at least 1; got {self.max_turns}.")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError(f"max_tokens must be at least 1; got {self.max_tokens}.")

    def apply(self, messages: Sequence[Message]) -> list[Message]:
        system, turns = split_turns(messages)
        head = system if self.keep_system else []
        kept = turns[-self.max_turns :] if self.max_turns is not None else list(turns)
        budget = None if self.max_tokens is None else self.max_tokens - _size(head, self.estimator)
        if budget is not None:
            kept = _fit(kept, budget, self.estimator)
        return _join(head, kept)




def _size(messages: Sequence[Message], estimator: TokenEstimator) -> int:
    return sum(estimator(message) for message in messages)


def _fit(
    turns: Sequence[list[Message]], budget: int, estimator: TokenEstimator
) -> list[list[Message]]:
    kept: list[list[Message]] = []
    used = 0
    for turn in reversed(turns):
        cost = _size(turn, estimator)
        if kept and used + cost > budget:
            break
        kept.append(turn)
        used += cost
    kept.reverse()
    return kept


def _join(head: Sequence[Message], turns: Sequence[list[Message]]) -> list[Message]:
    return [*head, *(message for turn in turns for message in turn)]
