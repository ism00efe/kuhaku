"""Windowing: whole turns, last-N, token budget, summarization."""

from __future__ import annotations

import pytest

from kuhaku.core.llm.base import LLMCapabilities, LLMResponse
from kuhaku.core.llm.provider import BaseLLMProvider
from kuhaku.core.models import Message
from kuhaku.core.session import (
    ContextWindow,
    approximate_tokens,
    last_turns,
    split_turns,
    within_token_budget,
)

SYSTEM = Message.system("be brief")


def convo(n: int) -> list[Message]:
    messages = [SYSTEM]
    for i in range(n):
        messages += [Message.user(f"q{i}"), Message.assistant(f"a{i}")]
    return messages


def texts(messages):
    return [m.text for m in messages]


def one_token(message: Message) -> int:
    return 1


def test_split_turns_groups_replies_with_their_user_message():
    system, turns = split_turns([SYSTEM, Message.assistant("greeting"), *convo(2)[1:]])
    assert system == [SYSTEM]
    assert [texts(t) for t in turns] == [["greeting"], ["q0", "a0"], ["q1", "a1"]]


def test_last_turns_keeps_system_and_whole_turns():
    assert texts(last_turns(convo(5), 2)) == ["be brief", "q3", "a3", "q4", "a4"]
    assert texts(last_turns(convo(5), 2, keep_system=False)) == ["q3", "a3", "q4", "a4"]
    assert last_turns(convo(1), 10) == convo(1)
    with pytest.raises(ValueError):
        last_turns(convo(1), 0)


def test_token_budget_counts_system_and_keeps_newest_turns_that_fit():
    # 1 system + 2 per turn: budget 5 fits system + two turns.
    window = within_token_budget(convo(5), 5, estimator=one_token)
    assert texts(window) == ["be brief", "q3", "a3", "q4", "a4"]


def test_the_newest_turn_is_kept_even_over_budget():
    assert texts(within_token_budget(convo(3), 1, estimator=one_token)) == ["be brief", "q2", "a2"]


def test_default_estimator_is_roughly_four_chars_per_token():
    assert approximate_tokens(Message.user("x" * 40)) == 14


def test_context_window_without_limits_keeps_everything():
    assert ContextWindow().apply(convo(3)) == convo(3)


def test_context_window_combines_turn_and_token_limits():
    window = ContextWindow(max_turns=3, max_tokens=5, estimator=one_token)
    assert texts(window.apply(convo(5))) == ["be brief", "q3", "a3", "q4", "a4"]


@pytest.mark.parametrize("kwargs", [{"max_turns": 0}, {"max_tokens": 0}])
def test_invalid_limits_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ContextWindow(**kwargs)


class RecordingChatLLM(BaseLLMProvider):
    capabilities = LLMCapabilities(chat=True)

    def __init__(self):
        self.calls = []

    @property
    def name(self):
        return "fake:chat"

    def generate(self, system, user):
        raise AssertionError("v2 chat should be used")

    def chat(self, messages, *, options=None):
        self.calls.append(list(messages))
        return LLMResponse(text="  the summary  ", model=self.name)


