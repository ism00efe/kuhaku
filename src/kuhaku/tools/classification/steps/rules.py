"""Optional step: deterministic rules that can decide before -- and instead of -- the model."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

from kuhaku.core.exceptions import ConfigError, CustomComponentError

from ..context import ClassificationContext
from ..pipeline import Step

RULES_STEP = "rules"

Rule = Callable[[ClassificationContext], "str | Iterable[str] | None"]
"""``ctx -> label name(s), or None when the rule does not apply``."""


class KeywordRule:
    """Matches when any of ``keywords`` occurs in the text.

    Case-insensitive by default, and on word boundaries so "refund" does not fire on
    "refunded" unless that is listed too -- a rule that decides without a model has to
    be the strict kind.
    """

    def __init__(
        self,
        label: str,
        keywords: Iterable[str],
        *,
        case_sensitive: bool = False,
        whole_word: bool = True,
    ) -> None:
        words = [keywords] if isinstance(keywords, str) else list(keywords)
        if not words or not all(isinstance(word, str) and word for word in words):
            raise ConfigError(f"KeywordRule('{label}') needs at least one non-empty keyword.")
        self.label = label
        self.keywords = tuple(words)
        alternatives = "|".join(re.escape(word) for word in words)
        pattern = rf"(?<!\w)(?:{alternatives})(?!\w)" if whole_word else alternatives
        self._pattern = re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)

    def __call__(self, ctx: ClassificationContext) -> str | None:
        match = self._pattern.search(ctx.text)
        return self.label if match else None

    def describe(self, ctx: ClassificationContext) -> str:
        match = self._pattern.search(ctx.text)
        return f"keyword {match.group(0)!r} matched" if match else "no keyword matched"

    def __repr__(self) -> str:
        return f"KeywordRule({self.label!r}, {list(self.keywords)!r})"


class RegexRule:
    """Matches when ``pattern`` is found in the text (``re.search``)."""

    def __init__(self, label: str, pattern: str | re.Pattern[str], *, flags: int = 0) -> None:
        self.label = label
        self._pattern = pattern if isinstance(pattern, re.Pattern) else re.compile(pattern, flags)

    def __call__(self, ctx: ClassificationContext) -> str | None:
        return self.label if self._pattern.search(ctx.text) else None

    def describe(self, ctx: ClassificationContext) -> str:
        match = self._pattern.search(ctx.text)
        return f"pattern {self._pattern.pattern!r} matched {match.group(0)!r}" if match else ""

    def __repr__(self) -> str:
        return f"RegexRule({self.label!r}, {self._pattern.pattern!r})"


class RuleStep(Step):
    """Runs every rule against the text; when they agree, decides without the model.

    Single-label mode: if the rules that fired all name the same label, that label is
    the decision. If they name different labels, the step does not pick one -- the
    disagreement is recorded in ``metadata["rule_conflict"]`` and the model decides,
    so the order rules happen to be listed in never becomes a hidden priority.

    Multi-label mode: the union of every fired rule's labels is the decision, and the
    model is not called -- so a rule fires only for a text whose labels it fully
    determines. Leave rules out where the model should add labels on top.

    A rule decision scores each label ``1.0``. Rules that fire on nothing leave the text
    to the next step. A plain callable ``ctx -> label | labels | None`` is a rule too.
    """

    name = RULES_STEP

    def __init__(self, *rules: Rule) -> None:
        for rule in rules:
            if not callable(rule):
                raise CustomComponentError(f"Rule {rule!r} is not callable.")
        self.rules = list(rules)

    def prepare(self, ctx: ClassificationContext) -> None:
        # Caught here rather than on the first text that happens to trigger the rule.
        for rule in self.rules:
            label = getattr(rule, "label", None)
            if isinstance(label, str) and label not in ctx.labels:
                raise ConfigError(
                    f"{rule!r} names label '{label}', which is not in the label set "
                    f"{list(ctx.labels.names)}."
                )

    def run(self, ctx: ClassificationContext) -> None:
        if ctx.settled:
            return
        fired: dict[str, list[str]] = {}
        for rule in self.rules:
            for name in _names(rule, rule(ctx)):
                if name not in ctx.labels:
                    raise CustomComponentError(
                        f"Rule {rule!r} returned '{name}', which is not in the label set "
                        f"{list(ctx.labels.names)}."
                    )
                fired.setdefault(name, []).append(_describe(rule, ctx))
        if not fired:
            return

        if not ctx.multi_label and len(fired) > 1:
            ctx.metadata["rule_conflict"] = sorted(fired)
            return

        ctx.decide(
            fired,
            by=self.name,
            reason="; ".join(f"{name}: {', '.join(why)}" for name, why in fired.items()),
            scores={name: 1.0 for name in fired},
        )


def _names(rule: Rule, returned: Any) -> list[str]:
    if returned is None:
        return []
    if isinstance(returned, str):
        return [returned]
    try:
        names = list(returned)
    except TypeError:
        raise CustomComponentError(
            f"Rule {rule!r} returned {returned!r}; expected a label name, names, or None."
        ) from None
    if not all(isinstance(name, str) for name in names):
        raise CustomComponentError(f"Rule {rule!r} returned non-string labels {names!r}.")
    return names


def _describe(rule: Rule, ctx: ClassificationContext) -> str:
    describe = getattr(rule, "describe", None)
    if callable(describe):
        return str(describe(ctx))
    return f"rule {getattr(rule, '__name__', None) or type(rule).__name__} matched"
