"""The label set: every label a text can receive, defined once.

The same :class:`Label` objects feed the prompt (name, description, examples), the rule
stage and the check of the model's answer, so what the model is told and what is
accepted from it cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from kuhaku.core.exceptions import ConfigError


@dataclass(frozen=True)
class Label:
    """One label. ``description`` and ``examples`` are shown to the model; a label
    without a description is classified by its name alone, which works only when the
    name says everything."""

    name: str
    description: str = ""
    examples: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigError(f"A label needs a non-empty name; got {self.name!r}.")
        if self.name != self.name.strip():
            raise ConfigError(f"Label name {self.name!r} has leading or trailing whitespace.")
        # A list passed by mistake would make the dataclass unhashable; normalize it.
        object.__setattr__(self, "examples", tuple(self.examples))


class LabelSet:
    """An ordered, duplicate-free collection of :class:`Label` objects.

    Accepts ``Label`` objects or bare names. Names are compared case-insensitively for
    uniqueness, because :meth:`resolve` matches the model's answer that way.
    """

    def __init__(self, labels: Iterable[Label | str]) -> None:
        self._labels: dict[str, Label] = {}
        self._by_key: dict[str, str] = {}
        for item in labels:
            label = item if isinstance(item, Label) else Label(item)
            key = _key(label.name)
            if key in self._by_key:
                raise ConfigError(
                    f"Label '{label.name}' is defined twice (names are compared "
                    f"case-insensitively, as the model's answer is)."
                )
            self._labels[label.name] = label
            self._by_key[key] = label.name
        if not self._labels:
            raise ConfigError("A classifier needs at least one label.")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._labels)

    def __iter__(self) -> Iterator[Label]:
        return iter(self._labels.values())

    def __len__(self) -> int:
        return len(self._labels)

    def __contains__(self, name: object) -> bool:
        return name in self._labels

    def __getitem__(self, name: str) -> Label:
        return self._labels[name]

    def resolve(self, answer: object) -> str | None:
        """The canonical name ``answer`` refers to, or ``None``.

        Tolerates case and surrounding whitespace only -- a model that writes "Billing"
        for "billing" meant it -- but nothing fuzzier: a near-miss name is reported back
        to the model rather than guessed, so the label set stays the only source of
        truth for what counts as a label.
        """

        if not isinstance(answer, str):
            return None
        if answer in self._labels:
            return answer
        return self._by_key.get(_key(answer))


def as_label_set(labels: LabelSet | Iterable[Label | str]) -> LabelSet:
    if isinstance(labels, LabelSet):
        return labels
    if isinstance(labels, str):
        # A bare string is iterable, so without this "spam" would become four labels.
        raise ConfigError("labels must be a sequence of labels, not a single string.")
    return LabelSet(labels)


def _key(name: str) -> str:
    return name.strip().casefold()
