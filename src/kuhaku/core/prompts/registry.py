"""``PromptRegistry``: named, versioned prompt templates, with user overrides layered
over a tool's built-in defaults.

A tool ships its prompts in its own registry and looks them up through
``PromptRegistry.chain(user_registry, builtin_registry)``: whatever the user registered
under a name wins, every other name falls through to the tool's defaults, and nothing
in kuhaku has to be edited to change a prompt. The chain is live -- a template
registered on the user's registry after the tool was built is seen by the next lookup.

"Latest" is decided by comparing versions, not by registration order, so registering
the same templates in any order gives the same answer:
numeric runs compare as numbers (``v10`` > ``v9``, ``1.10`` > ``1.9``). :meth:`pin`
overrides that for a name -- e.g. to keep production on ``v2`` while ``v3`` is
registered for evaluation.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterable, Mapping

from .template import (
    PromptConflictError,
    PromptNotFoundError,
    PromptTemplate,
    RenderedPrompt,
)

_VERSION_PART_RE = re.compile(r"\d+|\D+")
# "v2", "2.0.1", "2.0.0-rc1", "1.4b2", "3.dev1", "1-custom". A single-letter pre-release
# marker needs its number ("1a" is a variant of 1, not an alpha) -- the words do not.
_VERSION_RE = re.compile(
    r"^[vV]?(?P<release>\d+(?:\.\d+)*)"
    r"(?:[-_.]?(?:(?P<word>dev|alpha|beta|pre|preview|rc)|(?P<letter>[abc])(?=\d))"
    r"[-_.]?(?P<num>\d*))?"
    r"(?P<rest>.*)$",
    re.IGNORECASE,
)
_STAGES = {"dev": 0, "alpha": 1, "a": 1, "beta": 2, "b": 2, "pre": 3, "preview": 3,
           "rc": 3, "c": 3}
_FINAL = 4

VersionKey = tuple[object, ...]


def _natural(text: str) -> tuple[tuple[int, int | str], ...]:
    # Tagged so a number and a word never get compared directly (which would raise).
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in _VERSION_PART_RE.findall(text)
    )


def version_key(version: str) -> VersionKey:
    """Sort key: semver/PEP 440-like ordering, then the raw string as a tie-break.

    - an optional leading ``v`` is ignored (``v2`` == ``2``), the release compares
      numerically part by part, trailing zeros ignored (``1.10`` > ``1.9``);
    - a pre-release sorts before its final release (``2.0.0-rc1`` < ``2.0.0``), in the
      order dev < alpha/a < beta/b < rc/pre;
    - any other suffix is a variant after the release (``1`` < ``1-custom``);
    - versions that do not start with a number sort before all that do, naturally.

    Two spellings of one version (``01`` and ``1``, ``2`` and ``2.0``) are ordered by
    their raw text, so "latest" never depends on registration order.
    """

    match = _VERSION_RE.match(version)
    if match is None:
        return (0, _natural(version), version)
    release = [int(part) for part in match["release"].split(".")]
    while len(release) > 1 and release[-1] == 0:
        release.pop()
    marker = (match["word"] or match["letter"] or "").lower()
    if marker:
        stage, number = _STAGES[marker], int(match["num"] or 0)
    else:
        stage, number = _FINAL, 0
    return (1, tuple(release), stage, number, _natural(match["rest"]), version)


class PromptRegistry:
    """Templates by name and version.

    ``fallbacks`` are consulted, in order, for a name this registry does not hold
    itself (or for an explicit version it does not hold). Registering always goes into
    this registry, never into a fallback.
    """

    def __init__(
        self,
        templates: Iterable[PromptTemplate] = (),
        *,
        fallbacks: Iterable[PromptRegistry] = (),
    ) -> None:
        self._templates: dict[str, dict[str, PromptTemplate]] = {}
        self._pins: dict[str, str] = {}
        self._fallbacks: tuple[PromptRegistry, ...] = tuple(fallbacks)
        # Re-entrant: get() resolves a pin and then looks the pinned version up.
        self._lock = threading.RLock()
        for template in templates:
            self.register(template)

    @classmethod
    def chain(cls, *registries: PromptRegistry | None) -> PromptRegistry:
        """A registry that looks names up in ``registries`` in order (``None`` skipped).

        The usual tool pattern: ``PromptRegistry.chain(user_prompts, DEFAULT_PROMPTS)``.
        """

        return cls(fallbacks=[r for r in registries if r is not None])

    # -- writing ---------------------------------------------------------------------

    def register(self, template: PromptTemplate, *, replace: bool = False) -> PromptTemplate:
        """Add ``template``. Registering the same name, version and text again is a
        no-op; a *different* text under an existing name and version raises
        :class:`PromptConflictError` unless ``replace=True`` -- a version label must
        keep meaning one text, or the recorded version stops identifying the prompt."""

        if not isinstance(template, PromptTemplate):
            raise TypeError(f"expected a PromptTemplate; got {type(template).__name__}")
        with self._lock:
            versions = self._templates.setdefault(template.name, {})
            existing = versions.get(template.version)
            if existing is not None and not replace:
                if existing.content_hash == template.content_hash:
                    return existing
                raise PromptConflictError(
                    f"Prompt '{template.name}' version '{template.version}' is already "
                    f"registered with different text. Register it under a new version, "
                    f"or pass replace=True."
                )
            versions[template.version] = template
            return template

    def pin(self, name: str, version: str) -> None:
        """Make ``get(name)`` return ``version`` instead of the latest. The version must
        already be resolvable here (in this registry or a fallback)."""

        with self._lock:
            self.get(name, version)
            self._pins[name] = version

    def unpin(self, name: str) -> None:
        with self._lock:
            self._pins.pop(name, None)

    # -- reading ---------------------------------------------------------------------

    def get(self, name: str, version: str | None = None) -> PromptTemplate:
        """The template ``name`` at ``version`` -- or, when ``version`` is ``None``, the
        pinned version if there is one, else the latest."""

        with self._lock:
            if version is None:
                version = self._pinned(name)
            if version is not None:
                found = self._find(name, version)
                if found is None:
                    raise PromptNotFoundError(
                        f"No prompt '{name}' version '{version}' (registered versions: "
                        f"{list(self.versions(name))})."
                    )
                return found
            latest = self._latest(name)
            if latest is None:
                raise PromptNotFoundError(
                    f"No prompt named '{name}' (registered: {list(self.names())})."
                )
            return latest

    def render(
        self,
        name: str,
        values: Mapping[str, object] | None = None,
        /,
        *,
        version: str | None = None,
        **variables: object,
    ) -> RenderedPrompt:
        """Look up and render in one call; the result carries the template's ref."""

        return self.get(name, version).render_prompt(values, **variables)

    def versions(self, name: str) -> tuple[str, ...]:
        """Every version resolvable for ``name``, oldest first."""

        with self._lock:
            found = set(self._templates.get(name, {}))
            for fallback in self._fallbacks:
                found.update(fallback.versions(name))
            return tuple(sorted(found, key=version_key))

    def names(self) -> tuple[str, ...]:
        with self._lock:
            found = {name for name, versions in self._templates.items() if versions}
            for fallback in self._fallbacks:
                found.update(fallback.names())
            return tuple(sorted(found))

    def templates(self) -> list[PromptTemplate]:
        """Every resolvable template, by name then version. Where this registry and a
        fallback both hold the same name and version, this registry's wins."""

        return [self.get(name, version) for name in self.names() for version in self.versions(name)]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self.names()

    def __len__(self) -> int:
        return len(self.templates())

    def __repr__(self) -> str:
        return f"PromptRegistry(names={list(self.names())})"

    # -- resolution ------------------------------------------------------------------

    def _pinned(self, name: str) -> str | None:
        if name in self._pins:
            return self._pins[name]
        # A pin made on a fallback still applies through a chain over it, unless this
        # registry holds the name itself (then its own latest is the answer).
        if self._templates.get(name):
            return None
        for fallback in self._fallbacks:
            pinned = fallback._pinned(name)
            if pinned is not None:
                return pinned
        return None

    def _find(self, name: str, version: str) -> PromptTemplate | None:
        found = self._templates.get(name, {}).get(version)
        if found is not None:
            return found
        for fallback in self._fallbacks:
            found = fallback._find(name, version)
            if found is not None:
                return found
        return None

    def _latest(self, name: str) -> PromptTemplate | None:
        own = self._templates.get(name)
        if own:
            return own[max(own, key=version_key)]
        for fallback in self._fallbacks:
            if name in fallback:
                return fallback.get(name)
        return None
