"""``PromptTemplate``: one named, versioned prompt text with declared variables.

Placeholders are ``{{name}}`` (whitespace inside the braces allowed). Not ``str.format``'s
single braces: prompts routinely show JSON, and every literal ``{`` would otherwise need
escaping -- a mistake that only shows up at render time. Not ``string.Template``'s
``$name`` either, because the RAG system prompt file (``tools/rag/prompts/
system_prompt.txt``) already uses ``{{NAME}}`` markers and is edited by hand outside
this code.

Rendering is a single left-to-right ``re.sub`` pass, so a value that itself contains
``{{...}}`` (a crafted persona, a source text) is written out literally and never
substituted a second time.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from ..exceptions import KuhakuError

PLACEHOLDER_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")

# Names and versions end up in span attributes, audit records and file names, so they
# are restricted to characters that are unambiguous in all three.
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9_.\-/]+")
_VARIABLE_RE = re.compile(r"\w+")

CUSTOM_VERSION = "custom"
"""The version :meth:`PromptRef.for_text` reports for prompt text that came from no
registered template (e.g. a whole system prompt passed in as a string)."""


class PromptError(KuhakuError):
    """Root of the prompt registry's errors."""


class PromptTemplateError(PromptError, ValueError):
    """A template is malformed: bad name/version, or declared variables that do not
    match the placeholders in its text."""


class PromptRenderError(PromptError, ValueError):
    """Rendering was given a missing, unexpected or ``None`` variable."""


class PromptNotFoundError(PromptError, LookupError):
    """No template is registered under the requested name (and version)."""


class PromptConflictError(PromptError):
    """A different text is already registered under the same name and version."""


def content_hash(text: str) -> str:
    """SHA-256 hex digest of ``text`` -- what identifies a template's content."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def placeholders(text: str) -> frozenset[str]:
    """Every ``{{name}}`` placeholder in ``text``."""

    return frozenset(PLACEHOLDER_RE.findall(text))


@dataclass(frozen=True)
class PromptRef:
    """Which prompt produced an LLM call: name, version and content hash.

    This is what gets recorded -- on spans, in audit records, in tool results -- never
    the prompt text itself, which may embed user data.
    """

    name: str
    version: str
    content_hash: str

    @classmethod
    def for_text(cls, name: str, text: str, version: str = CUSTOM_VERSION) -> PromptRef:
        """A ref for prompt text that did not come from a registered template."""

        return cls(name=name, version=version, content_hash=content_hash(text))

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version, "hash": self.content_hash}


@dataclass(frozen=True)
class RenderedPrompt:
    """Rendered text together with the ref of the template it came from."""

    text: str
    ref: PromptRef


@dataclass(frozen=True)
class PromptTemplate:
    """A named, versioned prompt.

    ``variables`` declares the placeholders the text must contain; when given, a
    mismatch with the text (either direction) is a :class:`PromptTemplateError` at
    construction, so a typo in a hand-edited prompt fails when it is loaded rather than
    when it is first used. When omitted, it is inferred from the text.

    ``content_hash`` is computed from ``template`` alone: two versions with identical
    text share a hash, which is how a version bump that changed nothing shows up.
    """

    name: str
    version: str
    template: str
    # Always a frozenset after construction; None only means "infer from the text".
    variables: frozenset[str] = None  # type: ignore[assignment]
    description: str = ""
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for label, value in (("name", self.name), ("version", self.version)):
            if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
                raise PromptTemplateError(
                    f"Prompt {label} must be a non-empty string of letters, digits and "
                    f"'_.-/'; got {value!r}."
                )
        if not isinstance(self.template, str):
            raise PromptTemplateError(
                f"Prompt '{self.name}' template must be a str; got "
                f"{type(self.template).__name__}."
            )
        found = placeholders(self.template)
        if self.variables is None:
            declared = found
        else:
            declared = _as_variable_set(self.name, self.variables)
            undeclared = sorted(found - declared)
            unused = sorted(declared - found)
            if undeclared or unused:
                problems = []
                if undeclared:
                    problems.append(f"placeholders not declared as variables: {undeclared}")
                if unused:
                    problems.append(f"declared variables with no placeholder: {unused}")
                raise PromptTemplateError(
                    f"Prompt '{self.name}' version '{self.version}': " + "; ".join(problems)
                )
        object.__setattr__(self, "variables", declared)
        object.__setattr__(self, "content_hash", content_hash(self.template))

    @property
    def ref(self) -> PromptRef:
        return PromptRef(self.name, self.version, self.content_hash)

    def render(self, values: Mapping[str, object] | None = None, /, **variables: object) -> str:
        """Substitute every placeholder. A missing or unexpected variable, or a
        ``None`` value, is a :class:`PromptRenderError` -- a prompt with a silently
        empty or literal ``{{name}}`` slot is worse than no prompt."""

        supplied = {**(values or {}), **variables}
        missing = sorted(self.variables - supplied.keys())
        extra = sorted(supplied.keys() - self.variables)
        if missing or extra:
            problems = []
            if missing:
                problems.append(f"missing variables {missing}")
            if extra:
                problems.append(f"unexpected variables {extra}")
            raise PromptRenderError(
                f"Rendering prompt '{self.name}' version '{self.version}': "
                + "; ".join(problems)
                + f" (declared: {sorted(self.variables)})."
            )
        none_valued = sorted(name for name, value in supplied.items() if value is None)
        if none_valued:
            raise PromptRenderError(
                f"Rendering prompt '{self.name}' version '{self.version}': variables "
                f"{none_valued} are None. Pass an empty string to leave a slot empty."
            )
        rendered = {name: str(value) for name, value in supplied.items()}
        return PLACEHOLDER_RE.sub(lambda match: rendered[match.group(1)], self.template)

    def render_prompt(
        self, values: Mapping[str, object] | None = None, /, **variables: object
    ) -> RenderedPrompt:
        """:meth:`render`, keeping the template's :attr:`ref` alongside the text."""

        return RenderedPrompt(self.render(values, **variables), self.ref)


def _as_variable_set(name: str, variables: Iterable[str]) -> frozenset[str]:
    if isinstance(variables, str):
        raise PromptTemplateError(
            f"Prompt '{name}' variables must be a collection of names, not a single string."
        )
    declared = frozenset(variables)
    bad = sorted(
        repr(v) for v in declared if not isinstance(v, str) or not _VARIABLE_RE.fullmatch(v)
    )
    if bad:
        raise PromptTemplateError(f"Prompt '{name}' has invalid variable names: {bad}.")
    return declared
