"""The UI port: where "is a human there", "announce", "ask" and "confirm" are defined.

The default :class:`ConsoleUI` is the *only* place in kuhaku that tests for a terminal.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from collections.abc import Hashable, Sequence
from typing import Protocol

from ..exceptions import FallbackWarning
from .cost import Candidate, Cost

_LOGGER_NAME = "kuhaku"


class UI(Protocol):
    def is_interactive(self) -> bool: ...

    def announce(
        self,
        message: str,
        *,
        prominent: bool = False,
        degraded: bool = False,
        dedupe_key: Hashable | None = None,
    ) -> None: ...

    def ask(self, question: str, options: Sequence[Candidate]) -> Candidate | None: ...

    def confirm(self, action: str, cost: Cost) -> bool: ...


class ConsoleUI:
    """stdin/stderr implementation.

    - ``is_interactive`` -> ``sys.stdin.isatty() and sys.stderr.isatty()``, and ``False``
      if ``CI`` or ``KUHAKU_NONINTERACTIVE`` is set. The single definition; no other
      module tests for a terminal.
    - ``announce`` -> the ``kuhaku`` logger at INFO (``prominent`` -> WARNING); never a
      direct write to stdout. If no handler is attached to that logger or any ancestor,
      it attaches a plain stderr handler once, so a decision the operator needs to see is
      never lost to a host application that configured no logging. A host that wants
      silence attaches its own handler (a ``logging.NullHandler`` on ``"kuhaku"`` is
      enough) -- the fallback is not added when one already exists.
    - ``ask`` / ``confirm`` -> ``None`` / ``False`` when not interactive; never blocks
      there. Consent is never inferred.

    Announcement de-duplication is per instance (keyed on whatever ``dedupe_key`` the
    caller passes -- the resolver uses ``(kind, candidate_id, reason)``), so a fresh
    ``ConsoleUI`` starts clean. Real usage goes through :func:`default_ui`, a
    process-wide singleton, so an identical decision is announced once per process; a
    caller that injects its own ``ConsoleUI`` (or a test) is isolated.
    """

    def __init__(self) -> None:
        self._announced: set[Hashable] = set()
        self._fallback_handler_attached = False

    # -- port ---------------------------------------------------------------
    def is_interactive(self) -> bool:
        if os.environ.get("CI") or os.environ.get("KUHAKU_NONINTERACTIVE"):
            return False
        try:
            return bool(sys.stdin.isatty() and sys.stderr.isatty())
        except (AttributeError, ValueError):
            return False

    def announce(
        self,
        message: str,
        *,
        prominent: bool = False,
        degraded: bool = False,
        dedupe_key: Hashable | None = None,
    ) -> None:
        if dedupe_key is not None:
            if dedupe_key in self._announced:
                return
            self._announced.add(dedupe_key)

        logger = logging.getLogger(_LOGGER_NAME)
        self._ensure_visible(logger)
        logger.warning("%s", message) if prominent else logger.info("%s", message)
        if degraded:
            warnings.warn(message, FallbackWarning, stacklevel=2)

    def ask(self, question: str, options: Sequence[Candidate]) -> Candidate | None:
        options = list(options)
        if not self.is_interactive() or not options:
            return None
        sys.stderr.write(question.rstrip() + "\n")
        for i, option in enumerate(options, 1):
            sys.stderr.write(f"  {i}) {option.label}\n")
        sys.stderr.write(f"Choose 1-{len(options)} (Enter to skip): ")
        sys.stderr.flush()
        try:
            raw = input().strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not raw.isdigit() or not (1 <= int(raw) <= len(options)):
            return None
        return options[int(raw) - 1]

    def confirm(self, action: str, cost: Cost) -> bool:
        if not self.is_interactive():
            return False
        sys.stderr.write(f"{action}\n  ({cost.note})\nProceed? [y/N]: ")
        sys.stderr.flush()
        try:
            return input().strip().lower() in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    # -- internals --------------------------------------------------------
    def _ensure_visible(self, logger: logging.Logger) -> None:
        if self._fallback_handler_attached:
            return
        node: logging.Logger | None = logger
        while node is not None:
            if node.handlers:
                self._fallback_handler_attached = True  # someone is listening; leave it
                return
            if not node.propagate:
                break
            node = node.parent
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[kuhaku] %(message)s"))
        logger.addHandler(handler)
        if logger.level == logging.NOTSET:
            logger.setLevel(logging.INFO)
        self._fallback_handler_attached = True


_default_ui: ConsoleUI | None = None


def default_ui() -> ConsoleUI:
    """The process-wide :class:`ConsoleUI`. ``build_llm_provider`` and ``RAG.__init__``
    use this when no UI is injected, so an identical decision (and the "auto disabled"
    line) is announced once per process rather than once per call site."""

    global _default_ui
    if _default_ui is None:
        _default_ui = ConsoleUI()
    return _default_ui


def _reset_default_ui() -> None:
    """Test hook -- drop the singleton so announcement-dedupe state does not leak
    between tests."""

    global _default_ui
    _default_ui = None
