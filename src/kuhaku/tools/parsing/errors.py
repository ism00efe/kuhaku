"""Errors the parsing tool raises on purpose. All derive from ``KuhakuError``."""

from __future__ import annotations

import importlib
from types import ModuleType

from kuhaku.core.exceptions import KuhakuError


class ParsingError(KuhakuError):
    """The content could not be parsed as the format it was routed to."""


class UnsupportedFormat(ParsingError, ValueError):
    """No registered parser handles this file type."""


class DocumentTooLarge(ParsingError):
    """The content exceeds the parser's ``max_bytes`` limit."""


class MissingDependency(KuhakuError, ImportError):
    """An optional library a parser needs is not installed."""


def require(module: str, *, extra: str, purpose: str) -> ModuleType:
    """Import ``module`` or raise :class:`MissingDependency` naming the extra to install.

    Every heavy library is imported through here, at the moment it is needed, so
    importing ``kuhaku.tools.parsing`` never pulls in a parser backend the caller does
    not use.
    """

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingDependency(
            f"{purpose} needs the '{module}' package, which is not installed. "
            f"Install it with: pip install \"kuhaku[{extra}]\""
        ) from exc


def is_installed(module: str) -> bool:
    try:
        importlib.import_module(module)
    except ImportError:
        return False
    return True
