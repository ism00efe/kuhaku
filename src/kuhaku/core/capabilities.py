"""Environment capability detection and resolution of ``"auto"`` settings.

Tool-agnostic, and layered exactly like :mod:`kuhaku.core.config`: this module owns the
generic probes (is a module importable, is a TCP endpoint reachable, what accelerator can
a local ML model use) and the resolution mechanism; a tool contributes its own resolution
chains from its own package -- see :mod:`kuhaku.tools.rag.capabilities` -- never by adding
a branch here. This is the same rule as ``Settings`` vs ``RAGSettings``.

``"auto"`` is the default for every setting whose ideal value depends on what is
installed or reachable at run time. Two hard constraints on resolution:

  - It only ever *downgrades* toward fewer external dependencies. It never turns on a
    model download (the project rule: a default may cost CPU and memory, never a
    download).
  - A concrete value the caller set is absolute -- :func:`resolve` returns it untouched.
    ``KUHAKU_AUTO=false`` freezes every ``"auto"`` setting at its documented baseline
    with no probing at all.

Detection runs once per process, at the point a component is built (there is no cache on
disk: a persisted snapshot goes stale the moment the operator installs a GPU, starts an
Ollama server, or pulls a model, and kuhaku never assumes a writable working directory).
The probes are cheap -- an import-spec lookup, or one short-timeout TCP connect.
"""

from __future__ import annotations

import importlib.util
import os
import socket
import sys
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from urllib.parse import urlparse

from .exceptions import FallbackWarning

# The sentinel a settings field carries when its value should be resolved from the
# environment rather than pinned. Accepted anywhere a concrete value is (see `resolve`).
AUTO = "auto"

_ENV_AUTO = "KUHAKU_AUTO"


def auto_enabled() -> bool:
    """The master switch. ``True`` unless ``KUHAKU_AUTO`` is set to a falsey string
    (``0``/``false``/``no``/``off``, case-insensitive).

    When ``False``, every ``"auto"`` setting resolves straight to its documented
    baseline (``cpu`` device, ``hybrid`` retrieval, ``ollama`` LLM) with no probing --
    the pre-``"auto"`` behavior, for operators who want a fully deterministic startup.
    """

    raw = os.environ.get(_ENV_AUTO)
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off")


def module_available(name: str) -> bool:
    """``True`` if ``import <name>`` would succeed, without actually importing it."""

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def endpoint_reachable(url: str, *, timeout: float = 0.5) -> bool:
    """``True`` if a TCP connection to ``url``'s host/port opens within ``timeout``.

    A liveness probe, not a health check -- enough to tell "there is an Ollama server
    here" from "there is nothing on this port", which is the only distinction the LLM
    provider chain needs. Defaults: port 443 for an ``https`` URL, 80 otherwise.
    """

    parsed = urlparse(url if "://" in url else f"//{url}", scheme="http")
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def torch_accelerator() -> str:
    """The best device a local sentence-transformers model can be placed on:
    ``"cuda"`` (any NVIDIA GPU visible to torch), ``"mps"`` (Apple Silicon), or
    ``"cpu"`` (everything else, including a torch that is not installed at all)."""

    if not module_available("torch"):
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        # A broken/partial torch install must not take the whole process down over a
        # capability probe -- CPU is always a safe answer.
        return "cpu"
    return "cpu"


@dataclass(frozen=True)
class Resolution:
    """The outcome of resolving one ``"auto"`` setting: what was chosen, what the
    fully-provisioned-machine baseline would have been, and why they differ."""

    field: str
    chosen: str
    baseline: str
    reason: str

    @property
    def changed(self) -> bool:
        return self.chosen != self.baseline


_emitted: set[tuple[str, str]] = set()


def emit(resolution: Resolution) -> None:
    """Announce a resolution whose pick differs from the baseline -- once per
    ``(field, chosen)`` per process -- on stderr *and* as a :class:`FallbackWarning`.

    kuhaku never configures logging, so a decision the operator needs to see about
    their own environment goes straight to stderr as one line rather than into a
    logger they may not have wired up.
    """

    if not resolution.changed:
        return
    key = (resolution.field, resolution.chosen)
    if key in _emitted:
        return
    _emitted.add(key)
    message = (
        f"{resolution.field}: using '{resolution.chosen}' "
        f"(baseline '{resolution.baseline}'; {resolution.reason}). "
        f"Pin {resolution.field} explicitly, or set KUHAKU_AUTO=false, to silence this."
    )
    print(f"[kuhaku] {message}", file=sys.stderr)
    warnings.warn(message, FallbackWarning, stacklevel=2)


def reset_emitted() -> None:
    """Clear the process-wide "already announced" set. For tests that assert on the
    notice; not part of the normal runtime path."""

    _emitted.clear()


def resolve(
    field: str,
    configured: str | None,
    *,
    baseline: str,
    candidates: Sequence[tuple[str, Callable[[], bool]]],
    reason_for: Callable[[str], str] | None = None,
) -> str:
    """Resolve one ``"auto"`` setting to a concrete value.

    ``configured`` is what the caller/environment set; ``None`` is treated as ``"auto"``.
    A concrete value is absolute and returned untouched. ``"auto"`` walks ``candidates``
    -- ``(value, probe)`` pairs, in preference order -- and returns the first ``value``
    whose ``probe()`` is truthy, falling back to ``baseline`` when none match. A pick
    other than ``baseline`` is announced via :func:`emit`.
    """

    if configured not in (None, AUTO):
        return configured
    if not auto_enabled():
        return baseline

    chosen = baseline
    reason = "no capability probe matched; using baseline"
    for value, probe in candidates:
        try:
            matched = probe()
        except Exception:
            matched = False
        if matched:
            chosen = value
            reason = reason_for(value) if reason_for is not None else f"detected {value}"
            break

    emit(Resolution(field=field, chosen=chosen, baseline=baseline, reason=reason))
    return chosen
