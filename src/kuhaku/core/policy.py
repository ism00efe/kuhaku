"""Default behavior policy: how kuhaku reacts when a configured component fails
to load. Three tiers:

  - **Security-critical** (guard, audit log): configured as the kuhaku default but
    broken -> fail startup loudly (``SecurityComponentError``). Skipped entirely when the
    component is explicitly disabled (the user opted out knowingly).
  - **Performance/helper** (BM25, cross-encoder reranker): configured as the kuhaku
    default but broken -> fall back to a simpler alternative and log a warning, so the
    app still starts. ``Settings.strict_performance_components=True`` upgrades this to a
    loud failure for advanced users who want it.
  - **Custom** (a user-supplied instance, e.g. ``build_retriever(reranker=MyReranker())``):
    validated against its required protocol at injection time and never caught -- a
    broken custom component must fail immediately, not degrade.

This module is intentionally generic (component name + callable + protocol, not a
hardcoded list of components) so wiring in a new component type is a call site, not a
change to the policy engine itself.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from .config import Settings
from .exceptions import CustomComponentError, FallbackWarning, SecurityComponentError
from .security.audit import _DEFAULT_AUDIT_LOG_PATH

logger = logging.getLogger(__name__)

T = TypeVar("T")


def apply_fallback_policy(
    settings: Settings,
    component_name: str,
    build: Callable[[], T],
    *,
    fallback: T,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """Build a "default" performance/helper component, falling back to ``fallback`` (and
    logging + issuing a ``FallbackWarning``) if ``build()`` raises one of ``exceptions``.

    Respects ``settings.strict_performance_components``: when True, the exception
    propagates instead of falling back, for callers who want failures loud.
    """

    try:
        return build()
    except exceptions as exc:
        if settings.strict_performance_components:
            raise
        message = (
            f"Performance component '{component_name}' is set to 'default' but failed "
            f"to load ({exc}); falling back to a simpler alternative."
        )
        logger.warning(message)
        warnings.warn(message, FallbackWarning, stacklevel=2)
        return fallback


def validate_custom_component(
    component: object, protocol: type, *, component_kind: str
) -> None:
    """Fail fast if a user-supplied ``component`` does not satisfy ``protocol``.

    Uses ``isinstance`` against the (``@runtime_checkable``) protocol -- a structural
    match passes even when ``component`` is not a subclass. Raises
    ``CustomComponentError`` naming the missing method(s) otherwise.
    """

    if isinstance(component, protocol):
        return
    required = [name for name in dir(protocol) if not name.startswith("_")]
    missing = [name for name in required if not callable(getattr(component, name, None))]
    detail = ", ".join(missing) if missing else "does not structurally match the protocol"
    raise CustomComponentError(
        f"Custom {component_kind} '{type(component).__name__}' does not satisfy "
        f"{protocol.__name__} protocol: missing method(s) {detail}"
    )


def enforce_guard_policy(settings: Settings, components: dict[str, object]) -> None:
    """Raise ``SecurityComponentError`` if ``guard_enabled=True`` but no working
    ``GuardPipeline`` was supplied. Skipped entirely when ``guard_enabled=False`` (the
    default) -- the user opted out, or never opted in, knowingly.

    Split out from :func:`enforce_security_policy` so a caller can enforce the guard
    piece as fatal -- it can only be True because the caller set it, through this exact
    ``Settings`` field, on purpose -- independently of that function's audit-log check,
    which a caller may want to treat as non-fatal instead (see ``RAG.__init__``, which
    does exactly this: guard misconfiguration raises, an unwritable audit log only
    warns).
    """

    if not settings.guard_enabled:
        return
    guard = components.get("guard")
    if guard is None:
        raise SecurityComponentError(
            "Security component 'GuardPipeline' is set to 'default' "
            "(guard_enabled=True) but was not constructed."
        )
    try:
        guard.validate()
    except Exception as exc:
        raise SecurityComponentError(
            f"Security component 'GuardPipeline' is set to 'default' but failed to "
            f"initialize: {exc}"
        ) from exc


def enforce_security_policy(settings: Settings, components: dict[str, object]) -> None:
    """Raise ``SecurityComponentError`` if an enabled security-critical component is
    missing or fails its self-check. A component whose ``*_enabled`` setting is False is
    skipped entirely -- the user opted out knowingly.
    """

    enforce_guard_policy(settings, components)
    validate_audit_log_path(settings)


def validate_audit_log_path(settings: Settings) -> None:
    """Audit logging is independent of ``guard_enabled`` -- when
    ``settings.audit_enabled`` (default ``True``), every request writes here regardless
    of the guard. There is no dedicated class to construct/validate, only a path every
    request must be able to write to, so this is a writability probe rather than a
    component self-check.

    Skipped entirely when ``audit_enabled`` is ``False`` (the "explicitly disabled ->
    no validation" tier of the policy above) -- the operator opted out knowingly, so
    there is no path to validate.
    """

    if not settings.audit_enabled:
        return

    log_path = settings.audit_log_path or _DEFAULT_AUDIT_LOG_PATH
    path = Path(log_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        probe = path.parent / f".audit_probe_{path.name}"
        probe.touch()
        probe.unlink()
    except OSError as exc:
        raise SecurityComponentError(
            f"Security component 'audit log' at '{log_path}' is not "
            f"writable: {exc}"
        ) from exc


def validate_startup(settings: Settings, components: dict[str, object]) -> dict[str, object]:
    """Single startup entry point wiring Features 1-3 together, meant to be called once
    from whichever composition root the caller uses to build a ``RAGEngine`` -- kuhaku
    ships none itself, so nothing in this package calls this today; it exists for a
    caller who wants the fail-loud startup guarantee described in the module docstring.

    Performance-component fallback (Feature 2) already happened during construction --
    see ``apply_fallback_policy`` usage in ``build_retriever`` -- and custom-component
    validation (Feature 3) already happened at injection time -- see
    ``validate_custom_component`` usage in ``build_retriever``. This only enforces the
    security policy (Feature 1) and returns ``components`` unchanged.
    """

    enforce_security_policy(settings, components)
    return components
