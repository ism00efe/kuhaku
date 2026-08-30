"""The ``"auto"`` sentinel and the ``KUHAKU_AUTO`` master switch.

``resolve()`` consults :func:`auto_enabled` *before* any probing: when it returns
``False`` every decision resolves straight to its documented baseline -- no probe, no
question, no install, no decision memory -- because determinism is the whole point of the
switch. See :mod:`kuhaku.core.resolve.resolver`.
"""

from __future__ import annotations

import os

# The value a settings field carries when it should be resolved from the environment
# rather than pinned. Accepted anywhere a concrete value is.
AUTO = "auto"

_ENV_AUTO = "KUHAKU_AUTO"


def auto_enabled() -> bool:
    """``True`` unless ``KUHAKU_AUTO`` is set to a falsey string (``0``/``false``/``no``/
    ``off``, case-insensitive)."""

    raw = os.environ.get(_ENV_AUTO)
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off")
