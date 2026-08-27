"""Hardware-capability probing for local ML components (embeddings, reranker).

Resolves an abstract device request ("auto", "cpu", "cuda", "mps") to a concrete
device string. kuhaku itself never depends on torch -- it is only pulled in
transitively by sentence-transformers -- so probing imports it lazily and degrades to
CPU whenever it, or the requested backend, isn't actually there. This is what lets the
same default config run unmodified on a bare CPU laptop, an NVIDIA/CUDA box, and
Apple Silicon (MPS).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_AUTO_PROBED = ("auto", "cuda", "mps")


def resolve_device(requested: str, *, component: str) -> str:
    """Resolve ``requested`` to a concrete device string for the given component.

    * ``"cpu"`` is always returned as-is (no probing needed).
    * ``"auto"`` picks the first available accelerator, in order CUDA, then MPS
      (Apple Silicon), else CPU.
    * ``"cuda"``/``"mps"`` are honored if available; otherwise this falls back to CPU
      and logs a warning naming ``component``, rather than letting torch/
      sentence-transformers raise an opaque error deep inside model loading.
    * Anything else (e.g. an explicit ``"cuda:1"``) is passed through unchanged --
      the caller knows exactly what they asked for.
    """

    normalized = (requested or "auto").strip().lower()
    if normalized == "cpu":
        return "cpu"
    if normalized not in _AUTO_PROBED:
        return requested

    try:
        import torch
    except ImportError:
        if normalized != "auto":
            logger.warning(
                "%s requested device '%s' but torch is not installed; using CPU instead.",
                component,
                normalized,
            )
        return "cpu"

    if normalized in ("auto", "cuda") and torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if normalized in ("auto", "mps") and mps_backend is not None and mps_backend.is_available():
        return "mps"

    if normalized != "auto":
        logger.warning(
            "%s requested device '%s' but it is not available on this machine; using CPU instead.",
            component,
            normalized,
        )
    return "cpu"


__all__ = ["resolve_device"]
