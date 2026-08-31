"""The probed environment, and the fingerprint that keys the decision memory.

``FIELDS_CONSUMED`` is what makes per-field invalidation possible: a stored decision is
still valid when the fingerprint fields *it* consumed are unchanged, regardless of the
rest. Installing or removing a package changes ``packages`` and re-opens exactly the
decisions that read it; plugging in a GPU changes ``gpu``/``vram_class`` and re-opens
only the device decision.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from .probes import detect_isolation, gpu_kind, hf_cache_roots, vram_class


@dataclass(frozen=True)
class Environment:
    python: str
    in_isolated_python: bool
    isolation_source: Literal["venv", "conda", "prefix"] | None
    gpu: Literal["cuda", "mps", "rocm"] | None
    vram_class: Literal["none", "small", "medium", "large", "unknown"]
    os: str


@dataclass(frozen=True)
class Fingerprint:
    python: str
    packages: Mapping[str, str | None]
    gpu: str | None
    vram_class: str | None
    model_dir: str | None


# decision kind -> the fingerprint fields that decision depends on
FIELDS_CONSUMED: Mapping[str, frozenset[str]] = {
    "device": frozenset({"gpu", "vram_class"}),
    "llm": frozenset({"packages"}),
    "embedding": frozenset({"packages", "model_dir"}),
    "retrieval": frozenset({"packages", "model_dir"}),
    "store": frozenset({"packages"}),
}


def probe_environment() -> Environment:
    """Run every cheap probe once and freeze the result."""

    isolated, source = detect_isolation()
    gpu = gpu_kind()
    return Environment(
        python=f"{sys.version_info.major}.{sys.version_info.minor}",
        in_isolated_python=isolated,
        isolation_source=source,
        gpu=gpu,
        vram_class=vram_class(gpu),
        os=platform.system().lower(),
    )


def _model_dir_digest() -> str | None:
    """A digest of the model caches -- entry names plus mtimes across every location
    :func:`hf_cache_roots` knows about. Changes when a model is pulled or removed;
    ``None`` when no cache directory exists. Kept in step with the embedding adapter's
    ``_model_cached`` so installing a model in any of those locations re-opens the
    decisions that depend on it."""

    entries: list[str] = []
    found = False
    for root in hf_cache_roots():
        if not root.is_dir():
            continue
        found = True
        try:
            entries.extend(f"{root}/{p.name}:{int(p.stat().st_mtime)}" for p in root.iterdir())
        except OSError:
            continue
    if not found:
        return None
    return hashlib.sha256("\n".join(sorted(entries)).encode()).hexdigest()[:16]


def fingerprint(env: Environment, *, packages: Iterable[str] = ()) -> Fingerprint:
    """Build the fingerprint for ``env``. ``packages`` is the set of distribution names
    the registry's adapters declare they depend on; each is resolved to its installed
    version or ``None``."""

    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        version = None  # type: ignore[assignment]

    pkg_versions: dict[str, str | None] = {}
    for name in sorted(set(packages)):
        if version is None:
            pkg_versions[name] = None
            continue
        try:
            pkg_versions[name] = version(name)
        except PackageNotFoundError:
            pkg_versions[name] = None
        except Exception:  # pragma: no cover - defensive
            pkg_versions[name] = None

    return Fingerprint(
        python=env.python,
        packages=pkg_versions,
        gpu=env.gpu,
        vram_class=env.vram_class,
        model_dir=_model_dir_digest(),
    )


def project(fp: Fingerprint, fields: frozenset[str]) -> dict:
    """The subset of ``fp`` a decision consumes, as a JSON-comparable dict."""

    values = {
        "python": fp.python,
        "packages": dict(fp.packages),
        "gpu": fp.gpu,
        "vram_class": fp.vram_class,
        "model_dir": fp.model_dir,
    }
    return {name: values[name] for name in sorted(fields)}
