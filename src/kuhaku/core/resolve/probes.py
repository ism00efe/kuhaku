"""Cheap, tool-agnostic environment probes.

Every function here is safe to call during ``Adapter.probe``: none installs, none
downloads, none raises, and the only network they do is a loopback TCP connect to a
fixed port to detect a locally running daemon (:func:`loopback_daemon_reachable`) -- no
DNS, no proxy, sub-``0.25s`` -- which is the honest way to tell "a server is running
here" from "nothing on this port".
"""

from __future__ import annotations

import importlib.util
import ipaddress
import os
import shutil
import socket
import sys
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

_LOOPBACK_NAMES = {"localhost", "ip6-localhost"}


def module_available(name: str) -> bool:
    """``True`` if ``import <name>`` would succeed, without importing it."""

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def endpoint_reachable(url: str, *, timeout: float = 0.5) -> bool:
    """``True`` if a TCP connection to ``url``'s host/port opens within ``timeout``."""

    parsed = urlparse(url if "://" in url else f"//{url}", scheme="http")
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _is_loopback(host: str) -> bool:
    if host in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def loopback_daemon_reachable(url: str, *, timeout: float = 0.25) -> bool:
    """``True`` if something is listening on ``url``'s port, and ``url``'s host is a
    loopback address or ``localhost``.

    Detecting a locally running daemon is exactly the case §14's
    "no outbound network call to decide whether a local resource is available" allows: a
    loopback connect resolves no name, traverses no proxy, and never leaves the machine.
    A non-loopback host returns ``False`` here -- use :func:`endpoint_reachable` for those.
    Never raises; ``timeout`` is capped at ``0.25s``.
    """

    parsed = urlparse(url if "://" in url else f"//{url}", scheme="http")
    host = parsed.hostname or "localhost"
    if not _is_loopback(host):
        return False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=min(timeout, 0.25)):
            return True
    except OSError:
        return False


def torch_accelerator() -> Literal["cuda", "mps", "cpu"]:
    """The best device a local ML model can be placed on: ``"cuda"`` (any NVIDIA GPU
    torch can see), ``"mps"`` (Apple Silicon), or ``"cpu"`` (everything else, torch
    absent included). Never raises."""

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
        return "cpu"
    return "cpu"


def gpu_kind() -> Literal["cuda", "mps", "rocm"] | None:
    """The GPU family visible to a local ML model, or ``None``. ``"rocm"`` is reported
    only when torch itself says it is a HIP build -- there is no reliable no-torch probe
    for AMD."""

    dev = torch_accelerator()
    if dev in ("cuda", "mps"):
        # torch reports ROCm through the CUDA API surface; disambiguate by version tag.
        try:
            import torch

            if dev == "cuda" and getattr(torch.version, "hip", None):
                return "rocm"
        except Exception:
            pass
        return dev  # type: ignore[return-value]
    return None


def nvidia_smi_present() -> bool:
    """``True`` if the ``nvidia-smi`` binary is on ``PATH`` -- the install-time hardware
    check (§9 moment 1) that decides whether a CUDA build variant should even be offered,
    without importing torch."""

    return shutil.which("nvidia-smi") is not None


def vram_class(gpu: str | None) -> Literal["none", "small", "medium", "large", "unknown"]:
    """Map visible VRAM to a coarse size class. ``"none"`` when there is no GPU;
    ``"unknown"`` when there is one but its VRAM cannot be read (Apple Silicon shares
    memory; AMD reporting is unreliable). The thresholds are estimates, not tuned values.
    """

    if gpu is None:
        return "none"
    try:
        import torch

        if gpu == "cuda" and torch.cuda.is_available():
            gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if gib < 6:
                return "small"
            if gib < 16:
                return "medium"
            return "large"
    except Exception:
        return "unknown"
    return "unknown"


def hf_cache_roots() -> list[Path]:
    """Every directory a HuggingFace / sentence-transformers model may be cached under,
    in priority order. Covers the env vars real deployments set (`HF_HUB_CACHE`,
    `HUGGINGFACE_HUB_CACHE`, `HF_HOME`, `SENTENCE_TRANSFORMERS_HOME`) plus the default
    layout. Existence is not checked here."""

    roots: list[Path] = []
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.environ.get(var)
        if value:
            roots.append(Path(value))
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home) / "hub")
    st_home = os.environ.get("SENTENCE_TRANSFORMERS_HOME")
    if st_home:
        roots.append(Path(st_home))
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    # de-dupe, keep order
    seen: set[str] = set()
    unique: list[Path] = []
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def detect_isolation() -> tuple[bool, Literal["venv", "conda", "prefix"] | None]:
    """Whether the running interpreter is inside an isolated environment, and how that
    was determined. Detection order (the single source of truth -- ``consent.py`` reads
    the result off :class:`~kuhaku.core.resolve.environment.Environment`, it does not
    re-detect): ``VIRTUAL_ENV`` -> ``CONDA_PREFIX`` -> ``sys.prefix != sys.base_prefix``.
    """

    if os.environ.get("VIRTUAL_ENV"):
        return True, "venv"
    if os.environ.get("CONDA_PREFIX"):
        return True, "conda"
    if getattr(sys, "prefix", None) != getattr(sys, "base_prefix", sys.prefix):
        return True, "prefix"
    return False, None
