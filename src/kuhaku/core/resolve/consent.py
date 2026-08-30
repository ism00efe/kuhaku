"""The consent flow (§5) and the install execution policy (§5.1).

Two separate approval moments, neither ever automatic: installing a package, and
downloading a model. Approving one never implies the other.

After a package install is approved, kuhaku runs ``pip`` itself *only if it can verify it
is inside an isolated environment* -- and it verifies that by reading
``Environment.in_isolated_python`` / ``isolation_source``, which
``probes.detect_isolation`` produced. This module adds no environment detection of its
own. If the interpreter is not isolated, kuhaku prints the exact command plus how to make
an environment, and stops. False negatives cost a printed command; false positives would
write into a system Python and are unacceptable.
"""

from __future__ import annotations

import subprocess
import sys

from ..exceptions import ConsentRequired
from ._text import format_bytes
from .cost import Candidate
from .environment import Environment


def run_pip(spec: str) -> int:
    """Run ``pip install <spec>`` in the current interpreter. Returns the exit code.
    Patched out in tests -- the real path is only reached inside a verified isolated
    environment."""

    completed = subprocess.run(
        [sys.executable, "-m", "pip", "install", *spec.split()],
        check=False,
    )
    return completed.returncode


def _install_spec(candidate: Candidate) -> str:
    """The package spec to install. ``Cost.note`` is the declared place for it; when it
    reads like a pip command we take its target, otherwise we fall back to the id."""

    note = candidate.cost.note.strip()
    lowered = note.lower()
    if lowered.startswith("pip install "):
        return note[len("pip install "):].strip()
    return candidate.id


def perform_install(candidate: Candidate, *, env: Environment, ui) -> None:
    spec = _install_spec(candidate)
    if not env.in_isolated_python:
        ui.announce(
            f"{candidate.label}: install approved, but this interpreter is not isolated "
            f"(no virtualenv or conda environment). kuhaku will not install into a system "
            f"Python. Create an environment first, then install:\n"
            f"    python -m venv .venv && . .venv/bin/activate\n"
            f"    pip install {spec}",
            prominent=True,
        )
        raise ConsentRequired(
            f"{candidate.kind}: install of '{spec}' approved but the interpreter is not "
            f"isolated; run `pip install {spec}` inside a virtualenv yourself."
        )
    code = run_pip(spec)
    if code != 0:
        raise ConsentRequired(
            f"{candidate.kind}: `pip install {spec}` exited {code}; install it manually "
            f"and retry."
        )


def consent_and_prepare(candidate: Candidate, *, env: Environment, ui) -> None:
    """Walk the §5 approvals for a not-ready candidate. Raises :class:`ConsentRequired`
    the moment an approval is withheld or an approved install cannot run. The download
    itself happens inside ``candidate.activate()``; this only gates it."""

    cost = candidate.cost
    if cost.install_required:
        if not ui.confirm(f"install the package for {candidate.label} ({cost.note})", cost):
            raise ConsentRequired(
                f"{candidate.kind}: {candidate.label} needs a package install; not "
                f"approved. Run: {cost.note or ('pip install ' + candidate.id)}"
            )
        perform_install(candidate, env=env, ui=ui)

    if cost.download_required:
        size = format_bytes(cost.download_bytes)
        if not ui.confirm(f"download the model for {candidate.label} ({size})", cost):
            raise ConsentRequired(
                f"{candidate.kind}: {candidate.label} needs a model download ({size}); "
                f"not approved. {cost.note}".rstrip()
            )
