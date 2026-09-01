"""The resolution loop (§3) and the separate activation step.

``resolve()`` is a pure decision -- it probes, enumerates, branches on the candidate
count, may ask, and records the selection. It never calls ``activate``. ``activate()``
runs afterwards: it walks the §5 consent flow for a not-ready candidate and then
constructs the backend, so a failure on a later decision point never leaves an earlier
one half-built.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..exceptions import CapabilityUnavailable
from ._auto import auto_enabled
from ._text import format_bytes
from .consent import consent_and_prepare
from .cost import Candidate
from .environment import Environment, fingerprint
from .registry import Registry
from .ui import UI

Reason = Literal[
    "user_requested",
    "only_option",
    "remembered",
    "user_choice",
    "safe_default",
    "unavailable",
    "auto_disabled",
]


@dataclass(frozen=True)
class Resolution:
    candidate: Candidate | None
    reason: Reason
    baseline: str | None = None
    """The documented-baseline id, included only when it differs from the chosen
    candidate -- the counterfactual the override hint needs (it is what
    ``KUHAKU_AUTO=false`` would give you). Takes no part in selection."""


def _baseline_meta(baseline_id: str | None, chosen: Candidate) -> str | None:
    return baseline_id if baseline_id and baseline_id != chosen.id else None


def _override_hint(kind: str) -> str:
    return f"Pin {kind} explicitly (or set KUHAKU_AUTO=false) to override."


def _cost_phrase(candidate: Candidate) -> str:
    cost = candidate.cost
    tags: list[str] = []
    if cost.sends_document_text:
        tags.append("sends every document's full text off this machine")
    if cost.network_per_use:
        tags.append("network call per use")
    if cost.monetary:
        tags.append("billed per use")
    if not tags:
        tags.append("local, no network")
    phrase = ", ".join(tags)
    return f"{phrase} -- {cost.note}" if cost.note else phrase


def _gap_message(kind: str, candidates: list[Candidate]) -> str:
    if not candidates:
        return (
            f"{kind}: no candidate is available and none is configured. "
            f"{_override_hint(kind)}"
        )
    parts: list[str] = []
    for candidate in candidates:
        cost = candidate.cost
        needs: list[str] = []
        if cost.install_required:
            needs.append("a package install")
        if cost.download_required:
            needs.append(f"a model download ({format_bytes(cost.download_bytes)})")
        step = cost.note or "; ".join(needs) or "setup"
        detail = f" (needs {' and '.join(needs)})" if needs else ""
        parts.append(f"{candidate.label}{detail} -- {step}")
    return f"{kind}: nothing is usable right now. " + "; ".join(parts) + "."


def _chosen_message(kind: str, candidate: Candidate, baseline_id: str | None) -> str:
    tail = ""
    if baseline_id and baseline_id != candidate.id:
        tail = f" (the documented baseline is '{baseline_id}')"
    return (
        f"{kind}: using {candidate.label} -- the only option usable now"
        f"{tail}. {_override_hint(kind)}"
    )


def _skipped_message(
    kind: str, chosen: Candidate, ready: list[Candidate], baseline_id: str | None
) -> str:
    others = ", ".join(c.id for c in ready)
    tail = ""
    if baseline_id and baseline_id != chosen.id:
        tail = f" The documented baseline is '{baseline_id}'."
    return (
        f"{kind}: {len(ready)} options were usable ({others}) but no interactive "
        f"terminal was available, so '{chosen.label}' was chosen as the lowest-risk "
        f"one.{tail} {_override_hint(kind)}"
    )


def _question(kind: str, ready: list[Candidate]) -> str:
    lines = [f"{kind}: more than one option is usable. Choose one:"]
    for candidate in ready:
        lines.append(f"  - {candidate.label}: {_cost_phrase(candidate)}")
    return "\n".join(lines)


def _remembered_message(kind: str, candidate: Candidate) -> str:
    return (
        f"remembered: {kind} = {candidate.label} -- reset with "
        f"`memory.reset({kind!r})` or delete .kuhaku/decisions.json"
    )


def resolve(
    kind: str,
    *,
    registry: Registry,
    env: Environment,
    ui: UI,
    memory,
    requested: str | None = None,
    required: bool,
    candidates: list[Candidate] | None = None,
) -> Resolution:
    """``candidates`` lets a caller that already enumerated this kind (e.g.
    ``build_llm_provider`` for its pinned-name check) pass the list in, so ``probe()``
    is not run a second time. Ignored when ``KUHAKU_AUTO`` is off."""

    baseline_candidate = registry.baseline(kind, env)
    baseline_id = baseline_candidate.id if baseline_candidate else None

    # KUHAKU_AUTO=false: documented baseline, no probing, no question, no memory.
    if not auto_enabled():
        ui.announce(
            "auto disabled (KUHAKU_AUTO); using documented baselines",
            dedupe_key=("auto_disabled",),
        )
        if baseline_candidate is None:
            raise CapabilityUnavailable(
                f"{kind}: KUHAKU_AUTO is set to false, which disables all capability "
                f"probing, and there is no documented baseline for {kind}. "
                f"Unset KUHAKU_AUTO, or pin {kind} explicitly."
            )
        return Resolution(baseline_candidate, "auto_disabled", baseline=None)

    if candidates is None:
        candidates = registry.candidates(kind, env)
    by_id = {c.id: c for c in candidates}

    if requested is not None:
        chosen = by_id.get(requested)
        if chosen is None:
            available = ", ".join(sorted(by_id)) or "none"
            raise CapabilityUnavailable(
                f"{kind}: '{requested}' was requested but no adapter offers it "
                f"(available: {available}). Pin a different {kind}, or install what "
                f"'{requested}' needs."
            )
        return Resolution(chosen, "user_requested", _baseline_meta(baseline_id, chosen))

    ready = [c for c in candidates if c.ready]
    fp = fingerprint(env, packages=registry.required_packages())

    remembered_id = memory.get(kind, fp)

    def remember(candidate_id: str) -> None:
        # Skip the write when the store already holds this exact selection -- no need to
        # rewrite decisions.json (or re-run its unwritable-dir handling) for a no-op.
        if candidate_id != remembered_id:
            memory.put(kind, fp, candidate_id)

    if remembered_id is not None:
        remembered = by_id.get(remembered_id)
        if remembered is not None and remembered.ready:
            ui.announce(
                _remembered_message(kind, remembered),
                dedupe_key=(kind, remembered.id, "remembered"),
            )
            return Resolution(remembered, "remembered", _baseline_meta(baseline_id, remembered))
        # The remembered choice is gone (credential rotated, package removed, server
        # stopped). Say so before re-deciding -- otherwise the fall-through to a lesser
        # option looks like a fresh decision rather than a demotion.
        ui.announce(
            f"{kind}: the remembered choice '{remembered_id}' is no longer usable; "
            f"re-deciding. Run `memory.reset({kind!r})` if that is unexpected.",
            dedupe_key=(kind, remembered_id, "remembered_stale"),
        )

    if not ready:
        message = _gap_message(kind, candidates)
        if required:
            ui.announce(message, prominent=True)
            raise CapabilityUnavailable(message)
        ui.announce(message, degraded=True)
        return Resolution(None, "unavailable", baseline_id)

    if len(ready) == 1:
        chosen = ready[0]
        ui.announce(
            _chosen_message(kind, chosen, baseline_id),
            dedupe_key=(kind, chosen.id, "only_option"),
        )
        remember(chosen.id)
        return Resolution(chosen, "only_option", _baseline_meta(baseline_id, chosen))

    if ui.is_interactive():
        picked = ui.ask(_question(kind, ready), ready)
        if picked is not None:
            remember(picked.id)
            return Resolution(picked, "user_choice", _baseline_meta(baseline_id, picked))

    safe = min(ready, key=lambda c: (c.safety_rank, c.id))
    ui.announce(
        _skipped_message(kind, safe, ready, baseline_id),
        prominent=True,
        dedupe_key=(kind, safe.id, "safe_default"),
    )
    remember(safe.id)
    return Resolution(safe, "safe_default", _baseline_meta(baseline_id, safe))


def activate(resolution: Resolution, *, env: Environment, ui: UI) -> Any:
    """Construct the backend for a resolved decision. Runs the §5 consent flow first for
    a not-ready candidate. Returns ``None`` when the resolution selected nothing (an
    optional capability that degraded)."""

    candidate = resolution.candidate
    if candidate is None:
        return None

    if resolution.reason == "auto_disabled":
        try:
            return candidate.activate()
        except Exception as exc:
            raise CapabilityUnavailable(
                f"{candidate.kind}: the documented baseline '{candidate.id}' is not "
                f"usable, and KUHAKU_AUTO is set to false, which disables any fallback: "
                f"{exc}. Unset KUHAKU_AUTO, or pin {candidate.kind} explicitly."
            ) from exc

    if not candidate.ready:
        consent_and_prepare(candidate, env=env, ui=ui)
    return candidate.activate()
