"""Append-only audit logging — FR5 (guard v2 decisions), D41/FR8 (every request), D42
(model/prompt version provenance), D43 (optional ``event_type`` labeling), and D44
(``action``/``target`` fields plus a :func:`read_recent` reader for the admin panel).

One JSON line per audit-worthy event, written to a local JSONL file (``data/audit.jsonl``
by default). Never carries raw user input or raw LLM output — only a hash of the raw
question and a short prefix of the already-sanitized retrieval query, plus whatever
decision/identity metadata is available at the call site.

Two kinds of call site share this one ``record()`` function and schema (D41): the
guard-v2-specific ones in ``RAGEngine._answer()`` (only reached when ``guard_enabled``),
which pass a real ``decision``/``guard_version``/``model_version``/``thresholds``; and the
unconditional per-request one (reached regardless of ``guard_enabled``), which passes
``decision=None`` and the other guard-specific fields as ``None`` -- the JSON output nulls
those keys rather than omitting them, so every audit line has the same shape. Call sites
also carry ``auth_context``/``accessed_chunks``, letting every record answer "who accessed
what" -- without kuhaku assuming any fixed role or clearance vocabulary (see
``kuhaku.core.auth``).

Security-critical (D53), unlike ``rag.cache.QueryAnswerCache``: while audit logging is
enabled (``Settings.audit_enabled``, default ``True``), a failure to write a record is
never swallowed -- it raises ``AuditWriteError`` so the operator sees it instead of
silently losing audit coverage. An operator who wants to run without audit logging must
say so explicitly via ``audit_enabled=False``, which makes ``record()`` an immediate,
filesystem-untouched no-op.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..auth import AuthContext
from ..observability.metrics import AUDIT_RECORDS_TOTAL, record_guard_degradation

if TYPE_CHECKING:
    from .guard import GuardDecision

logger = logging.getLogger(__name__)

# Guards interleaved writes from FastAPI's threadpool-dispatched sync handlers (same
# single-process-MVP scope as api/middleware.py's RateLimiter -- see D19/D23). Does not
# protect against multiple OS processes writing the same file concurrently.
_write_lock = threading.Lock()

# Fallback location when `Settings.audit_log_path` is unset/empty, so audit logging is on
# by default rather than silently disabled -- relative to the process's current working
# directory, same convention as the rest of the app's local-file defaults.
_DEFAULT_AUDIT_LOG_PATH = "./logs/kuhaku_audit.jsonl"


class AuditWriteError(Exception):
    """Audit logging is enabled but a record could not be written (e.g. the log
    directory could not be created, or the file could not be opened for append).

    Raised rather than logged-and-swallowed: audit coverage silently going missing is a
    security regression, not a degraded-but-functional state (D53).
    """


def record(
    path: str | None,
    *,
    enabled: bool = True,
    trace_id: str,
    raw_question: str,
    sanitized_retrieval_query: str,
    decision: GuardDecision | None = None,
    guard_version: str | None = None,
    model_version: str | None = None,
    thresholds: tuple[float, float] | None = None,
    output_checks: dict[str, object] | None = None,
    auth_context: AuthContext | None = None,
    accessed_chunks: list[str] | None = None,
    llm_version: str | None = None,
    embedding_version: str | None = None,
    system_prompt_version: str | None = None,
    event_type: str | None = None,
    action: str | None = None,
    target: str | None = None,
    faithfulness_score: float | None = None,
    hallucination_rate: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Append one audit record.

    ``raw_question`` is only ever hashed, never stored. ``sanitized_retrieval_query`` is
    the already-sanitized text computed earlier in ``RAGEngine._answer()`` -- truncated to
    100 chars, never re-derived from raw input. ``decision``/``guard_version``/
    ``model_version``/``thresholds`` are ``None`` for the unconditional per-request call
    site (D41/FR8), which has no guard-v2 decision to report -- their JSON keys are still
    present, just null, so every audit line has the same shape regardless of call site.
    ``llm_version``/``embedding_version``/``system_prompt_version`` (D42) are the
    deployed versions from ``Settings`` at the time this record was written -- ``None``
    only when the caller has none to report (e.g. a direct/test construction of
    ``RAGEngine`` with no versions configured). ``event_type`` (D43) is an optional free-text
    label distinguishing what kind of event this record represents. ``action``/``target``
    (D44) are free-form fields for a mutating event -- the specific action performed
    (e.g. ``"lock_user"``) and the thing it acted on. ``auth_context`` (see
    ``kuhaku.core.auth``) is the acting identity, if any -- its ``identity``/
    ``is_authenticated``/``roles`` are written as their own top-level fields so the audit
    trail shows who performed an action without kuhaku assuming any fixed role or
    clearance vocabulary; anything else about the identity (a custom "admin tier", a
    session id, ...) belongs in ``auth_context.metadata`` or the generic ``metadata``
    parameter below, not a new named parameter here. ``faithfulness_score``/
    ``hallucination_rate`` (D47) are set only by the asynchronous
    ``event_type="faithfulness_evaluation"`` call site (``api/routes.py``'s background
    task) -- ``None`` everywhere else. All default to ``None``, so every existing call
    site keeps working unchanged.

    ``metadata`` is a generic, tool-scoped bag for anything a caller wants recorded that
    doesn't warrant its own named parameter here (this module is core, tool-agnostic
    infrastructure -- it must not grow a new named field for every tool's own audit
    needs). Written under the ``"metadata"`` key, ``{}`` when omitted, so every audit line
    has the same shape. The RAG-specific named parameters above (``accessed_chunks``,
    ``faithfulness_score``, ``hallucination_rate``) predate this and stay their own
    top-level fields for backward compatibility with existing consumers of this schema;
    a new tool-specific field should prefer ``metadata`` over adding another named
    parameter here.

    ``enabled`` mirrors ``Settings.audit_enabled`` -- ``False`` returns immediately
    without touching the filesystem (the operator opted out explicitly); the default
    ``True`` keeps every pre-existing call site's behavior, since none of them passed
    this parameter before it existed.

    ``path`` empty or ``None`` (``Settings.audit_log_path``'s own default) means the
    caller hasn't configured an explicit location -- falls back to
    ``_DEFAULT_AUDIT_LOG_PATH`` (``./logs/kuhaku_audit.jsonl``, created if missing) so
    audit logging is on by default whenever it's enabled. Unlike the rest of this
    function's inputs, a failure to create that directory or write the record is never
    swallowed while ``enabled`` is ``True`` -- it raises :class:`AuditWriteError` so the
    operator sees it rather than silently losing audit coverage; an operator who wants no
    audit logging must say so via ``enabled=False``.
    """

    if not enabled:
        return

    if not path:
        path = _DEFAULT_AUDIT_LOG_PATH

    record_obj = {
        "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "trace_id": trace_id,
        "event_type": event_type,
        "action": action,
        "target": target,
        "input_hash": hashlib.sha256(raw_question.encode("utf-8")).hexdigest(),
        "input_redacted": sanitized_retrieval_query[:100],
        "identity": auth_context.identity if auth_context else None,
        "is_authenticated": auth_context.is_authenticated if auth_context else None,
        "roles": list(auth_context.roles) if auth_context else None,
        "faithfulness_score": faithfulness_score,
        "hallucination_rate": hallucination_rate,
        "accessed_chunks": accessed_chunks or [],
        "llm_version": llm_version,
        "embedding_version": embedding_version,
        "system_prompt_version": system_prompt_version,
        "normalized_length": decision.normalized_length if decision else None,
        "norm_drift": decision.norm_drift if decision else None,
        "stage1_score": decision.stage1.score if decision else None,
        "stage1_top_features": (
            [list(f) for f in decision.stage1.top_features] if decision else None
        ),
        "stage2_score": decision.stage2.confidence if decision else None,
        "stage2_degraded": decision.stage2.degraded if decision else None,
        "zone": decision.zone if decision else None,
        "model_version": model_version,
        "guard_version": guard_version,
        "thresholds": (
            {"low": thresholds[0], "high": thresholds[1]} if thresholds else None
        ),
        "output_checks": output_checks,
        "metadata": metadata or {},
    }
    line = json.dumps(record_obj, ensure_ascii=False)

    file_path = Path(path)
    try:
        with _write_lock:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with file_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    except OSError as exc:
        logger.exception("failed to write audit record")
        record_guard_degradation("audit")
        raise AuditWriteError(
            f"Audit logging is enabled but the record at '{path}' could not be written "
            f"({exc}). Fix the path's permissions, point `audit_log_path` at a writable "
            f"location, or set `audit_enabled=False` to disable audit logging explicitly."
        ) from exc
    AUDIT_RECORDS_TOTAL.add(1)


def read_recent(path: str) -> list[dict[str, object]]:
    """Read every parseable audit record from ``path``, oldest first (file order) (D44).

    Best-effort on the read side (unlike :func:`record`'s write side, which now raises
    on failure while enabled -- see D56): a missing file returns ``[]``; a line that
    fails to parse as JSON is skipped (logged), never raised. Reading is not
    security-critical the way writing is -- a corrupt/missing log file surfaces as an
    empty admin-panel view, not a silently-missing audit trail.

    Deliberately unfiltered and unlimited -- callers that need "the last 50 of one
    ``event_type``" must filter by ``event_type`` *before* truncating to 50: truncating
    first (e.g. reading only the last 50 raw lines and filtering after) would silently
    under-return whenever the file's tail happens to be dominated by the other
    ``event_type``. Filtering and the "last N" cutoff are therefore both left to the
    caller.
    """

    file_path = Path(path)
    if not file_path.exists():
        return []
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        logger.exception("failed to read audit log")
        return []
    records: list[dict[str, object]] = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("skipping unparsable audit line")
    return records


read_audit_log = read_recent

