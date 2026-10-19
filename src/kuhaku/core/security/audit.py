"""Append-only audit logging — covers guard v2 decisions, an unconditional record of
every request, model/prompt version provenance, optional ``event_type`` labeling, and
``action``/``target`` fields, plus a :func:`read_recent` reader for the admin panel.

One JSON line per audit-worthy event, written to a local JSONL file (``data/audit.jsonl``
by default). Never carries raw user input or raw LLM output — only a hash of the raw
question and a short prefix of the already-sanitized retrieval query, plus whatever
decision/identity metadata is available at the call site.

Two kinds of call site share this one ``record()`` function and schema: the
guard-v2-specific ones in ``RAGEngine._answer()`` (only reached when ``guard_enabled``),
which pass a real ``decision``/``guard_version``/``model_version``/``thresholds``; and the
unconditional per-request one (reached regardless of ``guard_enabled``), which passes
``decision=None`` and the other guard-specific fields as ``None`` -- the JSON output nulls
those keys rather than omitting them, so every audit line has the same shape. Call sites
also carry ``auth_context``/``accessed_chunks``, letting every record answer "who accessed
what" -- without kuhaku assuming any fixed role or clearance vocabulary (see
``kuhaku.core.auth``).

Security-critical, unlike ``rag.cache.QueryAnswerCache``: while audit logging is
enabled (``Settings.audit_enabled``, default ``True``), a failure to write a record is
never swallowed -- it raises ``AuditWriteError`` so the operator sees it instead of
silently losing audit coverage. An operator who wants to run without audit logging must
say so explicitly via ``audit_enabled=False``, which makes ``record()`` an immediate,
filesystem-untouched no-op.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..auth import AuthContext
from ..exceptions import KuhakuError
from ..observability.metrics import AUDIT_RECORDS_TOTAL, record_guard_degradation
from ..sanitization import sanitize

if TYPE_CHECKING:
    from .guard import GuardDecision

logger = logging.getLogger(__name__)

# Guards interleaved writes from the embedding application's threadpool-dispatched
# request handlers (same single-process-MVP scope as its own rate limiter). Does not
# protect against multiple OS processes writing the same file concurrently.
_write_lock = threading.Lock()

# Fallback location when `Settings.audit_log_path` is unset/empty, so audit logging is on
# by default rather than silently disabled -- relative to the process's current working
# directory, same convention as the rest of the app's local-file defaults.
_DEFAULT_AUDIT_LOG_PATH = "./logs/kuhaku_audit.jsonl"


# The key that turns `input_hash` into an HMAC. An unsalted SHA-256 of a short question
# is reversible by guessing ("is it 'what is my balance?'"), so the hash is keyed per
# deployment: from this variable when set, else from a key file created in the log's
# directory (one per directory, so every log of one deployment hashes alike).
AUDIT_KEY_ENV = "KUHAKU_AUDIT_HMAC_KEY"
AUDIT_KEY_FILENAME = ".kuhaku_audit.key"
_keys: dict[str, bytes] = {}

# Values under keys named like a digest ("hash", "content_hash", "sql_sha256", ...) are
# digests by convention (64 hex chars) and would otherwise be masked as long-hex secrets.
_DIGEST_KEY_RE = re.compile(r"(?:hash|digest|sha\d*)$", re.IGNORECASE)


class AuditWriteError(KuhakuError):
    """Audit logging is enabled but a record could not be written (e.g. the log
    directory could not be created, or the file could not be opened for append).

    Raised rather than logged-and-swallowed: audit coverage silently going missing is a
    security regression, not a degraded-but-functional state.
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

    ``raw_question`` is only ever hashed (HMAC, see :func:`_audit_key`), never stored.
    Every free-text field is sanitized here, whatever the caller already did.
    ``sanitized_retrieval_query`` is the already-sanitized text computed earlier in ``RAGEngine._answer()`` -- truncated to
    100 chars, never re-derived from raw input. ``decision``/``guard_version``/
    ``model_version``/``thresholds`` are ``None`` for the unconditional per-request call
    site, which has no guard-v2 decision to report -- their JSON keys are still
    present, just null, so every audit line has the same shape regardless of call site.
    ``llm_version``/``embedding_version``/``system_prompt_version`` are the
    deployed versions from ``Settings`` at the time this record was written -- ``None``
    only when the caller has none to report (e.g. a direct/test construction of
    ``RAGEngine`` with no versions configured). ``event_type`` is an optional free-text
    label distinguishing what kind of event this record represents. ``action``/``target``
    are free-form fields for a mutating event -- the specific action performed
    (e.g. ``"lock_user"``) and the thing it acted on. ``auth_context`` (see
    ``kuhaku.core.auth``) is the acting identity, if any -- its ``identity``/
    ``is_authenticated``/``roles`` are written as their own top-level fields so the audit
    trail shows who performed an action without kuhaku assuming any fixed role or
    clearance vocabulary; anything else about the identity (a custom "admin tier", a
    session id, ...) belongs in ``auth_context.metadata`` or the generic ``metadata``
    parameter below, not a new named parameter here. ``faithfulness_score``/
    ``hallucination_rate`` are set only by the asynchronous
    ``event_type="faithfulness_evaluation"`` call site (an embedding application's
    background evaluation task) -- ``None`` everywhere else. All default to ``None``, so
    every existing call site keeps working unchanged.

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
        "event_type": _clean(event_type),
        "action": _clean(action),
        "target": _clean(target),
        "input_hash": _input_hash(path, raw_question),
        "input_redacted": sanitize(sanitized_retrieval_query)[:100],
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
        "output_checks": _clean_tree(output_checks),
        "metadata": _clean_tree(metadata or {}),
    }
    # ensure_ascii: a raw U+2028/U+0085 inside a record would be a line break to some
    # readers; escaped, every record is exactly one "\n"-terminated line. default=str:
    # a non-JSON metadata value is recorded as its text, never a lost record.
    try:
        line = json.dumps(record_obj, ensure_ascii=True, default=str)
    except (TypeError, ValueError) as exc:
        raise AuditWriteError(f"Audit record is not serializable: {exc}.") from exc

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
    """Read every parseable audit record from ``path``, oldest first (file order).

    Best-effort on the read side (unlike :func:`record`'s write side, which now raises
    on failure while enabled): a missing file returns ``[]``; a line that
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
        # Split on "\n" only: str.splitlines() also breaks on U+2028, U+0085 and other
        # separators that older records (written with ensure_ascii=False) may contain
        # inside a JSON string.
        text = file_path.read_text(encoding="utf-8")
        lines = [line for line in text.split("\n") if line.strip()]
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


def _clean(value: str | None) -> str | None:
    return sanitize(value) if isinstance(value, str) else value


def _clean_tree(value: Any, key: str = "") -> Any:
    """Sanitize every string leaf of a JSON-like value (dict keys are left alone)."""

    if isinstance(value, str):
        if _DIGEST_KEY_RE.search(key):
            return value
        return sanitize(value)
    if isinstance(value, dict):
        return {k: _clean_tree(v, str(k)) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean_tree(v, key) for v in value]
    return value


def _input_hash(path: str, raw_question: str) -> str:
    key = _audit_key(path)
    return hmac.new(key, raw_question.encode("utf-8"), hashlib.sha256).hexdigest()


def _audit_key(path: str) -> bytes:
    """The deployment's HMAC key: ``$KUHAKU_AUDIT_HMAC_KEY``, else
    ``<log directory>/.kuhaku_audit.key`` (created with owner-only permissions on first
    use, then reused). On Windows the mode bits only set read-only/not; the file inherits
    the directory's ACL."""

    env_key = os.environ.get(AUDIT_KEY_ENV)
    if env_key:
        return env_key.encode("utf-8")
    key_path = Path(path).parent / AUDIT_KEY_FILENAME
    cache_key = str(key_path.resolve())
    cached = _keys.get(cache_key)
    if cached is not None:
        return cached
    try:
        with _write_lock:
            key_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                # O_EXCL: two processes racing to create the key must not both write
                # one -- the loser reads the winner's.
                fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                key = _read_key(key_path)
            else:
                generated = secrets.token_hex(32)
                with os.fdopen(fd, "w", encoding="ascii") as fh:
                    fh.write(generated)
                key = generated.encode("ascii")
    except OSError as exc:
        logger.exception("failed to create or read the audit HMAC key")
        record_guard_degradation("audit")
        raise AuditWriteError(
            f"Audit logging is enabled but the record at '{path}' could not be written: "
            f"its HMAC key file '{key_path}' could not be created or read ({exc}). Set "
            f"{AUDIT_KEY_ENV}, or make the directory writable."
        ) from exc
    if not key:
        raise AuditWriteError(f"Audit HMAC key file '{key_path}' is empty; delete it.")
    _keys[cache_key] = key
    return key


def _read_key(key_path: Path) -> bytes:
    # Another process may have created the file an instant ago and not written it yet.
    for _ in range(50):
        key = key_path.read_text(encoding="ascii").strip().encode("ascii")
        if key:
            return key
        time.sleep(0.01)
    return b""
