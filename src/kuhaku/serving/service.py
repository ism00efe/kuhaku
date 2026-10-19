"""The transport-agnostic core of the serving layer.

:class:`ToolService` holds the configured ``RAG`` and extractors and performs one call
on behalf of an already-authenticated :class:`~kuhaku.core.auth.AuthContext`:
authorization, the call itself, mapping its exceptions to
:class:`~kuhaku.serving.errors.ServingError`, and shaping the result into a wire schema.
The HTTP app and the MCP server are thin adapters over it, so both enforce exactly the
same rules.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from kuhaku.core.auth import AuthContext, AuthorizationPolicy
from kuhaku.core.config import Settings, get_settings
from kuhaku.core.exceptions import ConfigError
from kuhaku.core.observability import get_trace_id, instrumented_step
from kuhaku.core.sanitization import Redaction
from kuhaku.core.security.audit import AuditWriteError
from kuhaku.core.security.audit import record as record_audit
from kuhaku.tools.extraction import Extractor
from kuhaku.tools.rag.models import Answer

from .errors import (
    AuthenticationRequired,
    NotFound,
    PermissionDenied,
    to_serving_error,
)
from .schemas import AskResponse, IngestResponse, extraction_payload

logger = logging.getLogger(__name__)

# MCP tool names allow [A-Za-z0-9_.-] up to 128 characters; this is narrower so the same
# name is also a clean URL path segment and `extract_<name>` stays within the limit.
_EXTRACTOR_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# The resource/action vocabulary an AuthorizationPolicy is asked about.
RAG_RESOURCE = "rag"
ASK_ACTION = "ask"
INGEST_ACTION = "ingest"
INGEST_UNTAGGED_ACTION = "ingest_untagged"
EXTRACT_ACTION = "extract"


def extractor_resource(name: str) -> str:
    """The policy resource name for the extractor registered as ``name``."""

    return f"extractor:{name}"


class ServableRAG(Protocol):
    """What the serving layer needs from a RAG: :class:`kuhaku.RAG` satisfies it."""

    def ask(
        self, question: str, context_text: str | None = None, *, auth_context: AuthContext | None = None
    ) -> Answer: ...

    def ingest(
        self, text: str, filename: str = ..., *, access_tags: Sequence[str] | None = None
    ) -> tuple[int, list[Redaction]]: ...


class ToolService:
    """The configured tools, callable on behalf of an authenticated identity.

    ``extractors`` maps a name to an :class:`~kuhaku.tools.extraction.Extractor`; a
    sequence is accepted too and named by each schema's ``name``. Names must match
    ``[A-Za-z0-9_-]{1,64}`` -- they become URL segments and MCP tool names.

    ``policy``, when given, is asked before every call: resource ``"rag"`` with action
    ``"ask"``/``"ingest"``, or resource ``"extractor:<name>"`` with action
    ``"extract"``. ``None`` allows every authenticated-or-anonymous caller the
    authenticator let through to read.

    ``enable_ingest`` is ``False`` by default: ingestion is a write path and has to be
    turned on deliberately. Without a ``policy`` every authenticated identity may
    write (fine for a demo; a warning is logged). Rules no configuration lifts:
    ingestion needs an authenticated identity, and every requested access tag must
    be one of the caller's own roles (no planting a document into a group you are not
    in). With a ``policy``, an untagged document -- visible to every caller --
    additionally needs it to allow action ``"ingest_untagged"`` on ``"rag"``. Every
    ingestion writes an audit record
    (``event_type="rag_ingest"``) to the audit log configured by ``settings``
    (default: the RAG's own settings, else :func:`~kuhaku.core.config.get_settings`).
    """

    def __init__(
        self,
        *,
        rag: ServableRAG | None = None,
        extractors: Mapping[str, Extractor[Any]] | Sequence[Extractor[Any]] | None = None,
        policy: AuthorizationPolicy | None = None,
        enable_ingest: bool = False,
        settings: Settings | None = None,
    ) -> None:
        if enable_ingest and rag is None:
            raise ConfigError("enable_ingest=True needs a rag= to ingest into.")
        if enable_ingest and policy is None:
            logger.warning(
                "ingestion enabled without a policy: every authenticated caller may write "
                "documents (tagged only with their own roles, or untagged)"
            )
        if settings is None:
            # RAG keeps its resolved Settings privately; using them sends ingestion
            # records to the same audit log as the RAG's own per-question records.
            rag_settings = getattr(rag, "_settings", None)
            settings = rag_settings if isinstance(rag_settings, Settings) else get_settings()
        self._audit_enabled = settings.audit_enabled
        self._audit_log_path = settings.audit_log_path
        self.rag = rag
        self.extractors = _named_extractors(extractors)
        if rag is None and not self.extractors:
            raise ConfigError("Nothing to serve: pass rag= and/or extractors=.")
        self.policy = policy
        self.enable_ingest = enable_ingest
        # RAG.ingest's single-writer guard is per store directory across processes; two
        # threads of this process ingesting at once would trip it against each other,
        # so writes from this server are serialized here.
        self._ingest_lock = threading.Lock()

    # --- calls -------------------------------------------------------------------

    def ask(self, auth: AuthContext, question: str, context_text: str | None = None) -> AskResponse:
        rag = self._require_rag()
        self._authorize(auth, RAG_RESOURCE, ASK_ACTION)
        with instrumented_step("serve_rag_ask"):
            try:
                answer = rag.ask(question, context_text, auth_context=auth)
            except Exception as exc:
                raise to_serving_error(exc) from exc
        return AskResponse.from_answer(answer)

    def ingest(
        self,
        auth: AuthContext,
        text: str,
        filename: str = "document.txt",
        access_tags: Sequence[str] | None = None,
    ) -> IngestResponse:
        if not self.enable_ingest:
            raise NotFound("Ingestion is not enabled on this server.")
        rag = self._require_rag()
        if not auth.is_authenticated:
            raise AuthenticationRequired("Ingestion requires an authenticated identity.")
        self._authorize(auth, RAG_RESOURCE, INGEST_ACTION)
        tags = list(access_tags or [])
        # Blank tags fall through to RAG.ingest, whose ValueError says what is wrong.
        foreign = sorted({t for t in tags if t.strip() and t not in auth.roles})
        if foreign:
            # Tags are compared as exact strings at retrieval, so the same comparison is
            # used here: a caller can only restrict a document to groups it belongs to.
            raise PermissionDenied("Every access tag must be one of your own roles.")
        if not tags:
            self._authorize(auth, RAG_RESOURCE, INGEST_UNTAGGED_ACTION)
        with instrumented_step("serve_rag_ingest") as rec:
            try:
                with self._ingest_lock:
                    chunks_added, redactions = rag.ingest(
                        text, filename, access_tags=tags or None
                    )
            except Exception as exc:
                # Ingestion's ValueErrors (too large, unsupported, empty, blank tag,
                # failed security check) carry EngineMessages text written for users.
                raise to_serving_error(exc, user_facing_value_errors=True) from exc
            rec.set(chunk_count=chunks_added)
        self._audit_ingest(auth, text, filename, tags, chunks_added)
        return IngestResponse.build(chunks_added, redactions, get_trace_id())

    def _audit_ingest(
        self, auth: AuthContext, text: str, filename: str, tags: list[str], chunks: int
    ) -> None:
        """Who changed the knowledge base, into the audit trail. The filename is hashed
        (it is caller-supplied over the wire) and the text is only ever hashed."""

        size = len(text.encode("utf-8"))
        filename_hash = hashlib.sha256(filename.encode("utf-8")).hexdigest()
        trace_id = get_trace_id()
        try:
            record_audit(
                self._audit_log_path,
                enabled=self._audit_enabled,
                trace_id=trace_id,
                raw_question=text,
                sanitized_retrieval_query="",
                auth_context=auth,
                event_type="rag_ingest",
                action="ingest",
                target=filename_hash,
                metadata={
                    "filename_sha256": filename_hash,
                    "size_bytes": size,
                    "chunk_count": chunks,
                    "access_tags": tags,
                },
            )
        except AuditWriteError:
            # The document is already in the store; failing the request now would
            # invite a retry that ingests it twice. The failure is logged and counted.
            logger.exception(
                "ingest audit record could not be written", extra={"trace_id": trace_id}
            )
        logger.info(
            "document ingested",
            extra={"file_size_bytes": size, "chunk_count": chunks},
        )

    def extract(
        self,
        name: str,
        auth: AuthContext,
        text: str,
        *,
        reference: Any = None,
        prior: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the extractor registered as ``name``; returns the JSON-ready payload.

        A ``needs_clarification``/``invalid``/``failed`` status is a normal result, not an
        error: the call worked and the status says what the text did or did not hold.
        """

        extractor = self.extractors.get(name)
        if extractor is None:
            raise NotFound(f"No extractor named '{name}'.")
        self._authorize(auth, extractor_resource(name), EXTRACT_ACTION)
        with instrumented_step("serve_extract"):
            try:
                result = extractor.extract(text, reference=reference, prior=prior)
            except Exception as exc:
                raise to_serving_error(exc) from exc
        return extraction_payload(result)

    # --- helpers -----------------------------------------------------------------

    def _require_rag(self) -> ServableRAG:
        if self.rag is None:
            raise NotFound("No RAG is configured on this server.")
        return self.rag

    def _authorize(self, auth: AuthContext, resource: str, action: str) -> None:
        if self.policy is None:
            return
        try:
            allowed = self.policy.check(auth, resource, action)
        except Exception as exc:
            # A policy that crashes must deny, never allow.
            raise to_serving_error(exc) from exc
        if not allowed:
            if not auth.is_authenticated:
                raise AuthenticationRequired()
            raise PermissionDenied()


def _named_extractors(
    extractors: Mapping[str, Extractor[Any]] | Sequence[Extractor[Any]] | None,
) -> dict[str, Extractor[Any]]:
    if extractors is None:
        return {}
    if isinstance(extractors, Mapping):
        named = dict(extractors)
    else:
        named = {}
        for extractor in extractors:
            if extractor.schema.name in named:
                raise ConfigError(
                    f"Two extractors share the schema name '{extractor.schema.name}'; "
                    f"pass a dict to name them."
                )
            named[extractor.schema.name] = extractor
    for name in named:
        if not _EXTRACTOR_NAME_RE.match(name):
            raise ConfigError(
                f"Extractor name {name!r} must match [A-Za-z0-9_-]{{1,64}}: it becomes a "
                f"URL path segment and an MCP tool name."
            )
    return named


__all__ = [
    "ASK_ACTION",
    "EXTRACT_ACTION",
    "INGEST_ACTION",
    "INGEST_UNTAGGED_ACTION",
    "RAG_RESOURCE",
    "ServableRAG",
    "ToolService",
    "extractor_resource",
]
