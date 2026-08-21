"""Typed, environment-driven configuration.

A single ``Settings`` object is the one place all knobs live. Values come from the
process environment / a local ``.env`` file (see ``.env.example``). Nothing here is
required to run the local Ollama default.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from kuhaku.tools.rag.config import RAGSettings


def _default_rag_settings() -> RAGSettings:
    """Deferred import, not a module-level one -- see the ``model_rebuild()`` call at
    the bottom of this file for why."""

    from kuhaku.tools.rag.config import RAGSettings

    return RAGSettings()


class Settings(BaseSettings):
    """Generic, cross-tool application settings loaded from the environment / an
    optional ``.env`` file. RAG-specific configuration lives on :attr:`rag`
    (:class:`~kuhaku.tools.rag.config.RAGSettings`), not here -- see that class's
    docstring. Adding a second tool to kuhaku means giving it its own
    ``<tool>Settings`` + a ``Settings`` field the same way, never adding tool-specific
    fields directly on this class.

    No dotenv file is read by default (kuhaku must not assume a current working
    directory) -- pass ``env_file`` to :func:`load_settings`/:func:`get_settings`, or
    ``_env_file=...`` directly to the constructor, to opt into one.
    """

    model_config = SettingsConfigDict(
        env_prefix="KUHAKU_",
        env_file=None,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        # Enables nested env vars for `rag`. Combined with env_prefix above, the
        # working names carry both: KUHAKU_RAG__TOP_K=8, KUHAKU_RAG__RERANK_ENABLED=true.
        # No top-level field name here contains "__", so the delimiter never leaks
        # into ordinary flat env var parsing.
        env_nested_delimiter="__",
    )

    # --- LLM provider ---------------------------------------------------------
    llm_provider: str = Field(default="ollama")  # ollama | anthropic | openai
    llm_temperature: float = Field(default=0.1)
    llm_max_tokens: int = Field(default=1024)

    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="qwen2.5:7b-instruct")

    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("KUHAKU_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    )
    anthropic_model: str = Field(default="claude-sonnet-5")

    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("KUHAKU_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    openai_model: str = Field(default="gpt-4o-mini")
    openai_base_url: str = Field(default="https://api.openai.com/v1")

    # Google Vertex AI (optional, requires the `vertex` extra -- see DECISIONS.md).
    # Auth is via Application Default Credentials (gcloud/service account), not an API
    # key field here. Generic (cross-tool) platform settings -- also consumed by the RAG
    # tool's Vertex embedding provider via RAGSettings.vertex_project/vertex_location,
    # see RAGSettings.from_settings().
    vertex_project: str | None = Field(
        default=None,
        validation_alias=AliasChoices("KUHAKU_VERTEX_PROJECT", "GOOGLE_CLOUD_PROJECT"),
    )
    vertex_location: str | None = Field(
        default="us-central1",
        validation_alias=AliasChoices("KUHAKU_VERTEX_LOCATION", "GOOGLE_CLOUD_LOCATION"),
    )
    vertex_model: str = Field(default="gemini-2.5-flash")

    # --- Default behavior policy (see DECISIONS.md D53) ------------------------
    # Performance/helper components (BM25, cross-encoder reranker) configured as the
    # kuhaku default fall back to a simpler alternative and log a warning when they
    # fail to load, so the app still starts (kuhaku.core.policy.apply_fallback_policy).
    # Setting this True upgrades those failures to a startup-halting exception instead,
    # for operators who would rather fail loudly than silently run degraded.
    strict_performance_components: bool = Field(default=False)

    # --- Logging ----------------------------------------------------------
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")  # json | text

    # --- Metrics (Prometheus) ----------------------------------------------
    # Metrics exposition is now handled by the JWT-protected GET /api/admin/metrics
    # endpoint in admin_routes.py. The standalone metrics HTTP server is disabled.
    metrics_enabled: bool = Field(default=False)

    # --- Security: prompt injection guard -----------------------------------
    input_guard_enabled: bool = Field(default=True)

    # --- Category 2: query-answer cache (SQLite) --------------------------------
    # Shared by the RAG query-answer cache (RAGSettings.cache_enabled/cache_ttl_seconds)
    # and the feedback table (two independently-managed tables, one file) -- simpler ops
    # than two SQLite files for two small, never-joined concerns. Stays a generic,
    # cross-tool field since the feedback table has nothing to do with RAG.
    sqlite_db_path: str = Field(default="")

    # --- Prompt Injection Guard v2 (normalize -> two-stage classify -> 3-zone) -------
    # Master kill switch for the whole v2 pipeline (normalizer, classifier, datamarked
    # prompt is unaffected -- see rag/prompts.py -- but the 3-zone decision, output guard,
    # and audit log are all dormant unless this is True). Defaults False, unlike this
    # project's other `*_enabled` toggles: with no Stage-2 model shipped (see
    # guard_stage2_onnx_path below), Stage-2 stays permanently degraded, which forces
    # every v2-guarded request into RESTRICTED regardless of Stage-1's score. The legacy
    # `input_guard_enabled` regex guard above is unaffected and keeps running either way.
    # See DECISIONS.md D39.
    guard_enabled: bool = Field(default=False)
    guard_low_threshold: float = Field(default=0.3)
    guard_high_threshold: float = Field(default=0.7)
    guard_stage1_model_path: str = Field(default="")
    guard_stage2_onnx_path: str = Field(default="")
    guard_stage2_tokenizer_path: str = Field(default="")
    guard_sampling_rate: float = Field(default=0.05)
    guard_norm_drift_tolerance: int = Field(default=5)
    guard_citation_grounding_threshold: float = Field(default=0.1)
    guard_model_version: str = Field(default="1.0.0")
    guard_version: str = Field(default="2.0.0")

    # --- Audit log (security/audit.py) ----------------------------------------------
    # Originally guard-v2-only (FR7/A12); D41 made audit logging unconditional -- every
    # request now writes a record here regardless of guard_enabled -- so this setting
    # moved out of the guard section and dropped its `guard_` prefix. Same file both
    # kinds of call site write to.
    #
    # audit_enabled (D53/D55): on by default -- a strong security default, explicit
    # opt-out only. audit_log_path unset/None/"" means "use the kuhaku-managed
    # default" (./logs/kuhaku_audit.jsonl, see security/audit.py); it is not itself
    # an enable/disable switch, so a caller can set an explicit path while still
    # disabling audit logging via audit_enabled=False.
    audit_enabled: bool = Field(default=True)
    audit_log_path: str | None = Field(default=None)

    # --- Retry (D40): LLM, embedding, vector store, reranker call sites -------------
    # Master kill switch for all four retry sites. All three LLM providers (Ollama,
    # Anthropic, OpenAI) share one generic RETRY_LLM_* config -- they are structurally
    # identical (requests.post -> raise_for_status -> LLMError, no vendor SDK) and the
    # project's LLM abstraction principle requires that swapping LLM_PROVIDER never
    # silently drops behavior. Per-provider tuning was discussed and deliberately
    # deferred -- see DECISIONS.md D40.
    # retry_enabled is the master kill switch shared by every subsystem's retry site,
    # RAG-owned ones included -- RAGSettings.retry_enabled mirrors this value (see
    # RAGSettings.from_settings()) so RAG components never need to read Settings
    # directly. The RAG-owned retry sites themselves (embedding, vector store,
    # reranker) live on RAGSettings; only the generic LLM retry site stays here.
    retry_enabled: bool = Field(default=True)
    retry_llm_max_attempts: int = Field(default=3)
    retry_llm_backoff_base_seconds: float = Field(default=1.0)
    retry_llm_backoff_max_seconds: float = Field(default=10.0)

    # --- LLM resilience: timeout + circuit breaker -----------------------------------
    # Shared across all four LLM providers (Ollama, Anthropic, OpenAI, Vertex AI), same
    # rationale as RETRY_LLM_* above: they must not silently diverge on LLM_PROVIDER swap.
    # Stack order per call: circuit breaker (fast-fail while the dependency is down) ->
    # call_with_retry (retries within one allowed attempt) -> the HTTP call, bounded by
    # this timeout.
    llm_timeout_seconds: int = Field(default=120)
    circuit_breaker_enabled: bool = Field(default=True)
    circuit_breaker_failure_threshold: int = Field(default=5)
    circuit_breaker_reset_timeout_seconds: float = Field(default=60.0)
    circuit_breaker_success_threshold: int = Field(default=1)

    # --- Model versioning (D42) -------------------------------------------------------
    # Deployment-time label, set manually by whoever upgrades the LLM model -- not
    # auto-detected. Independent of the functional selector fields above
    # (ollama_model/anthropic_model/openai_model) by design: those choose which model
    # loads, this is a reproducibility label for what actually produced a response (can
    # intentionally diverge -- e.g. a provider swaps a checkpoint behind the same name).
    # RAG-specific versioning (embedding_model_version, prod_prompt_version,
    # eval_prompt_version) lives on RAGSettings.
    llm_model_version: str = Field(default="qwen2.5:7b-instruct-q4_k_m")

    # --- RAG configuration -------------------------------------------------------------
    # All RAG-specific settings (corpus/chunking/retrieval/re-ranking/caching/
    # contradiction-detection/query-rewriting, the vector store, RAG-owned retry sites,
    # and RAG model/prompt versioning) live on RAGSettings, not here -- see
    # kuhaku.tools.rag.config.RAGSettings. Supports nested env vars, which carry the
    # KUHAKU_ prefix like every other field: KUHAKU_RAG__TOP_K=8,
    # KUHAKU_RAG__RERANK_ENABLED=false,
    # KUHAKU_RAG__DOC_TYPE_PREFIX_MAPPING='{"log_": "log_sample"}'.
    rag: RAGSettings = Field(default_factory=_default_rag_settings)


# D46: bootstrap-only fields -- must always come from .env/the environment, never from
# app_config, since they're either needed to even locate app_config's own database
# (the *_db_path fields) or are secrets that have no business being duplicated into a
# second store.
_BOOTSTRAP_KEYS = frozenset(
    {
        "sqlite_db_path",
        "anthropic_api_key",
        "openai_api_key",
        "ollama_base_url",
    }
)


def load_settings(
    app_config_loader: Any = None,
    env_file: str | None = None,
) -> Settings:
    """Layer defaults < ``app_config`` (DB) < ``.env``/env (bootstrap keys only).

    ``base`` already reflects today's defaults + ``.env``/environment (pydantic-settings'
    normal behavior, unchanged). Every ``app_config`` key that isn't bootstrap-only is
    then overlaid on top of ``base``'s own dump, and re-passed as explicit constructor
    kwargs -- pydantic-settings' documented precedence (init kwargs > env > dotenv >
    defaults) makes that overlay win over ``.env`` automatically, and pydantic's lax
    coercion turns the DB's ``TEXT`` values back into ``bool``/``int``/``float``. See
    DECISIONS.md D46 for why this, not manual coercion, is the whole mechanism.

    ``app_config_loader`` is an optional callable (no arguments) that returns a
    ``dict[str, str]`` of persisted config key/value pairs, forming the third
    (DB) config layer. When ``None`` (the default), the DB layer is skipped and
    only defaults + env vars are used. An embedding application passes its own
    loader here; kuhaku-only callers pass nothing. This keeps
    ``kuhaku.core.config`` free of any import from the application layer.

    ``env_file`` is an optional dotenv file path, forwarded as ``_env_file`` to
    ``Settings()``. ``None`` (the default) reads no dotenv file at all -- kuhaku
    never assumes a current working directory; callers that want ``.env`` support
    resolve their own path (e.g. relative to their package) and pass it explicitly.
    """

    base = Settings(_env_file=env_file)
    if app_config_loader is None:
        return base

    stored = app_config_loader()
    if not stored:
        return base

    merged = base.model_dump()
    for key, value in stored.items():
        if key in _BOOTSTRAP_KEYS or key not in merged:
            continue
        merged[key] = value
    return Settings(_env_file=env_file, **merged)


@lru_cache
def get_settings(
    app_config_loader: Any = None,
    env_file: str | None = None,
) -> Settings:
    """Return a cached ``Settings`` instance, refreshed via :func:`load_settings`.

    Runtime config changes (``PUT /api/admin/config``) do not rely on this cache being
    cleared to take effect -- ``AssistantService.reconfigure`` mutates the one shared
    ``Settings`` instance every route closure already holds a reference to, in place.
    ``cache_clear()`` is called defensively after a write anyway, for any future code
    path that constructs ``Settings`` fresh. See DECISIONS.md D46.

    ``app_config_loader``/``env_file`` are forwarded to :func:`load_settings`; see its
    docstring. Kuhaku-only callers that never need the DB layer or a dotenv file omit
    both.
    """

    return load_settings(app_config_loader, env_file)


def configure_logging(level: str | None = None, log_format: str | None = None) -> None:
    """Configure root logging once.

    ``log_format="json"`` (the default) emits structured, machine-parseable log lines
    with a propagated ``trace_id`` — see ``observability.logging_context``.
    ``log_format="text"`` keeps the original human-readable console format, useful for
    interactive debugging.
    """

    settings = get_settings()
    resolved_level = (level or settings.log_level).upper()
    resolved_format = (log_format or settings.log_format).lower()

    # Local import: keeps config.py free of the observability import at module load
    # time (config is imported by nearly every module; observability is not).
    from .observability.logging_context import JsonFormatter, TraceIdFilter

    handler = logging.StreamHandler()
    handler.addFilter(TraceIdFilter())
    if resolved_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | [%(trace_id)s] | %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    root = logging.getLogger()
    root.setLevel(getattr(logging, resolved_level, logging.INFO))
    root.handlers = [handler]


# Resolve the forward-referenced `rag: "RAGSettings"` annotation now that Settings is
# fully defined. This import cannot move to the top of the file: kuhaku.tools.rag's
# package __init__ transitively imports kuhaku.core.llm, which imports Settings from
# this very module -- doing that import before the `class Settings` statement above has
# executed would be a circular import (ImportError: cannot import name 'Settings' from
# partially initialized module). Importing here, after Settings already exists in this
# module's namespace, breaks the cycle; model_rebuild() then lets pydantic finish
# building Settings' schema against the now-resolvable RAGSettings type.
from kuhaku.tools.rag.config import RAGSettings  # noqa: E402

Settings.model_rebuild()
