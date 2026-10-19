"""Typed, environment-driven configuration.

A single ``Settings`` object is the one place all knobs live. Values come from the
process environment / a local ``.env`` file (see ``.env.example``). Nothing here is
required to run the local Ollama default.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Fields that used to exist, mapped to what to do instead. Settings ignores unknown
# environment variables (extra="ignore"), which is right for the environment but would
# make `Settings(rag=...)` silently do nothing -- a removed keyword fails loudly instead.
_REMOVED_FIELDS: dict[str, str] = {
    "rag": (
        "Settings no longer carries tool settings (kuhaku.core may not import a tool). "
        "Build kuhaku.tools.rag.RAGSettings instead -- RAGSettings.from_env() reads the "
        "same KUHAKU_RAG__* variables -- and pass it as RAG(rag_settings=...)."
    ),
}


class Settings(BaseSettings):
    """Generic, cross-tool application settings loaded from the environment / an
    optional ``.env`` file. Tool settings never live here: each tool owns its own
    settings object read from its own ``KUHAKU_<TOOL>__*`` variables (RAG's is
    :class:`~kuhaku.tools.rag.config.RAGSettings`), because ``kuhaku.core`` may never
    import a tool.

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
        # Four fields below carry a validation_alias so ecosystem-standard names
        # (OPENAI_API_KEY, GOOGLE_CLOUD_PROJECT, ...) keep working unprefixed. Without
        # this, pydantic would accept ONLY those aliases, and constructing Settings in
        # code by field name -- Settings(vertex_project="p") -- would be silently
        # ignored rather than rejected.
        populate_by_name=True,
    )

    def __init__(self, **values: Any) -> None:
        for name, advice in _REMOVED_FIELDS.items():
            if name in values:
                raise TypeError(f"Settings({name}=...) was removed. {advice}")
        super().__init__(**values)

    # --- LLM provider ---------------------------------------------------------
    # "auto" (the default) is resolved at build time by kuhaku.core.resolve from the
    # credentials that are set and which models a local Ollama has pulled -- nothing else
    # about the machine is probed. The provider used last (remembered in
    # .kuhaku/decisions.json, explicit choices included) wins and is announced;
    # otherwise a provider whose key is set (the free OpenRouter first, then openai ->
    # anthropic -> vertex -> groq), and only without one a local Ollama with a pulled
    # model. A concrete value is absolute and built directly. KUHAKU_AUTO=false pins
    # "auto" to the documented baseline ("ollama", which then needs ollama_model).
    # auto | ollama | openrouter | anthropic | openai | vertex | groq
    llm_provider: str = Field(default="auto")
    llm_temperature: float = Field(default=0.1)
    llm_max_tokens: int = Field(default=1024)
    # Content types the built provider declares (comma-separated: "text,image,document").
    # Unset, each provider decides from its model name, claiming images/PDFs only for
    # model families documented to accept them -- set this for a model it does not know.
    llm_content_types: str | None = Field(default=None)

    ollama_base_url: str = Field(default="http://localhost:11434")
    # Unset: one of the models already pulled on the server (asked in a terminal when
    # there are several). kuhaku never names a model of its own -- it would have to be
    # downloaded, and a fixed name goes stale.
    ollama_model: str | None = Field(default=None)

    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("KUHAKU_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    )
    anthropic_model: str = Field(default="claude-opus-5-5")

    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("KUHAKU_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    openai_model: str = Field(default="gpt-4o-mini")
    openai_base_url: str = Field(default="https://api.openai.com/v1")

    # Groq: OpenAI-compatible API, free tier as of writing. Added to the `auto` chain at
    # the end (a reachable local Ollama and every other credentialed provider are
    # preferred first -- see kuhaku.core.resolve.adapters.llm); prompt text still leaves
    # the machine, so it is never selected silently over a local option.
    groq_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("KUHAKU_GROQ_API_KEY", "GROQ_API_KEY"),
    )
    groq_model: str = Field(default="openai/gpt-oss-20b")
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1")

    # OpenRouter: OpenAI-compatible gateway. The default model is a free one; free models
    # come and go on OpenRouter, so this may need updating.
    openrouter_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("KUHAKU_OPENROUTER_API_KEY", "OPENROUTER_API_KEY"),
    )
    openrouter_model: str = Field(default="google/gemma-4-31b-it:free")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")

    # Google Vertex AI (optional, requires the `vertex` extra).
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
    vertex_model: str = Field(default="gemini-3.5-flash")

    # --- Default behavior policy ------------------------------------------------
    # Performance/helper components (BM25, cross-encoder reranker) configured as the
    # kuhaku default fall back to a simpler alternative and log a warning when they
    # fail to load, so the app still starts (kuhaku.core.policy.apply_fallback_policy).
    # Setting this True upgrades those failures to a startup-halting exception instead,
    # for operators who would rather fail loudly than silently run degraded.
    strict_performance_components: bool = Field(default=False)

    # --- Metrics (Prometheus) ----------------------------------------------
    # Metrics exposition is handled by the embedding application's own JWT-protected
    # admin metrics endpoint, not by kuhaku. The standalone metrics HTTP server is
    # disabled.
    metrics_enabled: bool = Field(default=False)

    # --- Security: prompt injection guard -----------------------------------
    input_guard_enabled: bool = Field(default=True)

    # --- Prompt Injection Guard v2 (normalize -> two-stage classify -> 3-zone) -------
    # Master kill switch for the whole v2 pipeline (normalizer, classifier, datamarked
    # prompt is unaffected -- see rag/prompts.py -- but the 3-zone decision, output guard,
    # and audit log are all dormant unless this is True). Defaults False, unlike this
    # project's other `*_enabled` toggles: with no Stage-2 model shipped (see
    # guard_stage2_onnx_path below), Stage-2 stays permanently degraded, which forces
    # every v2-guarded request into RESTRICTED regardless of Stage-1's score. The legacy
    # `input_guard_enabled` regex guard above is unaffected and keeps running either way.
    guard_enabled: bool = Field(default=False)
    guard_high_threshold: float = Field(default=0.7)
    guard_stage1_model_path: str = Field(default="")
    guard_stage2_onnx_path: str = Field(default="")
    guard_stage2_tokenizer_path: str = Field(default="")
    guard_norm_drift_tolerance: int = Field(default=5)
    guard_citation_grounding_threshold: float = Field(default=0.1)
    guard_model_version: str = Field(default="1.0.0")
    guard_version: str = Field(default="2.0.0")

    # --- Audit log (security/audit.py) ----------------------------------------------
    # Originally guard-v2-only; audit logging was later made unconditional -- every
    # request now writes a record here regardless of guard_enabled -- so this setting
    # moved out of the guard section and dropped its `guard_` prefix. Same file both
    # kinds of call site write to.
    #
    # audit_enabled: on by default -- a strong security default, explicit
    # opt-out only. audit_log_path unset/None/"" means "use the kuhaku-managed
    # default" (./logs/kuhaku_audit.jsonl, see security/audit.py); it is not itself
    # an enable/disable switch, so a caller can set an explicit path while still
    # disabling audit logging via audit_enabled=False.
    audit_enabled: bool = Field(default=True)
    audit_log_path: str | None = Field(default=None)

    # --- Retry: LLM, embedding, vector store, reranker call sites -------------------
    # Master kill switch for all four retry sites. All three LLM providers (Ollama,
    # Anthropic, OpenAI) share one generic RETRY_LLM_* config -- they are structurally
    # identical (requests.post -> raise_for_status -> LLMError, no vendor SDK) and the
    # project's LLM abstraction principle requires that swapping LLM_PROVIDER never
    # silently drops behavior. Per-provider tuning was discussed and deliberately
    # deferred.
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

    # --- LLM client-side rate limiting (core/llm/limits/) -----------------------------
    # Unset (the default) = no limiter, nothing changes. Setting any of these makes
    # build_llm_provider wrap the provider in a RateLimitedLLM backed by one process-wide
    # limiter per provider name ("groq:llama-3.3-70b-versatile"), so RAG() and an
    # Extractor in the same process share one quota. Per provider *and model* because
    # that is how Groq and OpenAI meter. Budgets are deliberately not here: a budget is
    # per run, and a run is something the calling code defines, not the environment.
    llm_rate_limit_requests_per_minute: float | None = Field(default=None)
    llm_rate_limit_tokens_per_minute: float | None = Field(default=None)
    llm_rate_limit_tokens_per_day: float | None = Field(default=None)

    # --- Model versioning --------------------------------------------------------------
    # Deployment-time label, set manually by whoever upgrades the LLM model -- not
    # auto-detected. Independent of the functional selector fields above
    # (ollama_model/anthropic_model/openai_model) by design: those choose which model
    # loads, this is a reproducibility label for what actually produced a response (can
    # intentionally diverge -- e.g. a provider swaps a checkpoint behind the same name).
    # RAG-specific versioning (embedding_model_version, prod_prompt_version,
    # eval_prompt_version) lives on RAGSettings.
    llm_model_version: str = Field(default="qwen2.5:7b-instruct-q4_k_m")


# The only fields an app_config (DB) row may override. An allowlist, not a denylist: the
# former denylist protected three keys and let a row set openai_base_url (sending the
# environment's API key to any host), groq_*/vertex_* endpoints and projects, or
# audit_enabled=false. Everything not listed here -- every *_api_key and *_base_url,
# vertex_project/location, and every audit and guard field -- is
# bootstrap-only: it comes from the process environment / dotenv file, never from a
# second store someone else may write to. A field added to Settings later is
# bootstrap-only until it is added here on purpose.
_DB_OVERRIDABLE_KEYS = frozenset(
    {
        "llm_provider",
        "llm_temperature",
        "llm_max_tokens",
        "llm_content_types",
        "ollama_model",
        "anthropic_model",
        "openai_model",
        "openrouter_model",
        "groq_model",
        "vertex_model",
        "strict_performance_components",
        "metrics_enabled",
        "retry_enabled",
        "retry_llm_max_attempts",
        "retry_llm_backoff_base_seconds",
        "retry_llm_backoff_max_seconds",
        "llm_timeout_seconds",
        "circuit_breaker_enabled",
        "circuit_breaker_failure_threshold",
        "circuit_breaker_reset_timeout_seconds",
        "circuit_breaker_success_threshold",
        "llm_rate_limit_requests_per_minute",
        "llm_rate_limit_tokens_per_minute",
        "llm_rate_limit_tokens_per_day",
        "llm_model_version",
    }
)


def load_settings(
    app_config_loader: Any = None,
    env_file: str | None = None,
) -> Settings:
    """Layer defaults < ``.env``/env < ``app_config`` (DB, allowlisted keys only).

    ``base`` already reflects today's defaults + ``.env``/environment (pydantic-settings'
    normal behavior, unchanged). Every ``app_config`` key in ``_DB_OVERRIDABLE_KEYS`` is
    then overlaid; any other key (secrets, endpoints, audit, guard, unknown names) is
    ignored with a warning naming it. The allowed keys are overlaid on top of ``base``'s own dump, and re-passed as explicit constructor
    kwargs -- pydantic-settings' documented precedence (init kwargs > env > dotenv >
    defaults) makes that overlay win over ``.env`` automatically, and pydantic's lax
    coercion turns the DB's ``TEXT`` values back into ``bool``/``int``/``float``.

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

    ignored = sorted(key for key in stored if key not in _DB_OVERRIDABLE_KEYS)
    if ignored:
        logger.warning(
            "app_config keys ignored (bootstrap-only or unknown; set them in the "
            "environment): %s",
            ", ".join(ignored),
        )
    merged = base.model_dump()
    for key, value in stored.items():
        if key in _DB_OVERRIDABLE_KEYS:
            merged[key] = value
    return Settings(_env_file=env_file, **merged)


@lru_cache
def get_settings(
    app_config_loader: Any = None,
    env_file: str | None = None,
) -> Settings:
    """Return a cached ``Settings`` instance, refreshed via :func:`load_settings`.

    Runtime config changes made through the embedding application's admin config
    endpoint do not rely on this cache being cleared to take effect -- its
    reconfiguration path mutates the one shared
    ``Settings`` instance every route closure already holds a reference to, in place.
    ``cache_clear()`` is called defensively after a write anyway, for any future code
    path that constructs ``Settings`` fresh.

    ``app_config_loader``/``env_file`` are forwarded to :func:`load_settings`; see its
    docstring. Kuhaku-only callers that never need the DB layer or a dotenv file omit
    both.
    """

    return load_settings(app_config_loader, env_file)
