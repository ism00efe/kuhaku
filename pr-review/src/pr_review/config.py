"""Layered configuration.

Precedence (low to high): built-in defaults -> repo ``.pr-review.toml`` ->
``PR_REVIEW_*`` environment variables -> explicit overrides (CLI flags).

Nothing here is repository-specific. A repo tunes behaviour by dropping a
``.pr-review.toml`` at its root; the core never grows a per-repo branch.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from pr_review.errors import ConfigError

# --------------------------------------------------------------------------- #
# Sub-structures
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProviderConfig:
    kind: str  # registry name: "mock" | "openai_compat" | "gemini"
    base_url: str = ""
    api_key_env: str = ""
    default_model: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "").strip() if self.api_key_env else ""


@dataclass(frozen=True)
class TierConfig:
    """A model tier binds a logical role to a provider + concrete model.

    ``fallbacks`` are further ``(provider, model)`` pairs tried, in order, when
    the primary is throttled or serves no such model. An empty model means "use
    that provider's ``default_model``"; a pair whose provider has no API key or
    no resolvable model is silently skipped, so an unconfigured fallback costs
    nothing.
    """

    provider: str
    model: str
    max_tokens: int = 1400
    temperature: float = 0.2
    fallbacks: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Candidate:
    """One concrete attempt for a tier, in priority order.

    Produced by :meth:`Config.resolve_chain`. ``tier`` is the tier that
    contributed it, which differs from the requested one once degradation
    kicks in -- that is what ``degraded`` marks, so the report can say so.
    """

    provider: str
    model: str
    max_tokens: int
    temperature: float
    tier: str
    degraded: bool = False

    def label(self) -> str:
        return f"{self.provider}:{self.model}"


DEFAULT_PROVIDERS: dict[str, ProviderConfig] = {
    "mock": ProviderConfig(kind="mock", default_model="mock"),
    "groq": ProviderConfig(
        kind="openai_compat",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        default_model="openai/gpt-oss-20b",
    ),
    "openrouter": ProviderConfig(
        kind="openai_compat",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        default_model="z-ai/glm-5.2:free",
    ),
    # OpenCode Zen: OpenAI-compatible, free coding-model catalogue. No default
    # model on purpose -- its free entries are time-limited trials that rotate,
    # so an id is only ever pinned by a repo that has checked it is still live.
    "opencode": ProviderConfig(
        kind="openai_compat",
        base_url="https://opencode.ai/zen/v1",
        api_key_env="OPENCODE_API_KEY",
        default_model="nemotron-3-ultra-free",
    ),
    "gemini": ProviderConfig(
        kind="gemini",
        api_key_env="GEMINI_API_KEY",
        default_model="gemini-2.5-flash",
    ),
}

# Logical roles, mapped onto the two free tiers whose limits are shaped
# oppositely:
#   Groq       -- many small requests (30 RPM) but a tight 8K tokens/minute
#   OpenRouter -- token-blind, capped on request count (1000 :free calls/day)
# Planning and basic reviews are small and frequent, so they go to Groq; the
# token-heavy normal/deep/verify work goes to OpenRouter, where a 6-7K-token
# prompt costs one request instead of most of a minute's token budget.
DEFAULT_TIERS: dict[str, TierConfig] = {
    "planner": TierConfig(
        "groq", "openai/gpt-oss-20b", max_tokens=700,
        fallbacks=(("openrouter", "minimax/minimax-m3:free"),),
    ),
    "basic": TierConfig(
        "groq", "openai/gpt-oss-20b", max_tokens=1200,
        fallbacks=(("openrouter", "minimax/minimax-m3:free"),),
    ),
    "normal": TierConfig(
        "openrouter", "minimax/minimax-m3:free", max_tokens=1600,
        fallbacks=(
            ("openrouter", "z-ai/glm-5.2:free"),
            ("groq", "openai/gpt-oss-120b"),
        ),
    ),
    "deep": TierConfig(
        "openrouter", "minimax/minimax-m3:free", max_tokens=2200,
        fallbacks=(
            ("openrouter", "z-ai/glm-5.2:free"),
            ("opencode", "nemotron-3-ultra-free"),
            ("groq", "openai/gpt-oss-120b"),
        ),
    ),
    # Leads with a different model than deep on purpose: a model asked to check
    # its own output tends to agree with it.
    "verify": TierConfig(
        "openrouter", "z-ai/glm-5.2:free", max_tokens=1200,
        fallbacks=(
            ("openrouter", "minimax/minimax-m3:free"),
            ("groq", "openai/gpt-oss-120b"),
        ),
    ),
}

# When every candidate of a tier is exhausted, borrow the next tier's models
# rather than dropping the task. The review still runs, on a weaker model, and
# the report records that it was degraded.
DEFAULT_TIER_DEGRADATION: dict[str, tuple[str, ...]] = {
    "deep": ("normal", "basic"),
    "normal": ("basic",),
    "verify": ("normal", "basic"),
    "basic": (),
    "planner": (),
}

DEFAULT_DEPTH_TIER = {"basic": "basic", "normal": "normal", "deep": "deep"}


@dataclass(frozen=True)
class Limits:
    diff_bytes: int = 16_000
    aux_bytes: int = 8_000
    max_changed_files_listed: int = 60
    context_max_bytes: int = 24_000


@dataclass(frozen=True)
class Concurrency:
    workers: int = 1
    axis_delay_seconds: float = 0.0
    retry_wait_seconds: float = 60.0
    max_retries: int = 2


@dataclass(frozen=True)
class VerificationConfig:
    enabled: bool = True
    llm_enabled: bool = True
    # A finding goes to LLM verification if it is at least this severe OR its
    # confidence is within the uncertain band, and deterministic checks did not
    # already settle it.
    llm_min_severity: str = "warning"
    uncertain_low: float = 0.35
    uncertain_high: float = 0.8
    max_llm_verifications: int = 6


# --------------------------------------------------------------------------- #
# Top-level config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Config:
    providers: dict[str, ProviderConfig] = field(
        default_factory=lambda: dict(DEFAULT_PROVIDERS)
    )
    tiers: dict[str, TierConfig] = field(default_factory=lambda: dict(DEFAULT_TIERS))
    depth_tier: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_DEPTH_TIER))
    tier_degradation: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_TIER_DEGRADATION)
    )

    # Last-resort provider appended to every chain. Empty by default: with no
    # key configured the engine reports a structural-only review rather than
    # inventing findings. Set to "mock" only for tests and offline demos.
    fallback_provider: str = ""
    # When set, every tier is forced to this provider (CLI --provider).
    force_provider: str = ""

    enabled_axes: tuple[str, ...] = ("correctness", "method", "scope", "structure")
    enabled_depths: tuple[str, ...] = ("basic", "normal", "deep")
    discoverers: tuple[str, ...] = ("languages", "manifests", "layout")
    deterministic_analyzers: tuple[str, ...] = ("core",)
    context_strategy: str = "default"
    verifiers: tuple[str, ...] = ("deterministic", "llm")
    reporters: tuple[str, ...] = ("markdown", "json")

    classifier_enabled: bool = True
    max_axis_findings: int = 6
    max_total_findings: int = 12

    limits: Limits = Limits()
    concurrency: Concurrency = Concurrency()
    verification: VerificationConfig = VerificationConfig()

    repo_root: str = "."

    # ----------------------------------------------------------------- #

    def tier(self, name: str) -> TierConfig:
        try:
            return self.tiers[name]
        except KeyError:
            raise ConfigError(f"no model tier {name!r}") from None

    def provider(self, name: str) -> ProviderConfig:
        try:
            return self.providers[name]
        except KeyError:
            raise ConfigError(f"no provider {name!r}") from None

    def resolve_chain(self, tier_name: str) -> list[Candidate]:
        """Ordered, usable attempts for a tier: primary, its fallbacks, then the
        degradation tiers' models, then ``fallback_provider``.

        Candidates that cannot possibly work -- unknown provider, no API key in
        the environment, no model id to send -- are dropped here, so callers
        never spend a request discovering it. An empty list means this tier has
        no usable model at all.
        """
        tier = self.tier(tier_name)

        if self.force_provider:
            prov = self.provider(self.force_provider)
            model = prov.default_model or tier.model
            return [
                Candidate(
                    self.force_provider, model, tier.max_tokens, tier.temperature, tier_name
                )
            ]

        out: list[Candidate] = []
        seen: set[tuple[str, str]] = set()

        def add(source_tier: str, provider_name: str, model: str, degraded: bool) -> None:
            prov = self.providers.get(provider_name)
            if prov is None:
                return
            model = model or prov.default_model
            if not model:
                return
            if prov.api_key_env and not prov.api_key():
                return
            key = (provider_name, model)
            if key in seen:
                return
            seen.add(key)
            src = self.tiers.get(source_tier, tier)
            out.append(
                Candidate(
                    provider_name, model, src.max_tokens, src.temperature,
                    source_tier, degraded,
                )
            )

        ladder: list[tuple[str, bool]] = [(tier_name, False)]
        ladder += [(t, True) for t in self.tier_degradation.get(tier_name, ())]
        for source_tier, degraded in ladder:
            t = self.tiers.get(source_tier)
            if t is None:
                continue
            add(source_tier, t.provider, t.model, degraded)
            for provider_name, model in t.fallbacks:
                add(source_tier, provider_name, model, degraded)

        if self.fallback_provider:
            add(tier_name, self.fallback_provider, "", False)
        return out

    def resolve_provider(self, tier_name: str) -> tuple[str, str]:
        """First usable ``(provider_name, model)`` for a tier.

        Kept as the single-candidate view of :meth:`resolve_chain`.
        """
        chain = self.resolve_chain(tier_name)
        if not chain:
            tier = self.tier(tier_name)
            return tier.provider, tier.model
        return chain[0].provider, chain[0].model

    def has_llm(self) -> bool:
        """True when at least one tier has a usable model configured."""
        return any(self.resolve_chain(name) for name in self.tiers)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _merge_providers(
    base: dict[str, ProviderConfig], data: dict[str, Any]
) -> dict[str, ProviderConfig]:
    out = dict(base)
    for name, raw in (data or {}).items():
        cur = out.get(name)
        out[name] = ProviderConfig(
            kind=raw.get("kind", cur.kind if cur else "openai_compat"),
            base_url=raw.get("base_url", cur.base_url if cur else ""),
            api_key_env=raw.get("api_key_env", cur.api_key_env if cur else ""),
            default_model=raw.get("default_model", cur.default_model if cur else ""),
            extra=raw.get("extra", cur.extra if cur else {}),
        )
    return out


def _parse_fallbacks(raw: Any) -> tuple[tuple[str, str], ...] | None:
    """Accept ``[{provider=..., model=...}]``, ``[["groq", "m"]]`` or ``["groq"]``."""
    if raw is None:
        return None
    out: list[tuple[str, str]] = []
    for item in raw:
        if isinstance(item, dict):
            provider = str(item.get("provider", "")).strip()
            model = str(item.get("model", "")).strip()
        elif isinstance(item, (list, tuple)):
            provider = str(item[0]).strip() if item else ""
            model = str(item[1]).strip() if len(item) > 1 else ""
        else:
            provider, model = str(item).strip(), ""
        if provider:
            out.append((provider, model))
    return tuple(out)


def _merge_tiers(base: dict[str, TierConfig], data: dict[str, Any]) -> dict[str, TierConfig]:
    out = dict(base)
    for name, raw in (data or {}).items():
        cur = out.get(name)
        fallbacks = _parse_fallbacks(raw.get("fallbacks"))
        if fallbacks is None:
            fallbacks = cur.fallbacks if cur else ()
        out[name] = TierConfig(
            provider=raw.get("provider", cur.provider if cur else "mock"),
            model=raw.get("model", cur.model if cur else ""),
            max_tokens=int(raw.get("max_tokens", cur.max_tokens if cur else 1400)),
            temperature=float(raw.get("temperature", cur.temperature if cur else 0.2)),
            fallbacks=fallbacks,
        )
    return out


def load_config(
    repo_root: str | os.PathLike[str] = ".",
    *,
    overrides: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
) -> Config:
    env = dict(os.environ if env is None else env)
    cfg = Config(repo_root=str(repo_root))

    toml_path = Path(repo_root) / ".pr-review.toml"
    if toml_path.is_file():
        try:
            data = tomllib.loads(toml_path.read_text("utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{toml_path}: {exc}") from exc
        cfg = _apply_mapping(cfg, data)

    cfg = _apply_env(cfg, env)

    if overrides:
        cfg = _apply_mapping(cfg, overrides)

    return cfg


def _apply_mapping(cfg: Config, data: dict[str, Any]) -> Config:
    changes: dict[str, Any] = {}
    if "providers" in data:
        changes["providers"] = _merge_providers(cfg.providers, data["providers"])
    if "model_tiers" in data:
        changes["tiers"] = _merge_tiers(cfg.tiers, data["model_tiers"])
    if "tiers" in data:
        changes["tiers"] = _merge_tiers(cfg.tiers, data["tiers"])
    if "depth_tier" in data:
        changes["depth_tier"] = {**cfg.depth_tier, **data["depth_tier"]}
    if "tier_degradation" in data:
        changes["tier_degradation"] = {
            **cfg.tier_degradation,
            **{k: tuple(v) for k, v in data["tier_degradation"].items()},
        }

    for key in (
        "fallback_provider",
        "force_provider",
        "context_strategy",
        "classifier_enabled",
        "max_axis_findings",
        "max_total_findings",
        "repo_root",
    ):
        if key in data:
            changes[key] = data[key]
    for key in (
        "enabled_axes",
        "enabled_depths",
        "discoverers",
        "deterministic_analyzers",
        "verifiers",
        "reporters",
    ):
        if key in data:
            changes[key] = tuple(data[key])

    if "limits" in data:
        changes["limits"] = replace(cfg.limits, **data["limits"])
    if "concurrency" in data:
        changes["concurrency"] = replace(cfg.concurrency, **data["concurrency"])
    if "verification" in data:
        changes["verification"] = replace(cfg.verification, **data["verification"])

    return replace(cfg, **changes)


_ENV_KEYS = {
    "PR_REVIEW_PROVIDER": "force_provider",
    "PR_REVIEW_FALLBACK_PROVIDER": "fallback_provider",
    "PR_REVIEW_CONTEXT_STRATEGY": "context_strategy",
}


def _apply_env(cfg: Config, env: dict[str, str]) -> Config:
    changes: dict[str, Any] = {}
    for env_key, attr in _ENV_KEYS.items():
        if env.get(env_key):
            changes[attr] = env[env_key].strip()

    if env.get("PR_REVIEW_CLASSIFIER"):
        changes["classifier_enabled"] = env["PR_REVIEW_CLASSIFIER"].lower() not in (
            "0",
            "false",
            "off",
            "no",
        )
    if env.get("PR_REVIEW_AXES"):
        changes["enabled_axes"] = tuple(
            a.strip() for a in env["PR_REVIEW_AXES"].split(",") if a.strip()
        )
    if env.get("PR_REVIEW_DIFF_LIMIT"):
        changes["limits"] = replace(cfg.limits, diff_bytes=int(env["PR_REVIEW_DIFF_LIMIT"]))
    if env.get("PR_REVIEW_MAX_CONCURRENCY"):
        changes["concurrency"] = replace(
            cfg.concurrency, workers=max(1, int(env["PR_REVIEW_MAX_CONCURRENCY"]))
        )
    if env.get("PR_REVIEW_AXIS_DELAY"):
        base = changes.get("concurrency", cfg.concurrency)
        changes["concurrency"] = replace(
            base, axis_delay_seconds=float(env["PR_REVIEW_AXIS_DELAY"])
        )

    # Tier model overrides: PR_REVIEW_TIER_DEEP_MODEL=... etc.
    tiers = dict(cfg.tiers)
    touched = False
    for name in list(tiers):
        mk = f"PR_REVIEW_TIER_{name.upper()}_MODEL"
        pk = f"PR_REVIEW_TIER_{name.upper()}_PROVIDER"
        if env.get(mk) or env.get(pk):
            touched = True
            tiers[name] = replace(
                tiers[name],
                model=env.get(mk, tiers[name].model),
                provider=env.get(pk, tiers[name].provider),
            )
    if touched:
        changes["tiers"] = tiers

    return replace(cfg, **changes) if changes else cfg
