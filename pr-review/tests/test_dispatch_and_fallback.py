"""Free-tier failure behaviour: failover, retry, degradation, structural-only.

These are the paths that only ever run in production -- a quota that ran out, a
free model that was retired overnight -- so they are exercised here with fake
transports rather than left to be discovered on a real pull request.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from pr_review.config import Config, ProviderConfig, TierConfig
from pr_review.errors import NoProviderAvailable, ProviderUnavailable, RateLimited
from pr_review.pipeline import Pipeline
from pr_review.providers import dispatch as dispatch_mod
from pr_review.providers.base import PROVIDERS
from pr_review.providers.dispatch import Dispatcher
from pr_review.providers.selector import ModelSelector
from pr_review.report.base import REPORTERS
from pr_review.source.local_git import LocalGitSource

CALLS: Counter[str] = Counter()


class _Base:
    def __init__(self, cfg: ProviderConfig) -> None:
        self.cfg = cfg


@PROVIDERS.register("t_gone")
class _Gone(_Base):
    """Model retired from the catalogue: permanent, must not be retried."""

    name = "t_gone"

    def complete(self, prompt, *, model, max_tokens=1400, temperature=0.2, system=None):
        CALLS["gone"] += 1
        raise ProviderUnavailable(f"404 from t_gone: unknown model {model}")


@PROVIDERS.register("t_throttled")
class _Throttled(_Base):
    """Daily quota exhausted: transient, worth the retry budget."""

    name = "t_throttled"

    def complete(self, prompt, *, model, max_tokens=1400, temperature=0.2, system=None):
        CALLS["throttled"] += 1
        raise RateLimited("429 from t_throttled", retry_after=0.0)


@PROVIDERS.register("t_ok")
class _Ok(_Base):
    name = "t_ok"

    def complete(self, prompt, *, model, max_tokens=1400, temperature=0.2, system=None):
        CALLS["ok"] += 1
        return '{"answer": "yes"}'


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    CALLS.clear()
    for var in ("GONE_KEY", "THROTTLED_KEY", "OK_KEY"):
        monkeypatch.setenv(var, "present")
    # Never actually wait during the retry tests.
    monkeypatch.setattr(dispatch_mod.time, "sleep", lambda _s: None)


def _config(primary_kind: str, *, degradation: tuple[str, ...] = ()) -> Config:
    providers = {
        "gone": ProviderConfig(kind="t_gone", api_key_env="GONE_KEY", default_model="m-gone"),
        "throttled": ProviderConfig(
            kind="t_throttled", api_key_env="THROTTLED_KEY", default_model="m-throttled"
        ),
        "ok": ProviderConfig(kind="t_ok", api_key_env="OK_KEY", default_model="m-ok"),
        "unkeyed": ProviderConfig(
            kind="t_ok", api_key_env="ABSENT_KEY", default_model="m-unkeyed"
        ),
    }
    primary = {"t_gone": "gone", "t_throttled": "throttled", "t_ok": "ok"}[primary_kind]
    return Config(
        providers=providers,
        tiers={
            "deep": TierConfig(primary, providers[primary].default_model, max_tokens=2200),
            "basic": TierConfig("ok", "m-ok", max_tokens=1200),
        },
        tier_degradation={"deep": degradation, "basic": ()},
        fallback_provider="",
    )


def _dispatcher(cfg: Config) -> Dispatcher:
    return Dispatcher(cfg, ModelSelector(cfg))


# --------------------------------------------------------------------------- #


def test_retired_model_fails_over_immediately():
    cfg = _config("t_gone")
    cfg.tiers["deep"] = TierConfig("gone", "m-gone", fallbacks=(("ok", "m-ok"),))
    d = _dispatcher(cfg)

    completion = d.complete("deep", "hello")

    assert completion.text == '{"answer": "yes"}'
    assert completion.candidate.provider == "ok"
    # The whole point: a 404 costs exactly one request, not max_retries + 1.
    assert CALLS["gone"] == 1
    assert any("unavailable" in n for n in d.notes)


def test_throttled_model_exhausts_retries_then_fails_over():
    cfg = _config("t_throttled")
    cfg.tiers["deep"] = TierConfig("throttled", "m-throttled", fallbacks=(("ok", "m-ok"),))
    d = _dispatcher(cfg)

    completion = d.complete("deep", "hello")

    assert completion.candidate.provider == "ok"
    assert CALLS["throttled"] == cfg.concurrency.max_retries + 1
    assert any("rate-limited" in n for n in d.notes)


def test_exhausted_tier_degrades_to_the_next_tier():
    cfg = _config("t_gone", degradation=("basic",))
    d = _dispatcher(cfg)

    completion = d.complete("deep", "hello")

    assert completion.degraded
    assert completion.candidate.tier == "basic"
    assert completion.candidate.max_tokens == 1200  # the borrowed tier's budget
    assert any("degraded to the basic tier" in n for n in d.notes)


def test_keyless_candidates_are_dropped_before_any_request():
    cfg = _config("t_ok")
    cfg.tiers["deep"] = TierConfig("unkeyed", "m-unkeyed", fallbacks=(("ok", "m-ok"),))

    chain = cfg.resolve_chain("deep")

    assert [c.provider for c in chain] == ["ok"]


def test_no_usable_candidate_raises_rather_than_returning_text():
    cfg = _config("t_ok")
    cfg.tiers["deep"] = TierConfig("unkeyed", "m-unkeyed")
    cfg.tier_degradation["deep"] = ()
    d = _dispatcher(cfg)

    assert cfg.resolve_chain("deep") == []
    with pytest.raises(NoProviderAvailable):
        d.complete("deep", "hello")


# --------------------------------------------------------------------------- #
# Run memory. A review makes one call per axis plus one per verification; a
# dispatcher without memory pays the full retry price on every one of them for
# a model it already proved dead. On a free tier that is the scarce resource.
# --------------------------------------------------------------------------- #


def test_a_retired_model_is_not_probed_again_later_in_the_run():
    cfg = _config("t_gone")
    cfg.tiers["deep"] = TierConfig("gone", "m-gone", fallbacks=(("ok", "m-ok"),))
    d = _dispatcher(cfg)

    for _ in range(4):
        assert d.complete("deep", "hello").candidate.provider == "ok"

    # One request total, not one per call.
    assert CALLS["gone"] == 1
    assert CALLS["ok"] == 4
    assert any("dropped for the rest of this run" in n for n in d.notes)


def test_a_throttled_model_is_benched_rather_than_retried_every_call():
    cfg = _config("t_throttled")
    cfg.tiers["deep"] = TierConfig("throttled", "m-throttled", fallbacks=(("ok", "m-ok"),))
    d = _dispatcher(cfg)

    for _ in range(4):
        assert d.complete("deep", "hello").candidate.provider == "ok"

    # The retry budget is spent once, on the first call, not four times over.
    assert CALLS["throttled"] == cfg.concurrency.max_retries + 1
    assert CALLS["ok"] == 4
    assert any("benched for" in n for n in d.notes)


def test_benching_never_permanently_disables_a_tier():
    """A throttle is temporary; if it is the only candidate we must try again."""
    cfg = _config("t_throttled")
    cfg.tiers["deep"] = TierConfig("throttled", "m-throttled")
    cfg.tier_degradation["deep"] = ()
    d = _dispatcher(cfg)

    for _ in range(2):
        with pytest.raises(NoProviderAvailable):
            d.complete("deep", "hello")

    # Both calls really tried: benching is waived when nothing else is left.
    assert CALLS["throttled"] == 2 * (cfg.concurrency.max_retries + 1)


def test_a_permanently_dead_only_candidate_stops_costing_requests():
    cfg = _config("t_gone")
    cfg.tiers["deep"] = TierConfig("gone", "m-gone")
    cfg.tier_degradation["deep"] = ()
    d = _dispatcher(cfg)

    for _ in range(3):
        with pytest.raises(NoProviderAvailable):
            d.complete("deep", "hello")

    # Unlike a throttle, a 404 is not waived -- one request, then silence.
    assert CALLS["gone"] == 1


# --------------------------------------------------------------------------- #


def test_structural_only_run_invents_nothing(tiny_repo: Path, monkeypatch):
    for var in (
        "GROQ_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENCODE_API_KEY",
        "GEMINI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)

    from pr_review.config import load_config

    cfg = load_config(tiny_repo)
    assert not cfg.has_llm()

    result = Pipeline(cfg).run(
        LocalGitSource(tiny_repo, base_ref="main", head_ref="feature")
    )

    assert result.structural_only
    assert result.stats["mode"] == "structural-only"
    assert result.findings == []
    # The deterministic half still did its job.
    assert result.analysis.changed_file_count >= 2
    assert result.plan.active()
    assert result.plan.source == "heuristic"

    md = REPORTERS.create("markdown").render(result)
    assert "Structural analysis only" in md
    assert "placeholder" not in md
    assert "[mock:" not in md
