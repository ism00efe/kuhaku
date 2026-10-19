"""PromptRegistry: versions, latest/pin, conflicts, layering over defaults."""

from __future__ import annotations

import pytest

from kuhaku.core.prompts import (
    PromptConflictError,
    PromptNotFoundError,
    PromptRegistry,
    PromptTemplate,
    version_key,
)


def _t(name: str, version: str, text: str = "text") -> PromptTemplate:
    return PromptTemplate(name, version, text)


def test_latest_is_the_highest_version_not_the_last_registered():
    registry = PromptRegistry([_t("p", "v10", "ten"), _t("p", "v9", "nine"), _t("p", "v2", "two")])
    assert registry.get("p").version == "v10"
    assert registry.versions("p") == ("v2", "v9", "v10")


@pytest.mark.parametrize(
    "lower, higher", [("1.9", "1.10"), ("v2", "v10"), ("2", "2.1"), ("1", "1a")]
)
def test_version_key_orders_naturally(lower, higher):
    assert version_key(lower) < version_key(higher)


def test_specific_version_and_missing_lookups():
    registry = PromptRegistry([_t("p", "1", "a"), _t("p", "2", "b")])
    assert registry.get("p", "1").template == "a"
    with pytest.raises(PromptNotFoundError, match="version '3'"):
        registry.get("p", "3")
    with pytest.raises(PromptNotFoundError, match="No prompt named 'q'"):
        registry.get("q")


def test_reregistering_identical_text_is_a_noop_but_different_text_conflicts():
    registry = PromptRegistry([_t("p", "1", "a")])
    registry.register(_t("p", "1", "a"))
    with pytest.raises(PromptConflictError):
        registry.register(_t("p", "1", "changed"))
    registry.register(_t("p", "1", "changed"), replace=True)
    assert registry.get("p").template == "changed"


def test_pin_overrides_latest_and_must_exist():
    registry = PromptRegistry([_t("p", "1", "a"), _t("p", "2", "b")])
    registry.pin("p", "1")
    assert registry.get("p").version == "1"
    with pytest.raises(PromptNotFoundError):
        registry.pin("p", "9")
    registry.unpin("p")
    assert registry.get("p").version == "2"


def test_chain_prefers_the_first_registry_and_falls_through_for_other_names():
    defaults = PromptRegistry([_t("a", "1", "default a"), _t("b", "1", "default b")])
    user = PromptRegistry([_t("a", "1-custom", "user a")])
    chained = PromptRegistry.chain(user, defaults)

    assert chained.get("a").template == "user a"  # user's name shadows the default
    assert chained.get("a", "1").template == "default a"  # default still reachable
    assert chained.get("b").template == "default b"
    assert chained.names() == ("a", "b")
    assert chained.versions("a") == ("1", "1-custom")
    assert {(t.name, t.version) for t in chained.templates()} == {
        ("a", "1"), ("a", "1-custom"), ("b", "1"),
    }


def test_chain_is_live_and_honours_pins_on_the_user_registry():
    defaults = PromptRegistry([_t("a", "1", "default")])
    user = PromptRegistry()
    chained = PromptRegistry.chain(user, defaults)
    assert chained.get("a").template == "default"

    user.register(_t("a", "2", "new"))
    assert chained.get("a").template == "new"

    user.register(_t("a", "3", "newer"))
    user.pin("a", "2")
    assert chained.get("a").template == "new"


def test_chain_skips_none():
    defaults = PromptRegistry([_t("a", "1")])
    assert PromptRegistry.chain(None, defaults).get("a").version == "1"


def test_render_returns_text_and_ref():
    registry = PromptRegistry([PromptTemplate("g", "1", "Hi {{who}}")])
    rendered = registry.render("g", who="Ada")
    assert rendered.text == "Hi Ada"
    assert rendered.ref.name == "g" and rendered.ref.version == "1"


def test_contains_and_len():
    registry = PromptRegistry([_t("a", "1"), _t("a", "2")])
    assert "a" in registry and "b" not in registry
    assert len(registry) == 2
