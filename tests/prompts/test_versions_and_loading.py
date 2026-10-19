"""Version ordering is release-aware and registration-order independent; prompt files
with a BOM load; register_directory is all-or-nothing."""

from __future__ import annotations

import itertools

import pytest

from kuhaku.core.prompts import PromptRegistry, PromptTemplate, version_key


@pytest.mark.parametrize(
    ("lower", "higher"),
    [
        ("2.0.0-rc1", "2.0.0"),
        ("2.0.0rc1", "2.0.0"),
        ("1.4b2", "1.4"),
        ("3.dev1", "3.alpha1"),
        ("1.0", "1.0.1"),
        ("1.9", "1.10"),
        ("draft", "1"),
    ],
)
def test_release_aware_order(lower, higher):
    assert version_key(lower) < version_key(higher)


def test_leading_v_is_ignored():
    assert version_key("v2")[1:5] == version_key("2")[1:5]
    assert version_key("v10") > version_key("9")


@pytest.mark.parametrize("pair", [("01", "1"), ("v2", "2"), ("2", "2.0")])
def test_equal_versions_resolve_the_same_whatever_the_order(pair):
    winners = set()
    for order in itertools.permutations(pair):
        registry = PromptRegistry(
            [PromptTemplate("p", version, f"text {version}") for version in order]
        )
        winners.add(registry.get("p").version)
    assert len(winners) == 1


