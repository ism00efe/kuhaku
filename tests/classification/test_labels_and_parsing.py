"""The label set and reply parsing: what counts as a label, and what counts as an answer."""

from __future__ import annotations

import pytest

from kuhaku.core.exceptions import ConfigError
from kuhaku.tools.classification import (
    Label,
    LabelSet,
    OutputParseError,
    UndecidedReason,
)
from kuhaku.tools.classification.labels import as_label_set
from kuhaku.tools.classification.parsing import parse_reply

from .conftest import LABELS

LABEL_SET = LabelSet(LABELS)


def test_bare_names_become_labels_and_order_is_kept():
    labels = LabelSet(["spam", Label("ham", "Legitimate mail")])
    assert labels.names == ("spam", "ham")
    assert labels["ham"].description == "Legitimate mail"
    assert "spam" in labels and len(labels) == 2


def test_duplicate_names_are_rejected_case_insensitively():
    with pytest.raises(ConfigError, match="defined twice"):
        LabelSet(["Billing", "billing"])


@pytest.mark.parametrize("name", ["", "   ", " padded "])
def test_label_names_must_be_clean(name):
    with pytest.raises(ConfigError):
        Label(name)


def test_an_empty_label_set_is_rejected():
    with pytest.raises(ConfigError, match="at least one label"):
        LabelSet([])


def test_a_single_string_is_not_split_into_characters():
    with pytest.raises(ConfigError, match="not a single string"):
        as_label_set("spam")


def test_examples_given_as_a_list_are_stored_as_a_tuple():
    assert Label("spam", examples=["buy now"]).examples == ("buy now",)


def test_resolve_tolerates_case_and_whitespace_but_nothing_fuzzier():
    assert LABEL_SET.resolve("billing") == "billing"
    assert LABEL_SET.resolve(" Billing ") == "billing"
    assert LABEL_SET.resolve("bill") is None
    assert LABEL_SET.resolve(None) is None


def test_a_valid_reply_maps_to_canonical_names():
    reply = parse_reply(
        '{"labels": [{"label": "Billing", "confidence": 0.9}], "reason": "a charge"}', LABEL_SET
    )
    assert reply.scores == {"billing": 0.9}
    assert reply.rationale == "a charge"


def test_a_code_fence_and_surrounding_prose_are_tolerated():
    raw = 'Sure:\n```json\n{"labels": [{"label": "technical", "confidence": 1}]}\n```'
    assert parse_reply(raw, LABEL_SET).scores == {"technical": 1.0}


def test_null_or_missing_labels_mean_no_label_fits():
    assert parse_reply('{"labels": null}', LABEL_SET).scores == {}
    assert parse_reply("{}", LABEL_SET).scores == {}


def test_a_repeated_label_keeps_its_highest_confidence():
    raw = '{"labels": [{"label": "billing", "confidence": 0.3}, {"label": "BILLING", "confidence": 0.8}]}'
    assert parse_reply(raw, LABEL_SET).scores == {"billing": 0.8}


def test_unknown_labels_are_reported_not_dropped():
    raw = '{"labels": [{"label": "billing", "confidence": 0.9}, {"label": "refunds", "confidence": 0.8}]}'
    with pytest.raises(OutputParseError) as caught:
        parse_reply(raw, LABEL_SET)
    assert caught.value.reason is UndecidedReason.UNKNOWN_LABEL
    assert "'refunds' is not in the label set" in caught.value.problems[0]
    assert "billing, technical, account" in caught.value.problems[0]


@pytest.mark.parametrize(
    "raw",
    [
        "not json at all",
        "[1, 2]",
        '{"labels": "billing"}',
        '{"labels": ["billing"]}',
        '{"labels": [{"label": "billing"}]}',
        '{"labels": [{"label": "billing", "confidence": 1.5}]}',
        '{"labels": [{"label": "billing", "confidence": true}]}',
        '{"labels": [{"label": "billing", "confidence": "0.9"}]}',
    ],
)
def test_malformed_replies_are_unparseable(raw):
    with pytest.raises(OutputParseError) as caught:
        parse_reply(raw, LABEL_SET)
    assert caught.value.reason is UndecidedReason.UNPARSEABLE


def test_every_problem_is_collected_for_one_repair():
    raw = '{"labels": [{"label": "billing", "confidence": 2}, {"label": "nope", "confidence": 0.5}]}'
    with pytest.raises(OutputParseError) as caught:
        parse_reply(raw, LABEL_SET)
    assert len(caught.value.problems) == 2
    # A format problem alongside an unknown name is reported as the broader failure.
    assert caught.value.reason is UndecidedReason.UNPARSEABLE
