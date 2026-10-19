"""Optional steps (rules, validation) and pipeline composition."""

from __future__ import annotations

import re

import pytest

from kuhaku.core.exceptions import ConfigError, CustomComponentError
from kuhaku.tools.classification import (
    ClassificationContext,
    Classifier,
    ClassifyStep,
    KeywordRule,
    RegexRule,
    RuleStep,
    Step,
    UndecidedReason,
    ValidationStep,
)

from .conftest import LABELS, ScriptedLLM, answer

# --- rules --------------------------------------------------------------------------------


def test_a_keyword_rule_decides_without_calling_the_model():
    llm = ScriptedLLM()  # any call fails the test
    clf = Classifier(LABELS, llm=llm, rules=[KeywordRule("billing", ["invoice", "refund"])])
    result = clf.classify("Where is my REFUND?")

    assert result.labels == ("billing",)
    assert result.decided_by == "rules"
    assert result.scores == {"billing": 1.0}
    assert "keyword 'REFUND' matched" in result.reason
    assert result.attempts == 0 and result.usage is None
    assert llm.calls == []


def test_keywords_match_whole_words_by_default():
    rule = KeywordRule("billing", ["refund"])
    loose = KeywordRule("billing", ["refund"], whole_word=False)
    clf = Classifier(LABELS, llm=ScriptedLLM(answer(("billing", 0.9))), rules=[rule])

    assert clf.classify("I was refunded").decided_by == "classify"
    ctx = ClassificationContext(labels=clf.labels, text="I was refunded", llm=clf.llm)
    assert loose(ctx) == "billing"


def test_keyword_case_sensitivity_is_opt_in():
    ctx_text = "API down"
    rule = KeywordRule("technical", ["api"], case_sensitive=True)
    clf = Classifier(LABELS, llm=ScriptedLLM())
    ctx = ClassificationContext(labels=clf.labels, text=ctx_text, llm=clf.llm)
    assert rule(ctx) is None


def test_a_keyword_rule_needs_keywords():
    with pytest.raises(ConfigError):
        KeywordRule("billing", [])


def test_regex_and_callable_rules():
    def account_rule(ctx: ClassificationContext) -> str | None:
        return "account" if "password" in ctx.text else None

    llm = ScriptedLLM()
    clf = Classifier(
        LABELS,
        llm=llm,
        rules=[RegexRule("technical", r"error \d{3}", flags=re.IGNORECASE), account_rule],
    )
    assert clf.classify("ERROR 503 again").labels == ("technical",)
    result = clf.classify("forgot my password")
    assert result.labels == ("account",)
    assert "rule account_rule matched" in result.reason


def test_conflicting_rules_leave_single_label_decisions_to_the_model():
    llm = ScriptedLLM(answer(("account", 0.8)))
    clf = Classifier(
        LABELS,
        llm=llm,
        rules=[KeywordRule("billing", ["charge"]), KeywordRule("account", ["login"])],
    )
    result = clf.classify("charge on login")

    assert result.decided_by == "classify"
    assert result.labels == ("account",)
    assert result.metadata["rule_conflict"] == ["account", "billing"]


def test_agreeing_rules_decide_together():
    clf = Classifier(
        LABELS,
        llm=ScriptedLLM(),
        rules=[KeywordRule("billing", ["charge"]), KeywordRule("billing", ["invoice"])],
    )
    result = clf.classify("invoice charge")
    assert result.labels == ("billing",)
    assert result.reason.count("matched") == 2


def test_multi_label_rules_decide_the_union():
    clf = Classifier(
        LABELS,
        llm=ScriptedLLM(),
        multi_label=True,
        rules=[KeywordRule("billing", ["charge"]), KeywordRule("account", ["login"])],
    )
    assert set(clf.classify("charge on login").labels) == {"billing", "account"}


def test_a_rule_for_an_unknown_label_fails_before_any_text_is_classified():
    clf = Classifier(LABELS, llm=ScriptedLLM(), rules=[KeywordRule("refunds", ["x"])])
    with pytest.raises(ConfigError, match="'refunds'"):
        clf.classify("nothing matches here")


def test_a_callable_rule_returning_an_unknown_label_fails_loudly():
    clf = Classifier(LABELS, llm=ScriptedLLM(), rules=[lambda ctx: "refunds"])
    with pytest.raises(CustomComponentError, match="'refunds'"):
        clf.classify("anything")


def test_rule_step_rejects_non_callables():
    with pytest.raises(CustomComponentError):
        RuleStep("billing")  # type: ignore[arg-type]


# --- validation ---------------------------------------------------------------------------


def test_a_validator_can_veto_a_decision():
    def needs_order_number(labels, ctx):
        if "billing" in labels and not re.search(r"#\d+", ctx.text):
            return "billing needs an order number"
        return None

    clf = Classifier(
        LABELS,
        llm=ScriptedLLM(answer(("billing", 0.9)), answer(("billing", 0.9))),
        steps=[ValidationStep(needs_order_number)],
    )
    rejected = clf.classify("refund please")
    assert rejected.undecided is UndecidedReason.REJECTED
    assert rejected.decided_by == "validate"
    assert rejected.reason == "billing needs an order number"
    assert rejected.labels == ()
    assert rejected.scores == {"billing": 0.9}

    assert clf.classify("refund for #123").labels == ("billing",)


def test_validators_also_check_rule_decisions():
    clf = Classifier(
        LABELS,
        llm=ScriptedLLM(),
        rules=[KeywordRule("billing", ["refund"])],
        steps=[ValidationStep(lambda labels, ctx: "never")],
    )
    assert clf.classify("refund").undecided is UndecidedReason.REJECTED


def test_validators_do_not_run_on_an_abstention():
    calls = []
    clf = Classifier(
        LABELS,
        llm=ScriptedLLM({"labels": []}),
        steps=[ValidationStep(lambda labels, ctx: calls.append(labels))],
    )
    clf.classify("hmm")
    assert calls == []


def test_a_validator_returning_the_wrong_type_fails_loudly():
    clf = Classifier(
        LABELS,
        llm=ScriptedLLM(answer(("billing", 0.9))),
        steps=[ValidationStep(lambda labels, ctx: False)],
    )
    with pytest.raises(CustomComponentError):
        clf.classify("refund")


# --- pipeline -----------------------------------------------------------------------------


def test_default_pipeline_shape():
    assert Classifier(LABELS, llm=ScriptedLLM()).pipeline.names == ["classify"]
    with_rules = Classifier(LABELS, llm=ScriptedLLM(), rules=[KeywordRule("billing", ["x"])])
    assert with_rules.pipeline.names == ["rules", "classify"]


def test_classify_cannot_be_removed_or_disabled():
    pipeline = Classifier(LABELS, llm=ScriptedLLM()).pipeline
    with pytest.raises(ConfigError):
        pipeline.remove("classify")
    with pytest.raises(ConfigError):
        pipeline.disable("classify")


def test_disabling_rules_sends_every_text_to_the_model():
    llm = ScriptedLLM(answer(("technical", 0.9)))
    clf = Classifier(LABELS, llm=llm, rules=[KeywordRule("billing", ["refund"])])
    clf.pipeline.disable("rules")

    assert clf.classify("refund").labels == ("technical",)
    assert len(llm.calls) == 1


def test_a_step_can_add_a_prompt_rule_from_anywhere_in_the_pipeline():
    class Tone(Step):
        name = "tone"

        def prepare(self, ctx):
            ctx.instructions.append("Sarcasm is not a complaint.")
            ctx.prompt_blocks.append(("Customer tier", "gold"))

    llm = ScriptedLLM(answer(("billing", 0.9)))
    clf = Classifier(LABELS, llm=llm, steps=[Tone()])
    clf.classify("great, another charge")
    system, user = llm.calls[0]
    assert "- Sarcasm is not a complaint." in system
    assert "Customer tier:\ngold" in user


def test_classify_can_be_replaced():
    class Fixed(Step):
        name = "classify"

        def run(self, ctx):
            ctx.decide(["account"], by=self.name, reason="fixed")

    clf = Classifier(LABELS, llm=ScriptedLLM())
    old = clf.pipeline.replace("classify", Fixed())
    assert isinstance(old, ClassifyStep)
    assert clf.classify("x").labels == ("account",)


def test_a_replacement_that_never_settles_fails_loudly():
    class Nothing(Step):
        name = "classify"

    clf = Classifier(LABELS, llm=ScriptedLLM())
    clf.pipeline.replace("classify", Nothing())
    with pytest.raises(CustomComponentError, match="without a step deciding"):
        clf.classify("x")


def test_a_custom_step_cannot_decide_outside_the_label_set():
    class Bad(Step):
        name = "bad"

        def run(self, ctx):
            ctx.decide(["refunds"], by=self.name)

    clf = Classifier(LABELS, llm=ScriptedLLM())
    clf.pipeline.add(Bad(), before="classify")
    with pytest.raises(CustomComponentError, match="not in the label set"):
        clf.classify("x")


def test_single_label_mode_refuses_two_labels_from_a_custom_step():
    class Two(Step):
        name = "two"

        def run(self, ctx):
            ctx.decide(["billing", "account"], by=self.name)

    clf = Classifier(LABELS, llm=ScriptedLLM())
    clf.pipeline.add(Two(), before="classify")
    with pytest.raises(CustomComponentError, match="single-label"):
        clf.classify("x")


def test_pipeline_composition_errors():
    pipeline = Classifier(LABELS, llm=ScriptedLLM()).pipeline
    with pytest.raises(ConfigError, match="already in the pipeline"):
        pipeline.add(ClassifyStep())
    with pytest.raises(ConfigError, match="same name"):
        pipeline.replace("classify", RuleStep())
    with pytest.raises(ConfigError, match="No step named"):
        pipeline.enable("missing")
    with pytest.raises(ConfigError, match="either 'before' or 'after'"):
        pipeline.add(RuleStep(), before="classify", after="classify")
