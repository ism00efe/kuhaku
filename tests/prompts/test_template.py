"""PromptTemplate: declared variables, content hash, strict rendering."""

from __future__ import annotations

import hashlib

import pytest

from kuhaku.core.exceptions import KuhakuError
from kuhaku.core.prompts import (
    CUSTOM_VERSION,
    PromptRef,
    PromptRenderError,
    PromptTemplate,
    PromptTemplateError,
)


def test_variables_are_inferred_when_not_declared():
    template = PromptTemplate("greet", "1", "Hello {{name}}, {{ place }}.")
    assert template.variables == frozenset({"name", "place"})


def test_declared_variables_must_match_the_placeholders_both_ways():
    with pytest.raises(PromptTemplateError, match="not declared"):
        PromptTemplate("greet", "1", "Hello {{name}} {{typo}}", variables={"name"})
    with pytest.raises(PromptTemplateError, match="no placeholder"):
        PromptTemplate("greet", "1", "Hello {{name}}", variables={"name", "unused"})


def test_content_hash_is_sha256_of_the_template_and_ignores_name_and_version():
    a = PromptTemplate("a", "1", "same text")
    b = PromptTemplate("b", "7", "same text")
    assert a.content_hash == hashlib.sha256(b"same text").hexdigest()
    assert a.content_hash == b.content_hash
    assert PromptTemplate("a", "1", "other").content_hash != a.content_hash


@pytest.mark.parametrize("name", ["", "has space", None, "x\n"])
def test_bad_names_are_rejected(name):
    with pytest.raises(PromptTemplateError):
        PromptTemplate(name, "1", "text")


def test_render_substitutes_and_leaves_single_braces_alone():
    template = PromptTemplate("j", "1", 'Reply {"key": "{{value}}"}')
    assert template.render(value="x") == 'Reply {"key": "x"}'
    assert template.render({"value": "y"}) == 'Reply {"key": "y"}'


def test_render_rejects_missing_extra_and_none_variables():
    template = PromptTemplate("t", "1", "{{a}} {{b}}")
    with pytest.raises(PromptRenderError, match="missing variables \\['b'\\]"):
        template.render(a="1")
    with pytest.raises(PromptRenderError, match="unexpected variables \\['c'\\]"):
        template.render(a="1", b="2", c="3")
    with pytest.raises(PromptRenderError, match="None"):
        template.render(a="1", b=None)


def test_rendered_values_are_never_substituted_again():
    template = PromptTemplate("t", "1", "{{a}} / {{b}}")
    assert template.render(a="{{b}}", b="real") == "{{b}} / real"


def test_render_prompt_carries_the_ref():
    template = PromptTemplate("t", "3", "x {{a}}")
    rendered = template.render_prompt(a="1")
    assert rendered.text == "x 1"
    assert rendered.ref == PromptRef("t", "3", template.content_hash)
    assert rendered.ref.as_dict() == {"name": "t", "version": "3", "hash": template.content_hash}


def test_ref_for_custom_text_hashes_the_text():
    ref = PromptRef.for_text("rag.system", "my own prompt")
    assert ref.version == CUSTOM_VERSION
    assert ref.content_hash == hashlib.sha256(b"my own prompt").hexdigest()


def test_errors_are_kuhaku_errors_and_value_errors():
    with pytest.raises(KuhakuError):
        PromptTemplate("t", "1", "{{a}}").render()
    with pytest.raises(ValueError):
        PromptTemplate("t", "1", "{{a}}").render()
