"""How extraction sits in kuhaku: tracing, the public surface, the dependency rule."""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import kuhaku
from kuhaku.core.observability import bind_trace_id
from kuhaku.tools.extraction import ExtractionRouter, Extractor

from .conftest import Order, ScriptedLLM

SRC = Path(kuhaku.__file__).parent
GOOD = {"data": {"product": "tea", "quantity": 5}}


def _imports(package: Path) -> set[str]:
    found: set[str] = set()
    for path in package.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                found.add(node.module)
            elif isinstance(node, ast.Import):
                found.update(alias.name for alias in node.names)
    return found


def test_extraction_depends_on_core_but_never_on_rag():
    modules = _imports(SRC / "tools" / "extraction")
    assert any(module.startswith("kuhaku.core") for module in modules)
    assert not any(module.startswith("kuhaku.tools.rag") for module in modules)


def test_core_never_depends_on_extraction():
    assert not any(
        module.startswith("kuhaku.tools.extraction") for module in _imports(SRC / "core")
    )


def test_entry_points_are_exported_from_kuhaku():
    for name in (
        "Extractor",
        "ExtractionRouter",
        "ExtractionSchema",
        "ExtractionResult",
        "ExtractionStatus",
    ):
        assert name in kuhaku.__all__
    assert kuhaku.Extractor is Extractor


def test_every_result_carries_the_trace_id_of_its_logs(caplog):
    with caplog.at_level(logging.INFO, logger="kuhaku.tools.extraction"):
        result = Extractor(Order, llm=ScriptedLLM(GOOD)).extract("5 tea", trace_id="t-123")

    assert result.trace_id == "t-123"
    steps = {getattr(record, "step", None) for record in caplog.records}
    assert {"extraction", "extraction.extract", "extraction.continuation"} <= steps
    summary = next(r for r in caplog.records if getattr(r, "step", None) == "extraction")
    assert summary.outcome == "complete"
    assert summary.schema == "Order"


def test_an_outer_trace_id_is_inherited():
    with bind_trace_id("outer-id"):
        result = Extractor(Order, llm=ScriptedLLM(GOOD)).extract("5 tea")
    assert result.trace_id == "outer-id"


def test_routing_and_extraction_share_one_trace_id(caplog):
    llm = ScriptedLLM({"schema": "Order"}, GOOD)
    router = ExtractionRouter([Extractor(Order, llm=llm)])

    with caplog.at_level(logging.INFO, logger="kuhaku.tools.extraction"):
        result = router.extract("5 tea", trace_id="shared")

    assert result.trace_id == "shared"
    route = next(r for r in caplog.records if getattr(r, "step", None) == "extraction.route")
    assert route.schema == "Order"
