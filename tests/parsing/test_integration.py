"""The parsing tool's output feeding the other tools -- through text only, never imports."""

from __future__ import annotations

import ast
from pathlib import Path

from kuhaku.tools.extraction import Extractor
from kuhaku.tools.parsing import DocumentParser
from kuhaku.tools.rag.chunking import StructuralChunker
from kuhaku.tools.rag.models import Document

from ..extraction.conftest import Order, ScriptedLLM

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "kuhaku" / "tools" / "parsing"

HTML = b"""<h1>Price list</h1><p>Valid from March.</p>
<table><tr><th>Product</th><th>Price</th></tr><tr><td>Tea</td><td>4</td></tr></table>"""


def _imports(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def test_parsing_depends_on_core_only() -> None:
    for path in PACKAGE.rglob("*.py"):
        for module in _imports(path):
            if module.startswith("kuhaku."):
                assert module.startswith(("kuhaku.core", "kuhaku.tools.parsing")), (path, module)


def test_markdown_rendering_feeds_the_structural_chunker() -> None:
    doc = DocumentParser(ocr=None).parse(HTML, filename="prices.html")
    rag_doc = Document(
        id="prices", title=doc.title or "", doc_type="document",
        text=doc.to_markdown(), source_path=doc.source,
    )
    chunks = StructuralChunker().chunk(rag_doc, chunk_size=800, overlap=0)
    tables = [c for c in chunks if c.content_type == "table"]
    assert len(tables) == 1 and "Tea" in tables[0].text


def test_markdown_rendering_feeds_the_extractor() -> None:
    doc = DocumentParser(ocr=None).parse(HTML, filename="prices.html")
    llm = ScriptedLLM({"data": {"product": "Tea", "quantity": 4}})
    result = Extractor(Order, llm=llm, enable_token_tracking=False).extract(doc.to_markdown())
    assert result.ok
    assert "| Tea | 4 |" in llm.calls[0][1]
