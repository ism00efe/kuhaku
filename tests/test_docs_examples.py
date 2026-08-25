"""Executes every fenced ```python block in docs/*.md so a documentation example
cannot silently drift from the code it claims to demonstrate.

A block is skipped only when the line immediately above its opening fence is the
literal HTML comment ``<!-- no-exec -->`` -- there is no other skip mechanism. Every
remaining block is executed in a fresh namespace, entirely offline, against the same
in-memory fakes (``FakeEmbeddings``/``FakeVectorStore``/``FakeLLM``) tests/test_rag_facade.py
wires at the same three ``kuhaku`` module-level globals ``kuhaku.RAG`` itself calls.
One ``FakeVectorStore`` is shared for the duration of a single block, so an ``ingest()``
earlier in a block is visible to an ``ask()`` later in the same block; each block gets
its own store, so blocks never see each other's data.
"""

from __future__ import annotations

import re
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

import kuhaku
from tests.conftest import FakeCache, FakeEmbeddings, FakeLLM, FakeVectorStore

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"

_FENCE_RE = re.compile(r"^```python[ \t]*\r?\n(?P<body>.*?)^```[ \t]*\r?$", re.DOTALL | re.MULTILINE)
_NO_EXEC_MARKER = "<!-- no-exec -->"


@dataclass(frozen=True)
class DocBlock:
    """One ```python fenced block discovered in a docs/*.md file."""

    path: Path  # relative to the repo root
    line: int  # 1-based line number of the opening ``` fence
    source: str
    skip: bool


def _discover_blocks() -> list[DocBlock]:
    blocks: list[DocBlock] = []
    for md_path in sorted(DOCS_DIR.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        rel_path = md_path.relative_to(REPO_ROOT)
        for match in _FENCE_RE.finditer(text):
            fence_line = text.count("\n", 0, match.start()) + 1
            preceding = lines[fence_line - 2].strip() if fence_line >= 2 else ""
            blocks.append(
                DocBlock(
                    path=rel_path,
                    line=fence_line,
                    source=match.group("body"),
                    skip=preceding == _NO_EXEC_MARKER,
                )
            )
    return blocks


_ALL_BLOCKS = _discover_blocks()


def _block_id(block: DocBlock) -> str:
    return f"{block.path.name}:{block.line}"


@pytest.fixture
def docs_offline_environment(monkeypatch, tmp_path):
    """Wires kuhaku.RAG() to run entirely offline, the same way tests/test_rag_facade.py
    does, plus two extras those tests don't need because they always pass explicit
    settings/cache kwargs -- docs examples call bare ``RAG()``, so this fixture has to
    make the *defaults* offline-safe instead:

    - the query-answer cache is forced to an in-memory fake (``RAG()``'s default is a
      real SQLite cache at ``./data/kuhaku_qa_cache.sqlite3``);
    - the audit log is forced off via ``KUHAKU_AUDIT_ENABLED`` (``RAG()``'s default
      writes to ``./logs/kuhaku_audit.jsonl``);
    - ``RAG()``'s own ``tempfile.mkdtemp()`` call (for an unset Chroma persist dir) is
      redirected under pytest's ``tmp_path`` rather than the real OS temp directory --
      moot for what actually gets written, since ``ChromaVectorStore`` is faked below
      and never touches that directory, but keeps every side effect under one
      pytest-managed root as instructed.

    A single ``FakeVectorStore`` is shared by every ``ChromaVectorStore(...)`` call for
    the lifetime of this fixture (one per test => one per doc block), so a block that
    ingests and then asks sees its own data.
    """

    shared_store = FakeVectorStore()

    monkeypatch.setattr(kuhaku, "build_embedding_provider", lambda rs: FakeEmbeddings())
    monkeypatch.setattr(kuhaku, "ChromaVectorStore", lambda *a, **k: shared_store)
    monkeypatch.setattr(kuhaku, "build_llm_provider", lambda s: FakeLLM())
    monkeypatch.setattr(kuhaku, "QueryAnswerCache", lambda *a, **k: FakeCache())

    tmp_dir_counter = iter(range(1_000_000))

    def _fake_mkdtemp(prefix: str = "", **_kwargs: object) -> str:
        d = tmp_path / f"{prefix}{next(tmp_dir_counter)}"
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    monkeypatch.setattr(kuhaku, "tempfile", types.SimpleNamespace(mkdtemp=_fake_mkdtemp))

    monkeypatch.setenv("KUHAKU_AUDIT_ENABLED", "false")
    kuhaku.get_settings.cache_clear()
    yield
    kuhaku.get_settings.cache_clear()


@pytest.mark.parametrize("block", _ALL_BLOCKS, ids=_block_id)
def test_doc_example(block: DocBlock, docs_offline_environment: None) -> None:
    if block.skip:
        pytest.skip(f"{block.path}:{block.line} marked <!-- no-exec -->")

    namespace: dict[str, object] = {}
    code = compile(block.source, f"{block.path}:{block.line}", "exec")
    try:
        exec(code, namespace)  # noqa: S102 -- the one job of this harness is running doc examples
    except Exception as exc:
        raise AssertionError(
            f"docs example failed -- {block.path}:{block.line}\n{type(exc).__name__}: {exc}"
        ) from exc
