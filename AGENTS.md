# AGENTS.md

Instructions for coding agents working in this repository. Humans: see
[CONTRIBUTING.md](CONTRIBUTING.md), which this file does not replace.

## What this project is

**kuhaku** is an AI orchestration framework for Python. `kuhaku.core` is tool-agnostic
runtime infrastructure — LLM abstraction, configuration, identity, security,
observability, retry. Tools are built on top of it. Retrieval-augmented generation is the
first tool, and it lives in `src/kuhaku/tools/rag/`.

RAG is not the framework. Do not restructure the package as if it were, and do not add
RAG-shaped fields to core.

Version 0.1.0, Apache-2.0, Python 3.11+. Not yet published to PyPI.

## The one hard rule

**`kuhaku.core` may never depend on a tool.**

A change that makes `core` import from `tools/`, or that adds a RAG-specific field to a
core model, is wrong regardless of how convenient it is. A second tool means a new
`tools/<name>/` package with its own settings dataclass, not a new branch inside core.

Everything else is negotiable. This is not.

## Running things

```bash
pip install -e ".[dev]"
pytest                      # ~1100 tests, entirely offline
ruff check .
mypy src
```

The suite uses in-memory fakes for the embedding provider, vector store and LLM
(`tests/conftest.py`). It needs no network, no model download and no running LLM server.
**A test of yours that needs any of those is testing the wrong thing.**

Never run `pip install` for the heavy optional path (torch, chromadb models) just to make
a test pass — use the fakes.

## Rules that came out of real mistakes

**Do not add a setting without a reader.** A field on `Settings` or `RAGSettings` that
changes nothing is worse than a missing one, because it reads as a working control. If
you add one, wire it in the same change and add a test that proves it has an effect. This
repository already carries eighteen such fields from earlier work; do not add a
nineteenth.

**Security-relevant code has no off switch.** Access filtering, PII sanitization and the
input guard have no parameter that disables them. If a change would be easier with a
bypass, change the approach instead.

**Never edit a test expectation to make the suite green.** When an assertion starts
failing, first establish whether the code or the expectation is wrong. Say which, in your
summary.

**LF line endings only.** `.gitattributes` sets `* text=auto eol=lf`. If `git diff` shows
whole files as changed, CRLF crept in — fix it rather than committing it.

**Leave nothing behind in the repository.** Build a verification virtualenv outside the
working tree and remove it. Do not commit `.venv`, `dist/`, caches or scratch files.

**Comments explain why, not what.** The code already says what it does. A comment earns
its place by recording a trade-off, a failure mode that was hit, or a constraint imposed
by a dependency.

## Documentation

`docs/` is plain Markdown and is the user-facing reference. Two rules:

1. **Every code block must be complete and runnable** — imports included, no reliance on
   a variable defined elsewhere on the page or on another page.
2. **Document only what works through the public API.** If a capability exists in the
   package but a `RAG()` caller cannot reach it, say so explicitly rather than describing
   it as if it worked.

The `README.md` is also the PyPI project page, so links to other files in the repository
must be absolute GitHub URLs, not relative paths.

## What not to touch without being asked

- `.gitattributes`, and anything to do with line endings
- Git history — it was rewritten to use a single noreply identity and must stay that way.
  Never amend, rebase or force-push.
- The default values in `core/config.py` and `tools/rag/config.py`. They encode a
  deliberate policy: **a default may cost CPU and memory, never a download.** Hybrid
  retrieval is on because it costs neither; the cross-encoder re-ranker is off because it
  is roughly a gigabyte. `retrieval`, `llm_provider` and `embedding_device` default to
  `"auto"`, resolved at build time by `core/capabilities.py` +
  `tools/rag/capabilities.py` — auto only ever downgrades toward fewer dependencies,
  never triggers a download, and never overrides an explicit value. Adding a new
  external-dependency setting means giving it an `"auto"` chain the same way.
- The layered system prompt in `tools/rag/prompts/`. The safety core — instruction
  precedence, data marking, the canary rule, grounding, mandatory citations,
  contradiction handling — is framework-owned and not up for trimming.

## Known gaps

These are known and deliberate for 0.1.0. Do not "fix" them as a side effect of another
change, and do not document them as working:

- The authorization and authentication classes in `core/auth/` are not wired into
  retrieval. Working access control is the `access_tags` intersection in
  `tools/rag/retriever.py`.
- The second-generation guard and the entire output guard are not reachable through the
  `RAG` facade; the engine only invokes them when a guard pipeline is supplied (via
  `RAGEngine(guard=...)` or `rag.engine.update_guard(...)`). `RAG(settings=Settings(
  guard_enabled=True))` does not build one for you, and now raises `SecurityComponentError`
  at construction rather than silently ignoring the setting (see `core.policy.enforce_guard_policy`).
- `core/policy.py` is tested but has no call sites.
- Query rewriting and contradiction detection are constructible but not wired to the
  facade.
- References to `service.build_service()` in comments point at a module that does not
  exist in this package. They are leftovers.

## Reporting back

State what you changed, how you verified it, and what you deliberately left out of scope.
If you could not verify something, say that rather than implying you did. For anything
touching retrieval, access filtering or the audit trail, describe the specific check you
ran — not just that you ran the suite.
