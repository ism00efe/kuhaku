# AGENTS.md

Instructions for coding agents working in this repository. Humans: see
[CONTRIBUTING.md](CONTRIBUTING.md), which this file does not replace.

## What this project is

**kuhaku** is an AI orchestration framework for Python. `kuhaku.core` is tool-agnostic
runtime infrastructure — LLM abstraction, configuration, identity, security,
observability, retry. Tools are built on top of it: retrieval-augmented generation in
`src/kuhaku/tools/rag/` and structured extraction in `src/kuhaku/tools/extraction/`.

Neither tool is the framework. Do not restructure the package as if one were, and do not
add tool-shaped fields to core.

Version 0.2.0, Apache-2.0, Python 3.11+.

## The one hard rule

**`kuhaku.core` may never depend on a tool.**

A change that makes `core` import from `tools/`, or that adds a tool-specific field to a
core model, is wrong regardless of how convenient it is. A new tool means a new
`tools/<name>/` package with its own settings object reading its own `KUHAKU_<NAME>__`
variables, not a new branch inside core. `tests/test_core_boundaries.py` enforces it.

A tool reaches core through `with_runtime` (model calls), `traced_call` (a call's
observability) and `InputPolicy` (input checks) rather than wiring the pieces itself, so
what core adds later reaches every tool.

Everything else is negotiable. This is not.

## Running things

```bash
pip install -e ".[dev]"
pytest                      # ~1300 tests, entirely offline
ruff check .
mypy src
```

The suite uses in-memory fakes for the embedding provider, vector store and LLM
(`tests/conftest.py`). It needs no network, no model download and no running LLM server.
**A test of yours that needs any of those is testing the wrong thing.**

Never run `pip install` for the heavy optional path (torch, chromadb models) just to make
a test pass — use the fakes.

## Rules that came out of real mistakes

**Do not add a setting without a reader.** A field on `Settings`, `RAGSettings` or
`ExtractionSettings` that
changes nothing is worse than a missing one, because it reads as a working control. If
you add one, wire it in the same change and add a test that proves it has an effect. This
repository already carries eighteen such fields from earlier work; do not add a
nineteenth.

**Security that costs nothing is always on; security that costs something is optional.**
A protection with no real cost to performance or usability runs unconditionally and has
no switch. A protection that slows things down, needs a download or extra
infrastructure, or gets in the way of a quick experiment is optional — a throwaway demo
must not be forced into it. Never add a bypass just because it makes a change easier to
write.

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
   package but a caller cannot reach it through `RAG()`, `Extractor` or another exported
   entry point, say so explicitly rather than describing it as if it worked.

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
  `"auto"`, resolved at build time by `core/resolve/` (the registry-driven capability
  resolver). Choosing the LLM checks only credentials and the Ollama endpoint, never the
  GPU (that probe imports torch), and continues with the provider used last. `"auto"` only ever selects something usable now, never triggers a download
  on its own, and never overrides an explicit value; consent for an install or a
  download is asked separately and never inferred. Adding a new external-dependency
  backend means a new adapter under `core/resolve/adapters/` or
  `tools/rag/resolve/adapters/`, not a branch in the resolver. The resolver announces
  through the `kuhaku` logger (attaching a stderr handler if the host configured none)
  and remembers decisions in `.kuhaku/decisions.json` (project-scoped; recommend adding
  `.kuhaku/` to `.gitignore`; an unwritable directory just means decisions are remade
  each run).
- The layered system prompt in `tools/rag/prompts/`. The safety core — instruction
  precedence, data marking, the canary rule, grounding, mandatory citations,
  contradiction handling — is framework-owned and not up for trimming.

## Known gaps

These are known and deliberate for 0.1.0. Do not "fix" them as a side effect of another
change, and do not document them as working:

- The authorization and authentication classes in `core/auth/` are not wired into
  retrieval. Working access control is the `access_tags` intersection in
  `tools/rag/retriever.py`. The serving layer (`kuhaku.serving`) is their one caller:
  it authenticates requests with the providers and checks an `AuthorizationPolicy`
  before calling a tool.
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
- The built-in lightweight (Tier-0) vector store and the Qdrant adapter are not
  implemented. Chroma is the only store candidate the resolver has, so the store
  decision always takes the "exactly one" branch, and `suggest_store_upgrade` has
  nothing to offer. `guard_single_writer` wraps `RAG.ingest` (a second concurrent writer
  to the same store dir gets `StoreConflict`), but a caller composing `RAGEngine`
  directly is not covered, and there is still no built-in store whose own write path
  invokes it.

## Reporting back

State what you changed, how you verified it, and what you deliberately left out of scope.
If you could not verify something, say that rather than implying you did. For anything
touching retrieval, access filtering or the audit trail, describe the specific check you
ran — not just that you ran the suite.
