# Contributing to kuhaku

Thanks for looking. kuhaku is early — version 0.2.0 — so the most useful contributions
right now are bug reports with a reproduction, and questions about anything the
documentation gets wrong.

## Getting set up

```bash
git clone https://github.com/ism00efe/kuhaku
cd kuhaku
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

The suite runs entirely offline against in-memory fakes for the embedding provider, vector
store and LLM. No network, no model download, no running LLM server. If a test of yours
needs any of those, it is testing the wrong thing.

## The one hard rule

**`kuhaku.core` may never depend on a tool.**

`core` is tool-agnostic runtime infrastructure. `tools/rag` and `tools/extraction` are
tools built on top of it. A change that makes `core` import from `tools/`, or that adds a
tool-shaped field to a core model, will be asked to move. Adding a tool means a new
`tools/<name>` package with its own settings — not a new branch inside core.

Everything else is negotiable. This is not.

## What good looks like here

**Comments explain why, not what.** The code says what it does. A comment earns its place
by recording the reasoning that is not visible in the code — a trade-off, a failure mode
that was hit, a constraint from a dependency.

**Do not add a setting without a reader.** A configuration field that changes nothing is
worse than a missing one, because it reads as a working control. If you add a field to
`Settings`, `RAGSettings` or `ExtractionSettings`, wire it in the same change.

**Security that costs nothing is always on; security that costs something is optional.**
A protection with no real cost to performance or usability runs unconditionally. One that
slows things down, needs a download, or gets in the way of a quick experiment is
optional. Never add a bypass just because it makes a change easier to write.

**Defaults may cost CPU and memory, never a download.** A bare `RAG()` must not pull a
model onto someone's machine, and `pip install kuhaku` must not pull anything heavy — a
feature's dependencies go in its extra. That is why the re-ranker is off and RAG's
libraries live in `kuhaku[rag]`.

## Tests

New behaviour needs a test. Changed behaviour needs its test updated *deliberately* — if
an existing assertion now fails, work out whether the code or the expectation is wrong
before touching either. A test expectation edited to make a suite green is how a
regression ships.

For anything touching retrieval, access filtering or the audit trail, say in the pull
request how you verified it, not just that you did.

## Style

- Python 3.11+, type hints throughout
- `ruff` and `mypy` are configured in the dev extra; run them before opening a PR
- LF line endings — `.gitattributes` enforces this. A diff that shows whole files as
  changed means CRLF crept in
- Match the surrounding code rather than the style you prefer

## Pull requests

Small and focused beats large and complete. A PR that does one thing and explains why is
easier to accept than one that does five.

Include:

- what changed and why
- how you verified it
- anything you deliberately left out of scope

## Reporting a security issue

See [SECURITY.md](SECURITY.md).

## License

By contributing you agree that your contributions are licensed under the Apache License
2.0, the same as the project.
