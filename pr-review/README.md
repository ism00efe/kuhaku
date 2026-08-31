# pr-review

A repository-agnostic, modular AI pull-request review engine for GitHub.

It implements one adaptive process:

```
GitHub PR
  -> understand the change      (deterministic analysis + small-LLM classifier)
  -> pick relevant review axes  (correctness / method / scope / structure / ...)
  -> pick a depth per axis      (basic / normal / deep)
  -> gather only the context each task needs
  -> choose a model tier for each task
  -> generate structured findings
  -> verify important / uncertain findings (deterministic first, LLM only if needed)
  -> produce the final review
```

The goal is the *minimum useful analysis* for a given change, escalating to
broader context and stronger models only when the change justifies it.

## Design

Nothing in `src/pr_review/` assumes a language, framework, layout, or test tool
-- repository facts are discovered at runtime. Every pluggable part is a small
class in a name-keyed [`Registry`](src/pr_review/registry.py); the
[pipeline](src/pr_review/pipeline.py) iterates registries and never switches on a
name. Adding a component is additive:

| To add a... | Do this | Nothing else changes |
|---|---|---|
| review axis | new module in `axes/`, `@AXES.register("name")` | pipeline, other axes |
| review depth | new module in `depths/` | every existing reviewer |
| model provider | new module in `providers/`, `@PROVIDERS.register("kind")` | review logic |
| provider endpoint | a `[providers.<name>]` block in `.pr-review.toml` | no code at all |
| verifier / reporter / discoverer / context strategy | same pattern in its package | core |

Runtime is **stdlib-only** (HTTP via `urllib`), so it drops into any repo's CI.

## Local use

```bash
pip install -e ".[dev]"

# with real providers
export GROQ_API_KEY=... OPENROUTER_API_KEY=...
pr-review run --repo /path/to/repo --base main --head HEAD

# no keys: structural-only review (deterministic analysis + plan, no findings)
pr-review run --base main --head HEAD
```

`pr-review run` writes markdown (`--out`) and JSON (`--json-out`), prints the
markdown, and exits `2` if a verified blocker is surfaced.

## GitHub use

```yaml
name: PR review
on: pull_request

permissions:
  contents: read
  pull-requests: write        # only needed for the PR comment

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - uses: <owner>/pr-review@v0        # or ./pr-review from within this repo
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
```

Both keys are optional and independent -- whichever is present is used, and the
tier chain routes around the missing one (see below).

Define them as **organization** secrets so every repository inherits them. On
GitHub Free that works for public repositories; private repositories cannot read
organization secrets without a paid plan, so those need a per-repository secret
(`gh secret set OPENROUTER_API_KEY --repo owner/name`).

**Fork pull requests.** GitHub withholds secrets and downgrades `GITHUB_TOKEN`
to read-only for PRs from forks. The action detects this: the review still runs
(structural-only, since no key reached it) and the report is written to the job
summary instead of a comment. Do not switch the caller to `pull_request_target`
to work around it -- that runs fork-authored code with write access and secrets.

## Providers, tiers and free-tier limits

Model choice is per *tier*, not per repository, and each tier carries an ordered
fallback list. The default split follows the shape of the two free tiers, which
are capped in opposite ways:

| | limit | suits |
|---|---|---|
| Groq | 30 req/min, **8K tokens/min** | many small calls |
| OpenRouter (`:free`) | 20 req/min, **1000 calls/day** | few large calls |

A normal-depth prompt is 6-7K tokens, which is a whole minute of Groq's token
budget but one of OpenRouter's thousand daily calls. So:

| tier | provider | model |
|---|---|---|
| `planner`, `basic` | Groq | `openai/gpt-oss-20b` |
| `normal`, `deep` | OpenRouter | `minimax/minimax-m3:free` |
| `verify` | OpenRouter | `z-ai/glm-5.2:free` |
| `deep` fallback | OpenCode Zen | `nemotron-3-ultra-free` |

`verify` leads with a different model than `deep` deliberately: a model asked to
check its own output tends to agree with it. `z-ai/glm-5.2:free` sits behind
`minimax-m3:free` rather than in front because the shared upstream pool serving
it returns 429 often -- that is not this account's daily quota, and the chain
absorbs it either way.

At roughly 10 requests per PR, 1000 calls/day is about 100 PRs a day. OpenRouter
grants that allowance once at least 10 credits have been purchased; below that
it is 50 calls/day.

Three behaviours keep this honest when a free tier gives out:

- **Failover.** A 404 (model retired from a free catalogue) or 401 is permanent
  for that pair and fails over immediately; a 429 is waited out first. Both are
  recorded, and the report names the model that actually answered.
- **Run memory.** A review makes one call per axis plus one per verification.
  Once a candidate has failed, it is benched for the rest of the run -- for good
  if the failure was permanent, until its cooldown expires if it was a throttle
  -- so the same dead model is not rediscovered, at full retry cost, by every
  later task. Benching is waived when a tier has nothing else left, so a
  temporary throttle can never disable a tier outright.
- **Degradation.** When a tier's whole chain is exhausted, it borrows the next
  tier's models rather than dropping the task, and the finding is marked as
  produced by a degraded tier.
- **Structural-only.** With no reachable model the run does not invent findings.
  Discovery, the deterministic change analysis and the heuristic plan still run
  and are reported, under an explicit banner saying the review itself did not.

`.pr-review.toml` overrides all of this per repository. Note that some free
catalogue entries forbid confidential data -- check before enabling one on a
private repository.

## How much of the change gets reviewed

Two separate questions, kept separate:

**What the deterministic layer sees: everything.** It costs no tokens and no
money, and every later decision depends on it -- which files changed, which
symbols moved, whether an interface or a dependency moved, and therefore which
axes run at which depth. `limits.diff_bytes` is a safety net against a
pathological repository, not a budget.

**What a model is sent: whatever that model can take.** The budget is declared
per model in `[models]`, from two independent facts:

| | window | throughput | budget |
|---|---|---|---|
| `groq openai/gpt-oss-20b` | 131K tokens | 8K tokens/min | ~20 KB |
| `z-ai/glm-5.2:free` | 256K tokens | metered by request | ~860 KB |
| `minimax/minimax-m3:free` | 1M tokens | metered by request | ~3.5 MB |

Groq's window is irrelevant: throughput is the wall. OpenRouter's `:free`
models meter *requests*, so a prompt that fills the window costs exactly what a
one-line prompt costs and there is no reason to send less than the work needs.
No single byte number can describe both, which is why depths no longer carry
one -- a `ContextSpec` says what *kinds* of context to gather, and the model
says how much fits.

**A change too large for one request is split into passes**, never truncated.
Each pass carries its own files' patches and is told, by name, which files the
other passes hold. `limits.max_passes_per_axis` caps the spend, since a pass is
a request; when it bites, the report names every file that was not reached:

```
_coverage_ **12/40** changed files · 4 passes per axis

> ⚠️ 28 changed file(s) were not reviewed — the pass budget ran out: …
```

An earlier version applied one 16 KB cap in the source layer, above the
analyser. On a 51-file pull request it reviewed 8 files, planned the review
from that 4%, and reported "changed files: 8" as though that were the change.

## What keeps a finding honest

A review tool is only as useful as the worst thing it says confidently. Three
rules constrain what a model is allowed to assert:

- **Severity has a per-axis ceiling.** `correctness` and `structure` describe
  defects and may emit `blocker`. `scope` and `method` judge process -- does the
  diff match what the PR claims, is the approach the right one -- and cap at
  `warning`. Being wrong about intent must never fail a build through
  `fail-on-blocker`. The ceiling is stated in the prompt *and* enforced on parse.
- **Unanchored evidence loses confidence.** The deterministic pass checks that
  the text a finding quotes actually appears in the file it cites. When it does
  not, the model paraphrased instead of quoting: confidence is capped into the
  uncertain band and the LLM verifier is told so explicitly, rather than being
  asked to judge blind.
- **A blocker has to be earned.** It is the only severity with automation
  attached, so a finding keeps the label only if verification confirmed it
  (verdict VALID) or its confidence sits outside the uncertain band. Otherwise
  it is reported as a warning, marked `reported as BLOCKER`, with the reason.
  No new threshold: both bars already existed, they simply were not applied to
  severity. Combined with the rule above, an unquotable claim is capped below
  the band and therefore can never block a build.
- **Whole-PR claims cite no file.** `scope` and `method` findings use
  `file: "-"`, and are verified against the complete changed-file list and the
  PR text. Slicing an arbitrary file and asking a model "is this true?" invites
  a yes.

- **The review root is the git top level.** `git diff` emits paths relative to
  it, and every later stage joins those paths onto the root -- to read a changed
  file into the context, to check a cited file exists, to slice source for the
  verifier. So `--repo` is lifted to `git rev-parse --show-toplevel`; pointing it
  at a subdirectory used to make all of it miss silently, leaving the review to
  run on the diff alone with nothing to anchor to. When the root moves, the
  report says so.

An axis added later needs none of this: `max_severity` and `whole_pr` are read
with `getattr`, so omitting them keeps today's behaviour.

## Configuration

Optional `.pr-review.toml` at the repo root overrides defaults (providers, model
tiers and their fallbacks, the degradation ladder, enabled axes/depths, limits,
verification thresholds). See the annotated [`.pr-review.toml`](.pr-review.toml)
in this directory. Env vars (`PR_REVIEW_PROVIDER`, `PR_REVIEW_AXES`,
`PR_REVIEW_DIFF_LIMIT`, `PR_REVIEW_TIER_DEEP_MODEL`, ...) and CLI flags take
precedence.

## Status

Prototype. The architectural seams are real; several components are deliberately
shallow (regex symbol extraction, substring "usage" search, heuristic discovery)
and are single modules behind a protocol, replaceable without touching the
pipeline.
