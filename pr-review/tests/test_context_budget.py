"""How much of the change reaches the models, and who decides.

The first real pull request this engine reviewed had 51 changed files and
3,803 changed lines. It reported on 8 files and 159 lines, and said nothing
about the rest -- a global 16 KB diff cap, sized for the throughput limit of
the cheapest provider, was applied in the source layer above the deterministic
analyser. Every later decision, including which axes ran at which depth, was
therefore made from 4% of the change.

These tests pin the three properties that failure violated:

* the deterministic analysis, which is free, always sees the whole change;
* how much a model is sent comes from that model's own limits;
* a change too large for one request is split into passes, never silently cut.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from pr_review.config import Config, Limits, load_config
from pr_review.context.default import DefaultContextGatherer
from pr_review.models import (
    ContextSpec,
    DeterministicAnalysis,
    FileChange,
    PlanEntry,
    PRContext,
    RepoMetadata,
    ReviewPlan,
    ReviewResult,
    ReviewTask,
)
from pr_review.plan.planner import Planner
from pr_review.report.base import REPORTERS
from pr_review.source.local_git import LocalGitSource

# --------------------------------------------------------------------------- #
# 1. The free layer is never starved
# --------------------------------------------------------------------------- #


def test_a_large_change_reaches_the_analyser_whole(tiny_repo: Path):
    """A change far bigger than the old 16 KB cap must arrive intact."""
    for i in range(40):
        body = "\n".join(f"def gen_{i}_{j}(a, b):\n    return a + b + {j}" for j in range(30))
        (tiny_repo / f"mod_{i:02d}.py").write_text(body + "\n")
    subprocess.run(["git", "add", "-A"], cwd=tiny_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "feat: many modules"],
        cwd=tiny_repo, check=True, capture_output=True,
    )

    pr, _root = LocalGitSource(tiny_repo, base_ref="main", head_ref="HEAD").load()

    assert len(pr.diff.encode()) > 40_000  # several times the old 16 KB cap
    assert pr.extra["diff_truncated"] is False
    assert len(pr.changed_paths) >= 40


def test_the_safety_cap_is_a_safety_cap():
    assert Limits().diff_bytes >= 1_000_000


# --------------------------------------------------------------------------- #
# 2. The budget belongs to the model
# --------------------------------------------------------------------------- #


def test_budget_follows_the_model_not_a_global_number(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    cfg = load_config(tmp_path)

    groq = cfg.resolve_chain("basic")[0]
    big = cfg.resolve_chain("deep")[0]

    assert groq.provider == "groq"
    assert big.provider == "openrouter"
    # Groq is throughput-capped at 8K tokens/min: the window is irrelevant.
    assert groq.input_budget_bytes < 40_000
    # MiniMax M3 is metered by request, so the whole million-token window counts.
    assert big.input_budget_bytes > 1_000_000


def test_an_unlisted_model_gets_a_conservative_budget():
    cfg = Config()
    unknown = cfg.input_budget_bytes("some/model-nobody-declared", 1000)
    known = cfg.input_budget_bytes("minimax/minimax-m3:free", 1000)
    assert unknown < known
    assert unknown > 0


def test_throughput_cap_beats_window_size():
    """A 131K window is worthless at 8K tokens a minute."""
    cfg = Config()
    assert cfg.input_budget_bytes("openai/gpt-oss-120b", 1200) < cfg.input_budget_bytes(
        "z-ai/glm-5.2:free", 1200
    )


# --------------------------------------------------------------------------- #
# 3. Too big to send is split, not cut
# --------------------------------------------------------------------------- #


def _analysis(sizes: dict[str, int]) -> DeterministicAnalysis:
    a = DeterministicAnalysis()
    a.files = [
        FileChange(path=p, status="modified", additions=1, deletions=0, diff_bytes=n)
        for p, n in sizes.items()
    ]
    return a


def _plan() -> ReviewPlan:
    return ReviewPlan(entries=[PlanEntry(axis="correctness", depth="basic")])


def _cfg(budget_model: str = "mock", passes: int = 4) -> Config:
    cfg = Config(fallback_provider="", limits=Limits(max_passes_per_axis=passes))
    return cfg


def test_a_change_that_fits_is_one_pass(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    cfg = load_config("/nonexistent")
    tasks = Planner(cfg).build(_plan(), _analysis({"a.py": 500, "b.py": 500}))
    assert len(tasks) == 1
    assert tasks[0].files == ()          # the whole change
    assert tasks[0].pass_count == 1


def test_a_change_that_does_not_fit_is_split_and_fully_covered(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    cfg = load_config("/nonexistent")
    budget = cfg.resolve_chain("basic")[0].input_budget_bytes
    sizes = {f"f{i:02d}.py": budget // 3 for i in range(9)}  # ~3x the budget

    tasks = Planner(cfg).build(_plan(), _analysis(sizes))

    assert len(tasks) > 1
    assert all(t.pass_count == len(tasks) for t in tasks)
    covered = [p for t in tasks for p in t.files]
    assert sorted(covered) == sorted(sizes)      # every file, exactly once
    assert len(covered) == len(set(covered))
    for t in tasks:
        assert sum(sizes[p] for p in t.files) <= budget or len(t.files) == 1


def test_the_pass_cap_is_a_spend_dial_and_is_reported(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    cfg = load_config("/nonexistent")
    cfg = Config(**{**cfg.__dict__, "limits": Limits(max_passes_per_axis=2)})
    budget = cfg.resolve_chain("basic")[0].input_budget_bytes
    sizes = {f"f{i:02d}.py": budget for i in range(6)}

    analysis = _analysis(sizes)
    tasks = Planner(cfg).build(_plan(), analysis)
    assert len(tasks) == 2

    result = ReviewResult(
        pr=PRContext("t", "", "main", "f", "0", "1", ""),
        repo=RepoMetadata(), analysis=analysis, plan=_plan(), tasks=tasks,
        stats={"mode": "llm"},
    )
    assert len(result.reviewed_paths()) == 2
    assert len(result.unreviewed_paths()) == 4

    md = REPORTERS.create("markdown").render(result)
    assert "**2/6** changed files" in md
    assert "were not reviewed" in md
    assert "max_passes_per_axis" in md


# --------------------------------------------------------------------------- #
# 4. A pass knows what it is not looking at
# --------------------------------------------------------------------------- #


def test_a_pass_carries_only_its_files_but_names_the_others(
    tiny_repo: Path, sample_diff: str
):
    pr = PRContext("t", "", "main", "feature", "0", "1", sample_diff)
    analysis = _analysis({"src/calc.py": 400, "pyproject.toml": 200})
    task = ReviewTask(
        axis="correctness", depth="basic", model_tier="basic",
        context_strategy="default", context_spec=ContextSpec(),
        files=("src/calc.py",), pass_index=1, pass_count=2,
        input_budget_bytes=60_000,
    )

    ctx = DefaultContextGatherer().gather(
        task, repo_root=tiny_repo, pr=pr, analysis=analysis, meta=RepoMetadata()
    )
    diff_item = next(i for i in ctx.items if i.kind == "diff")
    assert "src/calc.py" in diff_item.content
    assert "pyproject.toml" not in diff_item.content

    note = next(i for i in ctx.items if i.kind == "note")
    assert "pyproject.toml" in note.content
    assert "another pass" in note.label
