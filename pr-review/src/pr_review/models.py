"""Internal data model shared by every stage of the pipeline.

These are plain dataclasses with no behaviour and no third-party dependency, so
any stage can be replaced without touching the schema. Fields are additive:
prefer adding an optional field over changing an existing one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# --------------------------------------------------------------------------- #
# GitHub / PR input (produced by a PRSource, consumed by the engine)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PRContext:
    """Everything the engine needs about the pull request itself."""

    title: str
    body: str
    base_ref: str
    head_ref: str
    base_sha: str
    head_sha: str
    diff: str
    """Unified diff of base...head (already size-limited by the source)."""
    changed_paths: tuple[str, ...] = ()
    repo_slug: str = ""
    number: int | None = None
    author: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Repository discovery
# --------------------------------------------------------------------------- #


@dataclass
class RepoMetadata:
    """Structured, reusable facts about the repository under review.

    Merged from independent discoverers; unknown values stay empty rather than
    guessed.
    """

    root: str = ""
    languages: dict[str, int] = field(default_factory=dict)
    """language name -> file count, most significant first when iterated sorted."""
    package_managers: list[str] = field(default_factory=list)
    dependency_files: list[str] = field(default_factory=list)
    frameworks: list[str] = field(default_factory=list)
    top_level_dirs: list[str] = field(default_factory=list)
    config_files: list[str] = field(default_factory=list)
    test_paths: list[str] = field(default_factory=list)
    doc_paths: list[str] = field(default_factory=list)
    architecture_docs: list[str] = field(default_factory=list)
    static_analysis_tools: list[str] = field(default_factory=list)
    public_api_hints: list[str] = field(default_factory=list)
    conventions: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def primary_language(self) -> str:
        if not self.languages:
            return ""
        return max(self.languages.items(), key=lambda kv: kv[1])[0]


# --------------------------------------------------------------------------- #
# Deterministic change analysis
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChangedSymbol:
    path: str
    name: str
    kind: str  # "function" | "class" | "method" | "other"
    change: str  # "added" | "removed" | "modified"


@dataclass
class FileChange:
    path: str
    status: str  # "added" | "deleted" | "modified" | "renamed"
    additions: int = 0
    deletions: int = 0
    old_path: str | None = None
    is_dependency_manifest: bool = False
    language: str = ""


@dataclass
class DeterministicAnalysis:
    files: list[FileChange] = field(default_factory=list)
    total_additions: int = 0
    total_deletions: int = 0
    added_paths: list[str] = field(default_factory=list)
    deleted_paths: list[str] = field(default_factory=list)
    changed_symbols: list[ChangedSymbol] = field(default_factory=list)
    added_imports: list[str] = field(default_factory=list)
    removed_imports: list[str] = field(default_factory=list)
    dependency_changes: list[str] = field(default_factory=list)
    touched_areas: list[str] = field(default_factory=list)
    """Top-level-ish repo areas the change touches (e.g. ``src/pkg/http``)."""
    interface_changes: list[str] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)
    """Free-form derived booleans/counts for the classifier heuristic."""

    @property
    def changed_file_count(self) -> int:
        return len(self.files)


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #


class Depth(StrEnum):
    OFF = "off"
    BASIC = "basic"
    NORMAL = "normal"
    DEEP = "deep"


@dataclass(frozen=True)
class PlanEntry:
    axis: str
    depth: str
    reason: str = ""


@dataclass
class ReviewPlan:
    entries: list[PlanEntry] = field(default_factory=list)
    change_type: str = ""
    risk_areas: list[str] = field(default_factory=list)
    source: str = ""  # "llm" | "heuristic" | "config"
    raw: str = ""

    def active(self) -> list[PlanEntry]:
        return [e for e in self.entries if e.depth != Depth.OFF.value]


@dataclass(frozen=True)
class ContextSpec:
    """What a depth wants gathered. The axis narrows *what kind*."""

    changed_file_body: bool = True
    surrounding_lines: int = 0
    sibling_files: bool = False
    callers_usages: bool = False
    dependency_files: bool = False
    architecture_docs: bool = False
    repo_tree: bool = False
    max_files: int = 6
    max_bytes: int = 24_000


@dataclass(frozen=True)
class ReviewTask:
    axis: str
    depth: str
    model_tier: str
    context_strategy: str
    context_spec: ContextSpec


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #


@dataclass
class ContextItem:
    kind: str  # "diff" | "file" | "usage" | "dependency" | "doc" | "tree" | "note"
    label: str
    content: str


@dataclass
class ReviewContext:
    task: ReviewTask
    items: list[ContextItem] = field(default_factory=list)

    def render(self) -> str:
        blocks = []
        for it in self.items:
            blocks.append(f"### {it.kind.upper()}: {it.label}\n{it.content}")
        return "\n\n".join(blocks)

    def digest(self) -> str:
        import hashlib

        h = hashlib.sha256()
        for it in self.items:
            h.update(it.label.encode())
            h.update(it.content.encode())
        return h.hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Findings & verification
# --------------------------------------------------------------------------- #


@dataclass
class Finding:
    axis: str
    severity: str  # "blocker" | "warning" | "info"
    file: str
    line: int | None
    issue: str
    evidence: str = ""
    reasoning: str = ""
    confidence: float = 0.5
    depth: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def key(self) -> tuple[str, str, int, str]:
        return (self.axis, self.file, self.line or -1, self.issue.strip().lower())


class Verdict(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    UNCERTAIN = "uncertain"


@dataclass
class VerdictResult:
    verdict: Verdict
    evidence: str = ""
    method: str = ""  # "deterministic" | "llm" | "skipped"
    checks: list[str] = field(default_factory=list)


@dataclass
class VerifiedFinding:
    finding: Finding
    result: VerdictResult


# --------------------------------------------------------------------------- #
# Final result
# --------------------------------------------------------------------------- #


@dataclass
class StageError:
    stage: str
    detail: str


@dataclass
class ReviewResult:
    pr: PRContext
    repo: RepoMetadata
    analysis: DeterministicAnalysis
    plan: ReviewPlan
    tasks: list[ReviewTask] = field(default_factory=list)
    findings: list[VerifiedFinding] = field(default_factory=list)
    errors: list[StageError] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    """Non-failure run events worth showing: provider failover, degradation."""
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def structural_only(self) -> bool:
        """True when no LLM ran: the report is deterministic analysis alone."""
        return self.stats.get("mode") == "structural-only"

    def surfaced(self) -> list[VerifiedFinding]:
        """Findings worth showing: not deterministically disproven."""
        return [vf for vf in self.findings if vf.result.verdict != Verdict.INVALID]
