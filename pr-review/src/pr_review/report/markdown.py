"""Markdown reporter -- the PR-comment body."""

from __future__ import annotations

from pr_review.models import ReviewResult, Verdict
from pr_review.report.base import REPORTERS

_SEV_ORDER = {"blocker": 0, "warning": 1, "info": 2}
_VERDICT_BADGE = {
    Verdict.VALID: "✅ verified",
    Verdict.UNCERTAIN: "❔ unverified",
    Verdict.INVALID: "🚫 disproven",
}


@REPORTERS.register("markdown")
class MarkdownReporter:
    name = "markdown"
    file_extension = "md"

    def render(self, result: ReviewResult) -> str:
        out: list[str] = ["## PR Review"]
        plan = result.plan
        active = plan.active()
        out.append(
            f"_change type_ **{plan.change_type or 'n/a'}** · "
            f"_plan_ {plan.source} · "
            f"_axes_ {', '.join(f'{e.axis}/{e.depth}' for e in active) or 'none'}"
        )
        if plan.risk_areas:
            out.append(f"_risk areas_: {', '.join(plan.risk_areas)}")

        total = len(result.changed_paths())
        seen = len(result.reviewed_paths())
        passes = result.pass_count()
        if total and not result.structural_only:
            line = f"_coverage_ **{seen}/{total}** changed files"
            if passes > 1:
                line += f" · {passes} passes per axis"
            out.append(line)
        if result.pr.extra.get("diff_truncated"):
            out.append("> ⚠️ the diff was truncated; review may be incomplete")
        missing = [] if result.structural_only else result.unreviewed_paths()
        if missing:
            shown = ", ".join(f"`{p}`" for p in missing[:12])
            more = f" and {len(missing) - 12} more" if len(missing) > 12 else ""
            out.append(
                f"\n> ⚠️ **{len(missing)} changed file(s) were not reviewed** — the "
                f"pass budget (`limits.max_passes_per_axis`) ran out: {shown}{more}"
            )
        if result.pr.extra.get("repo_root_adjusted"):
            out.append(
                "> ℹ️ the review root was moved to the repository top level "
                f"(`{result.pr.extra['repo_root_adjusted']}`), because diff paths "
                "are relative to it"
            )
        if result.structural_only:
            out.append(
                "\n> ℹ️ **Structural analysis only.** No LLM provider is configured "
                "for this repository, so no code-level findings were produced. "
                "Everything below is deterministic: the change analysis and the "
                "review plan that *would* have run. Set a provider API key to "
                "enable the review itself."
            )

        surfaced = [
            vf for vf in result.surfaced()
            if vf.result.verdict != Verdict.UNCERTAIN or vf.finding.severity != "info"
        ] or result.surfaced()
        surfaced.sort(
            key=lambda vf: (
                _SEV_ORDER.get(vf.finding.severity, 3),
                0 if vf.result.verdict == Verdict.VALID else 1,
                -vf.finding.confidence,
            )
        )
        keep = surfaced[: max(1, len(surfaced))]

        out.append(f"\n### Findings ({len(keep)})")
        if not keep:
            out.append(
                "Not run — no model was reachable."
                if result.structural_only
                else "No problems found by the selected review axes."
            )
        for vf in keep:
            f = vf.finding
            loc = f.file + (f":{f.line}" if f.line else "")
            badge = _VERDICT_BADGE.get(vf.result.verdict, "")
            demoted = f.extra.get("severity_capped_from")
            note = f" · _reported as {str(demoted).upper()}_" if demoted else ""
            out.append(
                f"- **[{f.severity.upper()}]** `{loc}` · _{f.axis}_ · {badge} "
                f"· conf {f.confidence:.2f}{note}\n"
                f"  - {f.issue}"
            )
            if demoted and f.extra.get("severity_capped_reason"):
                out.append(f"  - _downgraded_: {f.extra['severity_capped_reason']}")
            if f.reasoning:
                out.append(f"  - _why_: {f.reasoning}")
            if f.evidence:
                out.append(f"  - _evidence_: `{f.evidence[:200]}`")
            if vf.result.evidence:
                out.append(f"  - _verification_: {vf.result.evidence[:240]}")

        disproven = [
            vf for vf in result.findings if vf.result.verdict == Verdict.INVALID
        ]
        if disproven:
            out.append("\n<details><summary>Disproven claims "
                       f"({len(disproven)})</summary>\n")
            for vf in disproven:
                out.append(
                    f"- `{vf.finding.file}` {vf.finding.issue} — "
                    f"{vf.result.evidence or '; '.join(vf.result.checks)}"
                )
            out.append("\n</details>")

        out.append("\n<details><summary>Review plan &amp; stats</summary>\n")
        out.append("```")
        out.append(f"repo language: {result.repo.primary_language() or 'unknown'}")
        out.append(f"changed files: {result.analysis.changed_file_count} "
                   f"(+{result.analysis.total_additions}/-{result.analysis.total_deletions})")
        for e in plan.entries:
            out.append(f"  {e.axis:12} {e.depth:6} {e.reason}")
        out.append(f"files reviewed: {seen}/{total} in {passes} pass(es) per axis")
        for k, v in result.stats.items():
            out.append(f"{k}: {v}")
        if result.notes:
            out.append("provider notes:")
            for n in result.notes:
                out.append(f"  {n}")
        if result.errors:
            out.append("errors:")
            for err in result.errors:
                out.append(f"  [{err.stage}] {err.detail}")
        out.append("```")
        out.append("</details>")
        out.append("\n<sub>Generated by pr-review — repository-agnostic modular PR review.</sub>")
        return "\n".join(out)
