"""Run one review task: build the prompt, call the model, parse findings.

Owns JSON-shape enforcement -- one repair retry on unparsable output -- so
providers stay dumb. Retry, throttling and failover across a tier's candidates
belong to :class:`~pr_review.providers.dispatch.Dispatcher`.
"""

from __future__ import annotations

from pr_review.axes.base import (
    AXES,
    REPORT_ONLY_PROBLEMS,
    allowed_severities,
    clamp_severity,
    is_whole_pr,
)
from pr_review.config import Config
from pr_review.depths.base import DEPTHS
from pr_review.models import Finding, PRContext, ReviewContext, ReviewTask
from pr_review.providers.base import extract_json
from pr_review.providers.dispatch import Dispatcher
from pr_review.providers.selector import ModelSelector

_SYSTEM = (
    "You are a rigorous senior code reviewer. You only report concrete, "
    "evidence-backed problems. You respond with a single JSON object and nothing "
    "else."
)

_SEVERITIES = {"blocker", "warning", "info"}


class Reviewer:
    def __init__(
        self,
        config: Config,
        selector: ModelSelector,
        dispatcher: Dispatcher | None = None,
    ) -> None:
        self.config = config
        self.selector = selector
        self.dispatcher = dispatcher or Dispatcher(config, selector)

    def run(self, task: ReviewTask, ctx: ReviewContext, pr: PRContext) -> list[Finding]:
        axis = AXES.create(task.axis)
        depth = DEPTHS.create(task.depth)
        prompt = self._prompt(axis, depth, task, ctx, pr)

        completion = self.dispatcher.complete(task.model_tier, prompt, system=_SYSTEM)
        findings = self._parse(completion.text, task, axis)
        if findings is None:
            completion = self.dispatcher.complete(
                task.model_tier,
                prompt + "\n\nYour previous reply was not valid JSON. Reply with ONLY "
                "the JSON object described above.",
                system=_SYSTEM,
            )
            findings = self._parse(completion.text, task, axis) or []
        # Which model actually answered matters when a tier has degraded, so
        # carry it on the finding rather than only in the run-level notes.
        for f in findings:
            f.extra.setdefault("model", completion.candidate.label())
            if completion.degraded:
                f.extra.setdefault("degraded", True)
        return findings[: self.config.max_axis_findings]

    # ------------------------------------------------------------------ #

    def _prompt(self, axis, depth, task: ReviewTask, ctx: ReviewContext, pr: PRContext) -> str:
        severities = "|".join(allowed_severities(axis))
        scope_rule = (
            'This axis judges the pull request as a WHOLE. Set "file" to "-" '
            "unless the problem is genuinely localised to one changed file; do "
            "not pick an arbitrary file to attach the finding to.\n"
            if is_whole_pr(axis)
            else 'Set "file" to the changed file the problem is in.\n'
        )
        return f"""TASK: review
AXIS: {task.axis}
DEPTH: {task.depth}

{axis.goal()}

{depth.instruction()}

{REPORT_ONLY_PROBLEMS}

{scope_rule}
PR TITLE: {pr.title}
PR BODY: {pr.body[:1500]}

--- CONTEXT ---
{ctx.render()}
--- END CONTEXT ---

Respond with ONLY this JSON object:
{{"axis": "{task.axis}",
  "findings": [
    {{"severity": "{severities}",
      "file": "<path from the context>",
      "line": <int or null>,
      "issue": "<one sentence>",
      "evidence": "<quote a line or fact from the context above>",
      "reasoning": "<why it is a problem, at most two sentences>",
      "confidence": <0.0-1.0>}}
  ]}}
An empty findings list is valid and expected for a clean change. Do not invent
findings or give generic advice. Every finding must cite evidence visible in the
context above."""

    def _parse(self, raw: str, task: ReviewTask, axis: object) -> list[Finding] | None:
        try:
            data = extract_json(raw)
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None
        items = data.get("findings")
        if not isinstance(items, list):
            return None
        out: list[Finding] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            sev = str(it.get("severity", "info")).lower()
            if sev not in _SEVERITIES:
                sev = "info"
            # The model is asked for the allowed set, but a ceiling that is only
            # requested is not a ceiling. Enforce it here.
            capped = clamp_severity(axis, sev)
            line = it.get("line")
            try:
                line_i = int(line) if line is not None else None
            except (TypeError, ValueError):
                line_i = None
            conf = it.get("confidence", 0.5)
            try:
                conf_f = max(0.0, min(1.0, float(conf)))
            except (TypeError, ValueError):
                conf_f = 0.5
            issue = str(it.get("issue", "")).strip()
            if not issue:
                continue
            extra: dict[str, object] = {}
            if capped != sev:
                extra["severity_capped_from"] = sev
            out.append(
                Finding(
                    axis=task.axis,
                    severity=capped,
                    file=str(it.get("file", "")).strip() or "-",
                    line=line_i,
                    issue=issue,
                    evidence=str(it.get("evidence", "")).strip(),
                    reasoning=str(it.get("reasoning", "")).strip(),
                    confidence=conf_f,
                    depth=task.depth,
                    extra=extra,
                )
            )
        return out
