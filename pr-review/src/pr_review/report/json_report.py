"""Machine-readable reporter -- the full result as JSON."""

from __future__ import annotations

import dataclasses
import json

from pr_review.models import ReviewResult
from pr_review.report.base import REPORTERS


def _default(o: object) -> object:
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return dataclasses.asdict(o)
    if hasattr(o, "value"):  # Enum
        return o.value
    return str(o)


@REPORTERS.register("json")
class JSONReporter:
    name = "json"
    file_extension = "json"

    def render(self, result: ReviewResult) -> str:
        payload = {
            "schema": "pr-review/result@0.1",
            "pr": dataclasses.asdict(result.pr),
            "repo": dataclasses.asdict(result.repo),
            "analysis": dataclasses.asdict(result.analysis),
            "plan": dataclasses.asdict(result.plan),
            "tasks": [dataclasses.asdict(t) for t in result.tasks],
            "findings": [
                {
                    "finding": dataclasses.asdict(vf.finding),
                    "verification": dataclasses.asdict(vf.result),
                }
                for vf in result.findings
            ],
            "errors": [dataclasses.asdict(e) for e in result.errors],
            "notes": list(result.notes),
            "coverage": {
                "changed_files": len(result.changed_paths()),
                "reviewed_files": len(result.reviewed_paths()),
                "unreviewed_files": result.unreviewed_paths(),
                "passes_per_axis": result.pass_count(),
            },
            "stats": result.stats,
        }
        return json.dumps(payload, indent=2, default=_default)
