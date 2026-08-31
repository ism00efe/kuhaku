"""``pr-review`` command-line entrypoint.

Wires a PR source to the engine and writes the reports. The engine itself never
imports this module.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pr_review.config import load_config
from pr_review.errors import PRReviewError
from pr_review.pipeline import Pipeline
from pr_review.report.base import REPORTERS
from pr_review.source.base import PRSource
from pr_review.source.github import GitHubActionsSource
from pr_review.source.local_git import LocalGitSource


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pr-review", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="review a pull request / diff")
    run.add_argument("--repo", default=".", help="path to the repository checkout")
    run.add_argument("--base", default="", help="base ref (local mode)")
    run.add_argument("--head", default="HEAD", help="head ref (local mode)")
    run.add_argument("--title", default="", help="override PR title (local mode)")
    run.add_argument("--body", default="", help="override PR body (local mode)")
    run.add_argument(
        "--github-event",
        default="",
        help="path to a GitHub Actions event payload (pull_request)",
    )
    run.add_argument("--provider", default="", help="force a provider for every tier")
    run.add_argument("--no-classifier", action="store_true", help="use the heuristic planner only")
    run.add_argument("--no-verify", action="store_true", help="skip finding verification")
    run.add_argument("--out", default="", help="write the markdown report here")
    run.add_argument("--json-out", default="", help="write the json report here")
    run.add_argument("--quiet", action="store_true", help="do not print the report to stdout")

    sub.add_parser("providers", help="list registered providers")
    sub.add_parser("axes", help="list registered review axes")
    return p


def _run(args: argparse.Namespace) -> int:
    overrides: dict = {}
    if args.provider:
        overrides["force_provider"] = args.provider
    if args.no_classifier:
        overrides["classifier_enabled"] = False
    if args.no_verify:
        from dataclasses import replace

        cfg = load_config(args.repo, overrides=overrides)
        cfg = replace(cfg, verification=replace(cfg.verification, enabled=False))
    else:
        cfg = load_config(args.repo, overrides=overrides)

    source: PRSource
    if args.github_event or (not args.base):
        source = GitHubActionsSource(
            event_path=args.github_event or None,
            repo_root=args.repo,
            diff_bytes=cfg.limits.diff_bytes,
        )
        try:
            source.load()
        except PRReviewError:
            if args.base or args.github_event:
                raise
            # No event and no --base: fall back to HEAD~1..HEAD locally.
            source = LocalGitSource(
                args.repo, base_ref="HEAD~1", head_ref="HEAD",
                title=args.title, body=args.body, diff_bytes=cfg.limits.diff_bytes,
            )
    else:
        source = LocalGitSource(
            args.repo,
            base_ref=args.base,
            head_ref=args.head,
            title=args.title,
            body=args.body,
            diff_bytes=cfg.limits.diff_bytes,
        )

    result = Pipeline(cfg).run(source)

    rendered: dict[str, str] = {}
    for name in cfg.reporters:
        if REPORTERS.has(name):
            rendered[name] = REPORTERS.create(name).render(result)

    if args.out and "markdown" in rendered:
        Path(args.out).write_text(rendered["markdown"], "utf-8")
    if args.json_out and "json" in rendered:
        Path(args.json_out).write_text(rendered["json"], "utf-8")
    if not args.quiet:
        print(rendered.get("markdown", next(iter(rendered.values()), "")))

    blockers = [
        vf for vf in result.surfaced()
        if vf.finding.severity == "blocker"
        and vf.result.verdict.value in ("valid", "uncertain")
    ]
    return 2 if blockers else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return _run(args)
        if args.command == "providers":
            from pr_review.providers.base import PROVIDERS

            print("\n".join(PROVIDERS.names()))
            return 0
        if args.command == "axes":
            from pr_review.axes.base import AXES

            print("\n".join(AXES.names()))
            return 0
    except PRReviewError as exc:
        print(f"pr-review: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
