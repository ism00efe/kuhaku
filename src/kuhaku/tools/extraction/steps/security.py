"""The security step: core's :class:`~kuhaku.core.security.input_policy.InputPolicy` on
everything the model would see.

Every extractor runs it, first, with the options from its :class:`ExtractionSettings`;
with all of them off it does nothing. The checks themselves live in core, so one added
there reaches extraction without a change here. What costs nothing stays on without
this step too: outside text is always fenced, and the rule that it is data, not
instructions, is not replaceable (see :mod:`..prompts`).

It works in the prepare phase, so it acts before the model is called and
:meth:`Extractor.preview` shows its effect.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from kuhaku.core.security.input_policy import InputPolicy, InputSecurity

from ..context import ExtractionContext
from ..pipeline import Step
from ..result import Issue, IssueKind

SECURITY_STEP = "security"

logger = logging.getLogger(__name__)


class SecurityStep(Step):
    """Refuses the source text when an input check says so (status ``BLOCKED``, the model
    is never called), then masks the source text, the history, the reference, the
    earlier turn's values and any block another step already contributed.

    With masking on, the model sees placeholders such as ``[EMAIL]`` and extracts those.
    What was masked -- labels and counts, never values -- is in
    ``result.metadata["redactions"]``.
    """

    name = SECURITY_STEP

    def __init__(self, options: InputSecurity | None = None) -> None:
        self.options = options or InputSecurity()

    def prepare(self, ctx: ExtractionContext) -> None:
        policy = InputPolicy(self.options)

        ctx.text = policy.mask(ctx.text)
        ctx.history = tuple(replace(m, content=policy.mask(m.content)) for m in ctx.history)
        ctx.reference = _map_strings(ctx.reference, policy.mask)
        if ctx.prior_data is not None:
            ctx.prior_data = _map_strings(ctx.prior_data, policy.mask)
        # A step that ran its prepare earlier has already rendered its block from the
        # unmasked data. Masked with a separate policy: that data was counted above.
        blocks = InputPolicy(self.options)
        ctx.prompt_blocks[:] = [(title, blocks.mask(body)) for title, body in ctx.prompt_blocks]
        if self.options.sanitize:
            ctx.metadata["redactions"] = policy.redactions()

        reason = policy.screen(ctx.text)
        if reason is None:
            return
        logger.warning("extraction input blocked", extra={"reason": reason})
        ctx.issues.append(
            Issue(IssueKind.BLOCKED, reason, f"the input guard refused the text ({reason})")
        )
        ctx.stop()


def _map_strings(node: Any, fn: Callable[[str], str]) -> Any:
    if isinstance(node, str):
        return fn(node)
    if isinstance(node, dict):
        return {key: _map_strings(value, fn) for key, value in node.items()}
    if isinstance(node, list | tuple):
        return type(node)(_map_strings(value, fn) for value in node)
    return node
