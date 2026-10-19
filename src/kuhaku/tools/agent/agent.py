"""``Agent``: a bounded tool-using loop that only ever shows the model the tools the
caller is entitled to.

One run is: filter the tools by the caller's ``AuthContext``; ask the model (with
``chat_with_tools``, native or emulated); run the calls it made; send the results back
marked as data; repeat until the model answers, a budget runs out, or a call needs a
human's approval -- in which case the run pauses as a sealed :class:`AgentState` and
continues in :meth:`Agent.resume`.
"""

from __future__ import annotations

import contextvars
import dataclasses
import logging
import math
import secrets
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from kuhaku.core.auth import AuthContext
from kuhaku.core.config import Settings, get_settings
from kuhaku.core.exceptions import ConfigError
from kuhaku.core.llm import build_llm_provider
from kuhaku.core.llm.base import (
    GenerationOptions,
    LLMError,
    LLMProvider,
    LLMResponse,
    TokenUsage,
    ToolCallingLLMProvider,
    combine_usage,
)
from kuhaku.core.llm.token_tracking import TokenTrackingLLM
from kuhaku.core.llm.wrapper import as_tool_calling_provider
from kuhaku.core.models import ExecutionResult, Message, ToolCall, ToolCallPart
from kuhaku.core.observability import bind_trace_id, instrumented_step
from kuhaku.core.sanitization import sanitize
from kuhaku.core.security.audit import AuditWriteError
from kuhaku.core.security.audit import record as record_audit

from .prompts import build_system_prompt, new_boundary, render_output, truncate, wrap
from .replay import ConsumedStateStore, InMemoryConsumedStateStore
from .result import (
    LIMIT_STEPS,
    LIMIT_TIME,
    LIMIT_TOKENS,
    LIMIT_TOOL_CALLS,
    AgentResult,
    AgentStatus,
    CallOutcome,
    ToolExecution,
)
from .state import (
    STATE_VERSION,
    AgentState,
    AgentStateError,
    call_digest,
    identity_of,
    message_from_dict,
    message_to_dict,
    seal,
    verify,
)
from .tool import AgentTool, ArgumentsInvalid, ToolError, is_entitled

logger = logging.getLogger(__name__)

InputGuard = Callable[[str], tuple[bool, str]]
"""``(text) -> (safe, reason)`` -- the shape of :func:`kuhaku.core.security.inspect_query`."""

OutputFilter = Callable[[str], str]
"""``(text) -> text`` -- the shape of :func:`kuhaku.core.sanitization.sanitize`."""

# How long a paused state stays resumable by default: a day covers a human approval
# queue without leaving approvals valid indefinitely.
DEFAULT_STATE_MAX_AGE = 24 * 3600.0
# Tolerated clock skew between the worker that paused a run and the one resuming it.
_ISSUED_AT_SKEW = 60.0


@dataclass
class _Run:
    """The mutable state of one run (or resumed segment of one)."""

    input: str
    auth_context: AuthContext | None
    messages: list[Message]
    boundary: str
    trace_id: str
    steps: int = 0
    tool_calls: int = 0
    invalid_calls: int = 0
    usage: TokenUsage | None = None
    elapsed_before: float = 0.0
    started: float = field(default_factory=time.perf_counter)
    executions: list[ToolExecution] = field(default_factory=list)
    llms: list[str] = field(default_factory=list)
    approvals: list[dict[str, Any]] = field(default_factory=list)

    def elapsed(self) -> float:
        return self.elapsed_before + (time.perf_counter() - self.started)

    def total_tokens(self) -> int:
        if self.usage is None:
            return 0
        return (self.usage.prompt_tokens or 0) + (self.usage.completion_tokens or 0)


@dataclass(frozen=True)
class _Job:
    index: int
    call: ToolCall
    tool: AgentTool
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class _Timeout(Exception):
    pass


class _Unavailable(Exception):
    pass


class Agent:
    """A tool-using agent scoped to the caller's identity.

    ``tools`` are :class:`AgentTool`\\ s (a plain callable is wrapped in one with no
    access tags). ``llm`` takes any :class:`~kuhaku.core.llm.base.LLMProvider`; without
    native tool calling it is emulated through structured output. When omitted it is
    built from ``settings`` (or the environment), as ``Extractor`` and ``RAG`` do.
    ``fallback_llms`` are tried in order, per model call, when the one before raises an
    :class:`~kuhaku.core.llm.base.LLMError`. Every provider is wrapped in
    :class:`~kuhaku.core.llm.token_tracking.TokenTrackingLLM` unless
    ``enable_token_tracking=False``. ``instructions`` are the application's part of the
    system prompt; the framework's safety rules always come first.

    Budgets: ``max_steps`` model calls, ``max_tool_calls`` tool calls (counted as the
    model makes them, valid or not), optionally ``max_total_tokens`` (only usage the
    provider reports counts) and ``max_seconds`` of wall-clock time (checked between
    steps -- a running model call or tool is not interrupted; time spent paused for
    approval does not count). Running out stops the run with ``STEP_LIMIT``.
    ``max_invalid_calls`` bounds calls with malformed or schema-violating arguments,
    or to unknown tools, before the run is ``FAILED``.

    Calls the model makes in one turn run one after another, in order; with
    ``max_concurrency > 1`` up to that many run at once in threads (their results are
    still sent back in the order the model made the calls). Tool output is truncated to
    ``max_tool_output_chars`` before the model sees it.

    ``state_secret`` keys the seal on a paused run's :class:`AgentState`. Without one a
    random key is generated per ``Agent`` instance, so a state can only be resumed by
    the instance that paused it; pass the same secret to resume in another process.
    A state expires ``state_max_age`` seconds after it was paused (default one day),
    and resumes only once: ``consumed_states`` remembers resumed states -- the default
    :class:`~kuhaku.tools.agent.InMemoryConsumedStateStore` sees only this process, so
    workers sharing a ``state_secret`` must share a store as well (e.g.
    :class:`~kuhaku.tools.agent.SQLiteConsumedStateStore`).

    Two optional hooks for checks core does not yet apply on its own here:
    ``input_guard`` vets the input before the model sees it (pass
    :func:`kuhaku.core.security.inspect_query`), and ``tool_output_filter`` rewrites
    every tool result before the model sees it (pass
    :func:`kuhaku.core.sanitization.sanitize` to mask PII).
    """

    def __init__(
        self,
        tools: Sequence[AgentTool | Callable[..., Any]],
        *,
        llm: LLMProvider | None = None,
        settings: Settings | None = None,
        instructions: str | None = None,
        fallback_llms: Sequence[LLMProvider] = (),
        enable_token_tracking: bool = True,
        max_steps: int = 10,
        max_tool_calls: int = 20,
        max_total_tokens: int | None = None,
        max_seconds: float | None = None,
        max_invalid_calls: int = 3,
        max_concurrency: int = 1,
        max_tool_output_chars: int = 20_000,
        state_secret: bytes | str | None = None,
        state_max_age: float = DEFAULT_STATE_MAX_AGE,
        consumed_states: ConsumedStateStore | None = None,
        input_guard: InputGuard | None = None,
        tool_output_filter: OutputFilter | None = None,
        options: GenerationOptions | None = None,
    ) -> None:
        for label, value, minimum in (
            ("max_steps", max_steps, 1),
            ("max_tool_calls", max_tool_calls, 0),
            ("max_invalid_calls", max_invalid_calls, 0),
            ("max_concurrency", max_concurrency, 1),
            ("max_tool_output_chars", max_tool_output_chars, 1),
        ):
            if value < minimum:
                raise ValueError(f"{label} must be >= {minimum}; got {value}.")
        if max_total_tokens is not None and max_total_tokens < 1:
            raise ValueError("max_total_tokens must be >= 1, or None.")
        if max_seconds is not None and max_seconds <= 0:
            raise ValueError("max_seconds must be > 0, or None.")
        if not math.isfinite(state_max_age) or state_max_age <= 0:
            raise ValueError("state_max_age must be a finite number of seconds > 0.")

        self._tools: dict[str, AgentTool] = {}
        for item in tools:
            tool = item if isinstance(item, AgentTool) else AgentTool(item)
            if tool.name in self._tools:
                raise ConfigError(f"Two agent tools are named '{tool.name}'.")
            self._tools[tool.name] = tool

        resolved = settings or get_settings()
        built = llm if llm is not None else build_llm_provider(resolved)
        track = _token_tracked if enable_token_tracking else _untracked
        self._llms: list[ToolCallingLLMProvider] = [
            as_tool_calling_provider(track(provider))
            for provider in (built, *fallback_llms)
        ]
        self.instructions = instructions
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.max_total_tokens = max_total_tokens
        self.max_seconds = max_seconds
        self.max_invalid_calls = max_invalid_calls
        self.max_concurrency = max_concurrency
        self.max_tool_output_chars = max_tool_output_chars
        self.input_guard = input_guard
        self.tool_output_filter = tool_output_filter
        self.options = options
        self._audit_enabled = resolved.audit_enabled
        self._audit_log_path = resolved.audit_log_path
        if isinstance(state_secret, str):
            state_secret = state_secret.encode("utf-8")
        self._state_key = state_secret or secrets.token_bytes(32)
        self.state_max_age = float(state_max_age)
        self._consumed_states: ConsumedStateStore = (
            consumed_states if consumed_states is not None else InMemoryConsumedStateStore()
        )

    # --- inspection -------------------------------------------------------------------

    @property
    def tools(self) -> tuple[AgentTool, ...]:
        return tuple(self._tools.values())

    @property
    def llm(self) -> ToolCallingLLMProvider:
        return self._llms[0]

    def visible_tools(self, auth_context: AuthContext | None) -> tuple[AgentTool, ...]:
        """The tools that exist for ``auth_context`` -- exactly what the model is shown."""

        return tuple(t for t in self._tools.values() if is_entitled(t, auth_context))

    # --- entry points -----------------------------------------------------------------

    def run(
        self,
        input: str,
        *,
        auth_context: AuthContext | None,
        history: Sequence[Message] | None = None,
        trace_id: str | None = None,
    ) -> AgentResult:
        """Run the agent on ``input`` for the caller identified by ``auth_context``.

        ``auth_context`` is required (``None`` is allowed, and sees only untagged
        tools) so that forgetting it is a ``TypeError``, not a quietly unscoped run.
        ``history`` (earlier turns) goes between the system prompt and ``input``.
        """

        if not isinstance(input, str):
            raise TypeError(f"input must be a str, got {type(input).__name__}.")
        prior = list(history or ())
        for message in prior:
            if not isinstance(message, Message):
                raise TypeError(
                    f"history must contain Message objects; got {type(message).__name__}."
                )
        with (
            bind_trace_id(trace_id) as tid,
            instrumented_step("agent", logger, phase="run") as rec,
        ):
            boundary = new_boundary()
            run = _Run(
                input=input,
                auth_context=auth_context,
                messages=[
                    Message.system(build_system_prompt(boundary, self.instructions)),
                    *prior,
                    Message.user(input),
                ],
                boundary=boundary,
                trace_id=tid,
            )
            if self.input_guard is not None:
                safe, reason = self.input_guard(input)
                if not safe:
                    result = self._finish(
                        run, AgentStatus.FAILED, error=f"input rejected: {reason}"
                    )
                    rec.set(outcome=result.status.value)
                    return result
            result = self._loop(run)
            rec.set(outcome=result.status.value, steps=result.steps)
            return result

    def resume(
        self,
        state: AgentState | str | Mapping[str, Any],
        approvals: Mapping[str, bool | str] | None = None,
        *,
        auth_context: AuthContext | None,
        trace_id: str | None = None,
        approved_by: str | None = None,
    ) -> AgentResult:
        """Continue a run that paused with ``AWAITING_APPROVAL``.

        ``approvals`` maps a pending call's id to ``True`` (run it) or ``False``; a
        pending call left out is denied. A denied call never runs -- the model is told it
        was not approved. Instead of ``True`` the value may be the call's digest from
        ``state.pending_digests`` (SHA-256 of tool name and canonical arguments), which
        approves only that exact call. Either way the approval is bound to the digest
        recorded when the run paused. ``state`` is the paused result's ``state``, or its
        ``to_json()``/``to_dict()``.

        ``approved_by`` names the human who decided (default: ``auth_context``'s
        identity); it is recorded with each decision in ``result.metadata["approvals"]``
        and in the audit record.

        Refused with :class:`AgentStateError` when the seal does not verify, when
        ``auth_context`` is not the identity the run was paused for, when the state is
        older than ``state_max_age``, when its pending calls do not match their recorded
        digests, or when the state was already resumed (per ``consumed_states`` -- by
        default in this process only). Entitlements are re-checked against
        ``auth_context``: a tool the caller lost access to while the run waited is
        unknown to the model again, and a pending call to it does not run, approved or
        not.
        """

        if isinstance(state, str):
            state = AgentState.from_json(state)
        elif not isinstance(state, AgentState):
            state = AgentState.from_dict(dict(state))
        verify(state, self._state_key)
        payload = state.payload
        issued_at = self._check_age(payload)
        if payload.get("identity") != identity_of(auth_context):
            raise AgentStateError(
                "This agent state belongs to a different identity and cannot be resumed "
                "with this auth_context."
            )
        digests = state.pending_digests
        pending_ids = list(digests)
        _check_pending(payload, digests)
        decisions = dict(approvals or {})
        unknown = sorted(set(decisions) - set(pending_ids))
        if unknown:
            raise ValueError(
                f"approvals name calls that are not pending: {', '.join(unknown)}."
            )
        approved: set[str] = set()
        for call_id, decision in decisions.items():
            if isinstance(decision, bool):
                if decision:
                    approved.add(call_id)
            elif isinstance(decision, str) and len(decision) == 64:
                if decision != digests[call_id]:
                    raise ValueError(
                        f"approval for {call_id!r} names a different call than the one "
                        f"pending (digest mismatch)."
                    )
                approved.add(call_id)
            else:
                raise TypeError(
                    f"approval for {call_id!r} must be True or False (or the pending "
                    f"call's digest)."
                )
        nonce = str(payload.get("nonce", ""))
        if not nonce or not self._consumed_states.consume(
            nonce, issued_at + self.state_max_age
        ):
            raise AgentStateError("This agent state was already resumed.")
        approver = approved_by if approved_by is not None else (
            auth_context.identity if auth_context is not None else None
        )

        with (
            bind_trace_id(trace_id or state.trace_id or None) as tid,
            instrumented_step("agent", logger, phase="resume") as rec,
        ):
            usage = payload.get("usage")
            run = _Run(
                input=str(payload.get("input", "")),
                auth_context=auth_context,
                messages=[message_from_dict(m) for m in payload["messages"]],
                boundary=str(payload["boundary"]),
                trace_id=tid,
                steps=int(payload.get("steps", 0)),
                tool_calls=int(payload.get("tool_calls", 0)),
                invalid_calls=int(payload.get("invalid_calls", 0)),
                usage=TokenUsage(usage[0], usage[1]) if usage else None,
                elapsed_before=float(payload.get("elapsed", 0.0)),
                llms=list(payload.get("llms", ())),
                approvals=[
                    {
                        "call_id": cid,
                        "digest": digests[cid],
                        "approved": cid in approved,
                        "approved_by": approver,
                    }
                    for cid in pending_ids
                ],
            )
            tools = self._visible(auth_context)
            calls = run.messages[-1].tool_calls if run.messages else ()
            if not calls:
                raise AgentStateError("This agent state has no pending tool calls.")
            result = self._execute_turn(run, tools, calls, approved=approved)
            if result is None:
                result = self._loop(run)
            rec.set(outcome=result.status.value, steps=result.steps)
            return result

    # --- the loop ---------------------------------------------------------------------

    def _loop(self, run: _Run) -> AgentResult:
        tools = self._visible(run.auth_context)
        while True:
            limit = self._exhausted(run)
            if limit is not None:
                return self._finish(run, AgentStatus.STEP_LIMIT, limit=limit)
            try:
                response = self._step(run, tools)
            except _Unavailable as exc:
                return self._finish(run, AgentStatus.FAILED, error=str(exc))

            calls = response.tool_calls
            if not calls:
                run.messages.append(response.message)
                return self._finish(run, AgentStatus.COMPLETED, text=response.text)
            message = response.message
            ids = [call.id for call in calls]
            if len(set(ids)) != len(ids) or not all(ids):
                # Approvals and results are keyed by call id; two calls sharing one
                # would let a single approval run both.
                message, calls = _rekey(message)

            # A turn that would overrun a budget is neither executed nor added to the
            # transcript: its calls would have no results, and a transcript with
            # unanswered calls cannot be sent to a model again.
            if run.tool_calls + len(calls) > self.max_tool_calls:
                return self._finish(run, AgentStatus.STEP_LIMIT, limit=LIMIT_TOOL_CALLS)
            limit = self._exhausted(run, after_call=True)
            if limit is not None:
                return self._finish(run, AgentStatus.STEP_LIMIT, limit=limit)

            run.messages.append(message)
            run.tool_calls += len(calls)
            pending = [call for call in calls if self._needs_approval(call, tools)]
            if pending:
                return self._pause(run, pending)
            result = self._execute_turn(run, tools, calls, approved=set())
            if result is not None:
                return result

    def _exhausted(self, run: _Run, *, after_call: bool = False) -> str | None:
        if not after_call and run.steps >= self.max_steps:
            return LIMIT_STEPS
        if (
            self.max_total_tokens is not None
            and run.total_tokens() >= self.max_total_tokens
        ):
            return LIMIT_TOKENS
        if self.max_seconds is not None and run.elapsed() >= self.max_seconds:
            return LIMIT_TIME
        return None

    def _step(self, run: _Run, tools: Mapping[str, AgentTool]) -> LLMResponse:
        specs = [tool.spec for tool in tools.values()]
        with instrumented_step(
            "agent.step", logger, turn=run.steps + 1, tool_count=len(specs)
        ) as rec:
            for position, llm in enumerate(self._llms):
                try:
                    if specs:
                        response = llm.chat_with_tools(
                            run.messages, specs, options=self.options
                        )
                    else:
                        # chat_with_tools refuses an empty tool list: a caller entitled to
                        # no tool gets a plain conversation.
                        response = llm.chat(run.messages, options=self.options)
                except LLMError as exc:
                    if position == len(self._llms) - 1:
                        rec.set(outcome="llm_unavailable")
                        raise _Unavailable(f"LLM unavailable: {exc}") from exc
                    logger.warning(
                        "agent provider unavailable, falling back",
                        extra={"llm": llm.name, "error": str(exc)},
                    )
                    continue
                run.steps += 1
                run.usage = combine_usage(run.usage, response.usage)
                run.llms.append(llm.name)
                rec.set(llm=llm.name, tool_call_count=len(response.tool_calls))
                return response
        raise AssertionError("unreachable: the agent always has at least one provider")

    def _needs_approval(self, call: ToolCall, tools: Mapping[str, AgentTool]) -> bool:
        tool = tools.get(call.name) if call.valid else None
        if tool is None or not tool.requires_approval:
            return False
        # Asking a human to approve arguments the tool would reject anyway is noise;
        # such a call fails like any other invalid call instead.
        try:
            tool.bind(call.arguments)
        except ArgumentsInvalid:
            return False
        return True

    def _pause(self, run: _Run, pending: list[ToolCall]) -> AgentResult:
        payload = {
            "v": STATE_VERSION,
            "nonce": secrets.token_hex(16),
            "issued_at": time.time(),
            "trace_id": run.trace_id,
            "identity": identity_of(run.auth_context),
            "input": run.input,
            "boundary": run.boundary,
            "messages": [message_to_dict(m) for m in run.messages],
            "pending": [{"id": call.id, "digest": call_digest(call)} for call in pending],
            "steps": run.steps,
            "tool_calls": run.tool_calls,
            "invalid_calls": run.invalid_calls,
            "elapsed": run.elapsed(),
            "usage": (
                [run.usage.prompt_tokens, run.usage.completion_tokens]
                if run.usage
                else None
            ),
            "llms": run.llms,
        }
        return self._finish(
            run,
            AgentStatus.AWAITING_APPROVAL,
            pending=tuple(pending),
            state=seal(payload, self._state_key),
        )

    # --- running one turn's calls -----------------------------------------------------

    def _execute_turn(
        self,
        run: _Run,
        tools: Mapping[str, AgentTool],
        calls: Sequence[ToolCall],
        *,
        approved: set[str],
    ) -> AgentResult | None:
        """Run (or refuse) every call of one model turn and append their results.
        Returns a final result when the run cannot continue, else ``None``."""

        executions: list[ToolExecution | None] = [None] * len(calls)
        jobs: list[_Job] = []
        for index, call in enumerate(calls):
            refused = self._refusal(run, tools, call, approved)
            if isinstance(refused, ToolExecution):
                executions[index] = refused
            else:
                tool, args, kwargs = refused
                jobs.append(_Job(index, call, tool, args, kwargs))

        if self.max_concurrency > 1 and len(jobs) > 1:
            with ThreadPoolExecutor(
                max_workers=min(self.max_concurrency, len(jobs))
            ) as pool:
                futures = [
                    pool.submit(contextvars.copy_context().run, self._execute, run, job)
                    for job in jobs
                ]
                for job, future in zip(jobs, futures, strict=True):
                    executions[job.index] = future.result()
        else:
            for job in jobs:
                executions[job.index] = self._execute(run, job)

        for execution in executions:
            assert execution is not None
            run.executions.append(execution)
            run.messages.append(self._result_message(run, execution))

        if run.invalid_calls > self.max_invalid_calls:
            return self._finish(
                run,
                AgentStatus.FAILED,
                error=(
                    f"The model made more than {self.max_invalid_calls} invalid tool "
                    f"calls (unknown tools or malformed arguments)."
                ),
            )
        return None

    def _refusal(
        self,
        run: _Run,
        tools: Mapping[str, AgentTool],
        call: ToolCall,
        approved: set[str],
    ) -> ToolExecution | tuple[AgentTool, tuple[Any, ...], dict[str, Any]]:
        """A call that must not run, as its :class:`ToolExecution`; otherwise the tool
        and bound arguments to run it with."""

        if not call.valid:
            run.invalid_calls += 1
            return _refused(
                call,
                CallOutcome.INVALID_ARGUMENTS,
                f"Malformed arguments for tool '{call.name}': {call.arguments_error}. "
                f"Send the arguments as a JSON object.",
            )
        tool = tools.get(call.name)
        if tool is None:
            # The same answer whether the tool is hidden from this caller or does not
            # exist at all -- telling them apart would confirm a hidden tool exists.
            run.invalid_calls += 1
            available = ", ".join(sorted(tools)) or "none"
            return _refused(
                call,
                CallOutcome.UNKNOWN_TOOL,
                f"Unknown tool. Available tools: {available}.",
            )
        try:
            args, kwargs = tool.bind(call.arguments)
        except ArgumentsInvalid as exc:
            run.invalid_calls += 1
            return _refused(
                call,
                CallOutcome.INVALID_ARGUMENTS,
                f"Invalid arguments for tool '{tool.name}': {exc}",
            )
        if tool.requires_approval and call.id not in approved:
            return _refused(
                call,
                CallOutcome.DENIED,
                f"The call to tool '{tool.name}' was not approved and did not run.",
            )
        return tool, args, kwargs

    def _execute(self, run: _Run, job: _Job) -> ToolExecution:
        tool, call = job.tool, job.call
        metadata: dict[str, Any] = {"tool_call_id": call.id, "tool": tool.name}
        start = time.perf_counter()
        with instrumented_step(
            f"agent.tool.{tool.name}", logger, tool=tool.name
        ) as rec:
            try:
                output = _call_with_timeout(
                    lambda: tool.invoke(job.args, job.kwargs, run.auth_context),
                    tool.timeout,
                )
            except _Timeout:
                outcome = CallOutcome.TIMEOUT
                result = ExecutionResult(
                    output=None,
                    success=False,
                    error=f"Tool '{tool.name}' did not finish within {tool.timeout} seconds.",
                    metadata=metadata,
                )
            except ToolError as exc:
                outcome = CallOutcome.ERROR
                result = ExecutionResult(
                    output=None,
                    success=False,
                    error=f"Tool '{tool.name}' failed: {exc}",
                    metadata={**metadata, "error_type": "ToolError"},
                )
            except Exception as exc:
                # A tool's failure is a result, not a crash. The exception text stays in
                # the operator's log; the model is told only that the tool failed,
                # because that text can carry anything.
                logger.warning(
                    "agent tool raised",
                    extra={"tool": tool.name, "error_type": type(exc).__name__},
                    exc_info=True,
                )
                outcome = CallOutcome.ERROR
                result = ExecutionResult(
                    output=None,
                    success=False,
                    error=f"Tool '{tool.name}' failed.",
                    metadata={**metadata, "error_type": type(exc).__name__},
                )
            else:
                outcome = CallOutcome.OK
                result = ExecutionResult(output=output, metadata=metadata)
            rec.set(outcome=outcome.value)
        return ToolExecution(call, result, outcome, time.perf_counter() - start)

    def _result_message(self, run: _Run, execution: ToolExecution) -> Message:
        result = execution.result
        if result.success:
            content = render_output(result.output)
        else:
            content = result.error or "Tool call failed."
        if self.tool_output_filter is not None:
            content = self.tool_output_filter(content)
        content = wrap(truncate(content, self.max_tool_output_chars), run.boundary)
        call = execution.call
        return Message.tool_result(
            call.id, content, name=call.name, is_error=not result.success
        )

    # --- finishing --------------------------------------------------------------------

    def _visible(self, auth_context: AuthContext | None) -> dict[str, AgentTool]:
        return {t.name: t for t in self.visible_tools(auth_context)}

    def _finish(
        self,
        run: _Run,
        status: AgentStatus,
        *,
        text: str = "",
        limit: str | None = None,
        error: str | None = None,
        pending: tuple[ToolCall, ...] = (),
        state: AgentState | None = None,
    ) -> AgentResult:
        result = AgentResult(
            status=status,
            text=text,
            messages=tuple(run.messages),
            tool_calls=tuple(run.executions),
            pending=pending,
            state=state,
            usage=run.usage,
            steps=run.steps,
            limit=limit,
            error=error,
            trace_id=run.trace_id,
            metadata=(
                {"llms": list(run.llms), "approvals": list(run.approvals)}
                if run.approvals
                else {"llms": list(run.llms)}
            ),
        )
        self._audit(run, result)
        return result

    def _audit(self, run: _Run, result: AgentResult) -> None:
        """One audit record per run or resume: who, what happened, which tools ran --
        never argument values or tool output, which can be anything."""

        ran = [
            e.call.name
            for e in result.tool_calls
            if e.outcome in (CallOutcome.OK, CallOutcome.ERROR, CallOutcome.TIMEOUT)
        ]
        outcomes: dict[str, int] = {}
        for execution in result.tool_calls:
            outcomes[execution.outcome.value] = (
                outcomes.get(execution.outcome.value, 0) + 1
            )
        try:
            record_audit(
                self._audit_log_path,
                enabled=self._audit_enabled,
                trace_id=run.trace_id,
                raw_question=run.input,
                sanitized_retrieval_query=sanitize(run.input),
                auth_context=run.auth_context,
                event_type="agent_run",
                action=result.status.value,
                metadata={
                    "tools_called": ran,
                    "tools_denied": [
                        e.call.name
                        for e in result.tool_calls
                        if e.outcome is CallOutcome.DENIED
                    ],
                    "tools_pending": [call.name for call in result.pending],
                    "call_outcomes": outcomes,
                    "steps": result.steps,
                    "limit": result.limit,
                    "approvals": list(run.approvals),
                },
            )
        except AuditWriteError:
            logger.exception(
                "audit record could not be written; agent result unaffected",
                extra={"trace_id": run.trace_id},
            )

    def _check_age(self, payload: Mapping[str, Any]) -> float:
        issued_at = payload.get("issued_at")
        if isinstance(issued_at, bool) or not isinstance(issued_at, int | float):
            raise AgentStateError("This agent state carries no issue time.")
        now = time.time()
        if issued_at > now + _ISSUED_AT_SKEW:
            raise AgentStateError("This agent state was issued in the future.")
        if now - issued_at > self.state_max_age:
            raise AgentStateError(
                "This agent state has expired (older than state_max_age)."
            )
        return float(issued_at)


def _rekey(message: Message) -> tuple[Message, tuple[ToolCall, ...]]:
    """``message`` with every tool call given a fresh, unique id."""

    parts = tuple(
        ToolCallPart(dataclasses.replace(part.call, id=f"call_{secrets.token_hex(12)}"))
        if isinstance(part, ToolCallPart)
        else part
        for part in message.parts
    )
    rekeyed = Message(message.role, parts, metadata=dict(message.metadata))
    return rekeyed, rekeyed.tool_calls


def _check_pending(payload: Mapping[str, Any], digests: Mapping[str, str]) -> None:
    """Every pending id names exactly one call in the paused turn, whose digest is the
    one recorded when the run paused."""

    if not digests:
        raise AgentStateError("This agent state has no pending tool calls.")
    messages = payload.get("messages") or ()
    last = message_from_dict(messages[-1]) if messages else None
    calls = last.tool_calls if last is not None else ()
    ids = [call.id for call in calls]
    if len(set(ids)) != len(ids):
        raise AgentStateError("This agent state has tool calls sharing one id.")
    by_id = {call.id: call for call in calls}
    for cid, digest in digests.items():
        call = by_id.get(cid)
        if call is None or call_digest(call) != digest:
            raise AgentStateError(
                "This agent state's pending calls do not match their approval digests."
            )


def _refused(call: ToolCall, outcome: CallOutcome, message: str) -> ToolExecution:
    return ToolExecution(
        call,
        ExecutionResult(
            output=None,
            success=False,
            error=message,
            metadata={"tool_call_id": call.id, "tool": call.name},
        ),
        outcome,
    )


def _call_with_timeout(fn: Callable[[], Any], timeout: float | None) -> Any:
    if timeout is None:
        return fn()
    box: dict[str, Any] = {}
    ctx = contextvars.copy_context()

    def target() -> None:
        try:
            box["value"] = ctx.run(fn)
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller's thread
            box["error"] = exc

    # A daemon thread, not an executor: a tool that never returns must not hold up
    # interpreter exit, and Python offers no way to stop it.
    worker = threading.Thread(target=target, daemon=True, name="kuhaku-agent-tool")
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise _Timeout
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _token_tracked(llm: LLMProvider) -> LLMProvider:
    if isinstance(llm, TokenTrackingLLM):
        return llm
    # A provider's name is "<provider>:<model>"; the part before the colon is what
    # TokenTrackingLLM labels its metrics with.
    return TokenTrackingLLM(llm, provider=llm.name.split(":", 1)[0])


def _untracked(llm: LLMProvider) -> LLMProvider:
    return llm
