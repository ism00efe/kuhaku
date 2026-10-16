"""Choosing which schema a text belongs to, before extracting it.

``Extractor([order, complaint])`` -- an :class:`Extractor` given a list of extractors --
routes through the :class:`Router` here. It holds the extractors and a chain of
routes. Each route looks at the text and returns a schema name or ``None`` ("not my
call"); the first name wins. Routes are plain callables, so a cheap keyword rule can
sit in front of the LLM route, the LLM route can be dropped entirely, or replaced.

Choosing a schema is itself an extraction -- of one field whose allowed values are the
schema names -- so :class:`LLMRoute` runs an :class:`Extractor` rather than a prompt
of its own, and gets the retry loop, fallback providers, caller instructions and
source fencing from it.

The schema choice runs with the same :class:`ExtractionSettings` as the extractions --
by default the first extractor's -- so masking and the input guard switched on there
apply to the router's own model call too. A text the guard refuses while routing comes
back ``BLOCKED``.

When no route names a schema the result is ``NO_MATCH``. The router never falls back
to a default schema: which way an unclassifiable text should go is a domain decision
(fatura sent it to "order"), so it belongs to the caller.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from kuhaku.core.exceptions import ConfigError, CustomComponentError, KuhakuError
from kuhaku.core.llm import with_runtime
from kuhaku.core.llm.base import LLMError, LLMProvider
from kuhaku.core.models import Message
from kuhaku.core.observability import instrumented_step, traced_call
from kuhaku.core.sanitization import sanitize
from kuhaku.core.security import record_audit
from kuhaku.evaluation.sample import EvaluationSample

from .config import ExtractionSettings
from .extractor import Extractor, evaluation_sample
from .prompts import ExtractionPrompts, as_prompts, fill
from .result import ExtractionResult, ExtractionStatus, Issue
from .schema import ExtractionSchema

logger = logging.getLogger(__name__)

CHOICE_FIELD = "choice"

# Where a result keeps the data the other extractors left behind (see Router.extract).
EARLIER_KEY = "earlier"


class InputBlocked(KuhakuError):
    """A route's own model call was refused by the input guard. ``issues`` holds why.
    :meth:`Router.extract` turns it into a ``BLOCKED`` result; it reaches the
    caller only from :meth:`Extractor.route`."""

    def __init__(self, issues: Sequence[Issue]) -> None:
        self.issues = tuple(issues)
        super().__init__("; ".join(issue.message for issue in self.issues))


@dataclass(frozen=True)
class RouteRequest:
    text: str
    schemas: tuple[ExtractionSchema[Any], ...]
    llm: LLMProvider
    history: Sequence[Message] = ()
    reference: Any = None
    fallback_llms: Sequence[LLMProvider] = ()
    extraction_settings: ExtractionSettings | None = None


Route = Callable[[RouteRequest], "str | None"]
"""``request -> schema name, or None to defer to the next route``."""


class LLMRoute:
    """Asks the LLM which schema fits, as an extraction of a single ``choice`` field.

    The field only accepts the registered schema names or null, so a reply naming
    anything else -- an unknown name, a list, a number -- is a schema violation the
    retry loop sends back to the model, then to ``request.fallback_llms``. A choice
    that is still unusable after that defers (``None``) and is logged; a provider
    failure on every provider raises :class:`LLMUnavailableError`.

    ``instructions`` are the caller's routing rules ("a message with only a quantity
    continues the previous order"), added after the framework's rules. ``prompts``
    replaces default prompt texts, as for :class:`Extractor`; its ``route`` text is
    the one listing the schemas.
    """

    def __init__(
        self,
        *,
        instructions: str = "",
        max_retries: int | None = None,
        prompts: ExtractionPrompts | Mapping[str, str] | None = None,
    ) -> None:
        self.instructions = instructions
        self.max_retries = max_retries
        self.prompts = as_prompts(prompts)

    def __call__(self, request: RouteRequest) -> str | None:
        if not request.text.strip():
            return None  # an empty text fits no schema; no model call needed to know
        names =tuple(schema.name for schema in request.schemas)
        catalogue = "\n".join(
            f"- {schema.name}: {schema.description or '(no description)'}"
            for schema in request.schemas
        )
        rules = fill(self.prompts.route, schemas=catalogue)
        if self.instructions:
            rules = f"{rules}\n\n{self.instructions}"
        chooser: Extractor[Any] = Extractor(
            {CHOICE_FIELD: Literal[names] | None},  # type: ignore[valid-type]
            name="route",
            description="Which of the listed schemas the source text belongs to.",
            instructions=rules,
            llm=request.llm,
            fallback_llms=request.fallback_llms,
            max_retries=self.max_retries,
            prompts=self.prompts,
            enable_token_tracking=False,  # the router's providers are tracked already
            # The router audits its own outcomes; the chooser's call is part of them.
            extraction_settings=(request.extraction_settings or ExtractionSettings())
            .model_copy(update={"audit_log_path": None}),
        )
        result = chooser.extract(request.text, history=request.history, reference=request.reference)
        if result.status is ExtractionStatus.BLOCKED:
            raise InputBlocked(result.issues)
        if result.status is not ExtractionStatus.COMPLETE or result.value is None:
            # Deferring is right for the result, but an NO_MATCH caused by an unusable
            # reply must be told apart from a text that genuinely fits no schema.
            logger.warning(
                "llm route reply unusable, deferring",
                extra={"issues": [issue.message for issue in result.issues]},
            )
            return None
        choice: str | None = getattr(result.value, CHOICE_FIELD)
        return choice


class Router:
    """Routes a text to one of several extractors, then extracts with it. Built by
    :class:`Extractor` when it is given a list of extractors; not used directly.

    ``routes`` defaults to ``[LLMRoute()]``. It is a plain list on the instance --
    ``router.routes.insert(0, keyword_rule)`` puts a rule in front of the LLM,
    ``router.routes.remove(...)`` drops one. ``llm`` and ``fallback_llms`` are what
    routes receive; when ``llm`` is omitted both default to the first extractor's.

    ``extraction_settings`` defaults to the first extractor's, so the schema choice
    behaves like the extractions -- see the module docstring. ``audit_log_path`` (or the
    settings' one) audits the outcomes the router ends itself (``BLOCKED``,
    ``NO_MATCH``); an extraction it hands on is audited by that extractor.
    """

    def __init__(
        self,
        extractors: Sequence[Extractor[Any]],
        *,
        routes: Sequence[Route] | None = None,
        llm: LLMProvider | None = None,
        fallback_llms: Sequence[LLMProvider] = (),
        audit_log_path: str | None = None,
        extraction_settings: ExtractionSettings | None = None,
    ) -> None:
        if not extractors:
            raise ConfigError("Routing needs at least one Extractor.")
        self.extraction_settings = extraction_settings or extractors[0].extraction_settings
        if audit_log_path is None:
            audit_log_path = self.extraction_settings.audit_log_path
        self.audit_log_path = audit_log_path
        self.extractors: dict[str, Extractor[Any]] = {}
        for extractor in extractors:
            name = extractor.schema.name
            if name in self.extractors:
                raise ConfigError(
                    f"Two extractors use the schema name '{name}'; routes pick by name, "
                    f"so names must be unique. Pass ExtractionSchema(..., name=...)."
                )
            self.extractors[name] = extractor
        self.routes: list[Route] = list(routes) if routes is not None else [LLMRoute()]
        if llm is None:
            self.llm: LLMProvider = extractors[0].llm
            self.fallback_llms: list[LLMProvider] = list(
                fallback_llms or extractors[0].fallback_llms
            )
        else:
            self.llm = with_runtime(llm)
            self.fallback_llms = [with_runtime(fallback) for fallback in fallback_llms]

    def route(
        self, text: str, *, history: Sequence[Message] = (), reference: Any = None
    ) -> str | None:
        if not isinstance(text, str):
            # Checked here, not left to a route: a route would report it as its own bug.
            raise TypeError(f"text must be a str, got {type(text).__name__}.")
        request = RouteRequest(
            text=text,
            schemas=tuple(e.schema for e in self.extractors.values()),
            llm=self.llm,
            history=tuple(history),
            reference=reference,
            fallback_llms=tuple(self.fallback_llms),
            extraction_settings=self.extraction_settings,
        )
        with instrumented_step("extraction.route", logger) as rec:
            for route in self.routes:
                name = _call_route(route, request)
                if name is None:
                    continue
                if not isinstance(name, str) or name not in self.extractors:
                    raise CustomComponentError(
                        f"Route {route!r} returned {name!r}, which is not a registered "
                        f"schema ({sorted(self.extractors)})."
                    )
                rec.set(schema=name, route=_route_name(route))
                return name
            rec.set(schema=None)
            return None

    def extract(
        self,
        text: str,
        *,
        history: Sequence[Message] = (),
        reference: Any = None,
        prior: ExtractionResult[Any] | None = None,
        trace_id: str | None = None,
    ) -> ExtractionResult[Any]:
        """Route, then extract. ``prior`` is the previous result, whatever it was: the
        data each extractor left behind travels in results under
        ``metadata["earlier"]``, so an order interrupted by a complaint or a greeting is
        picked up again when the conversation comes back to it, and only the data of the
        extractor the text goes to is carried in. That needs the schema names, which is
        why ``prior`` here is a result and not a bare ``data`` dict as
        :meth:`Extractor.extract` also accepts.

        Routing and extraction share one ``trace_id``."""

        if prior is not None and not isinstance(prior, ExtractionResult):
            raise TypeError(
                f"prior must be the ExtractionResult of the earlier turn, got "
                f"{type(prior).__name__}: it is carried over only when the text "
                f"routes to the schema it came from, which a bare dict does not name."
            )
        # What each extractor left behind so far, by name; kuhaku keeps no state, so it
        # rides along in the results the caller hands back.
        earlier: dict[str, dict[str, Any]] = {}
        if prior is not None:
            earlier.update(prior.metadata.get(EARLIER_KEY) or {})
            if prior.schema_name and prior.data is not None:
                earlier[prior.schema_name] = prior.data
        with traced_call("extraction.router", logger, trace_id=trace_id) as (tid, _rec):
            try:
                name = self.route(text, history=history, reference=reference)
            except InputBlocked as blocked:
                self._audit(text, tid, ExtractionStatus.BLOCKED)
                return ExtractionResult(
                    status=ExtractionStatus.BLOCKED,
                    schema_name="",
                    issues=blocked.issues,
                    trace_id=tid,
                    metadata={EARLIER_KEY: earlier},
                )
            if name is None:
                self._audit(text, tid, ExtractionStatus.NO_MATCH)
                return ExtractionResult(
                    status=ExtractionStatus.NO_MATCH,
                    schema_name="",
                    trace_id=tid,
                    metadata={EARLIER_KEY: earlier},
                )
            carried = earlier.pop(name, None)
            result = self.extractors[name].extract(
                text, history=history, reference=reference, prior=carried
            )
            return replace(result, metadata={**result.metadata, EARLIER_KEY: earlier})

    def _audit(self, text: str, trace_id: str, status: ExtractionStatus) -> None:
        if self.audit_log_path is None:
            return
        record_audit(
            self.audit_log_path,
            trace_id=trace_id,
            raw_question=text,
            sanitized_retrieval_query=sanitize(text),
            event_type="extraction.route",
            metadata={"status": status.value},
        )

    def evaluate_sample(self, query: str) -> EvaluationSample:
        """Implements kuhaku's ``EvaluationTarget``. Answers with
        ``{"schema": <chosen schema or null>, "data": <extracted data>}`` so one
        ``golden_answer`` can check the routing (``/schema``) and the values
        (``/data/...``) together."""

        result = self.extract(query)
        answer = {"schema": result.schema_name or None, "data": result.data}
        return evaluation_sample(query, result, answer_data=answer)


def _call_route(route: Route, request: RouteRequest) -> object:
    try:
        return route(request)
    except (KuhakuError, LLMError):
        raise
    except Exception as exc:
        # A caller's route that crashes is a bug in that route; naming it saves the
        # caller from reading the router's internals to find which one.
        raise CustomComponentError(f"Route {_route_name(route)} raised: {exc}") from exc


def _route_name(route: Route) -> str:
    return getattr(route, "__name__", None) or type(route).__name__
