"""The serving layer's error vocabulary, and the one place a tool's exception becomes one.

Every transport (HTTP, MCP) reports a failure as a :class:`ServingError`: a status code,
a stable machine-readable ``code``, and a public message written for the caller. The
original exception is chained for logs but its text never reaches the response --
provider errors carry hostnames, file paths and sometimes keys in their messages.
"""

from __future__ import annotations

import logging

from kuhaku.core.exceptions import (
    CapabilityUnavailable,
    ConfigError,
    KuhakuError,
    StoreConflict,
)
from kuhaku.core.llm.base import LLMUnavailableError
from kuhaku.tools.rag.vectorstore import VectorStoreError

logger = logging.getLogger(__name__)


class ServingError(KuhakuError):
    """A request failed in a way the caller may be told about.

    ``message`` is shown to the caller verbatim, so it must never be built from an
    exception's text unless that text was itself written for end users.
    """

    status_code = 500
    code = "internal_error"
    default_message = "The request could not be completed."

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.default_message
        super().__init__(self.message)


class AuthenticationRequired(ServingError):
    status_code = 401
    code = "unauthenticated"
    default_message = "A valid credential is required."


class PermissionDenied(ServingError):
    status_code = 403
    code = "forbidden"
    default_message = "The authenticated identity may not perform this action."


class NotFound(ServingError):
    status_code = 404
    code = "not_found"
    default_message = "No such tool."


class Conflict(ServingError):
    status_code = 409
    code = "conflict"
    default_message = "Another writer is using the store; retry later."


class InvalidRequest(ServingError):
    status_code = 422
    code = "invalid_request"
    default_message = "The request is invalid."


class ServiceUnavailable(ServingError):
    status_code = 503
    code = "unavailable"
    default_message = "A backend this tool depends on is unavailable; retry later."


def to_serving_error(exc: BaseException, *, user_facing_value_errors: bool = False) -> ServingError:
    """Map ``exc`` to the :class:`ServingError` a transport reports.

    ``user_facing_value_errors`` is set only around calls whose ``ValueError`` messages
    are written for end users (ingestion's ``EngineMessages``, the blank-tag check); a
    ``ValueError`` from anywhere else is an internal fault and stays a 500.
    """

    if isinstance(exc, ServingError):
        return exc
    if isinstance(exc, LLMUnavailableError):
        return ServiceUnavailable("The language model is unavailable; retry later.")
    if isinstance(exc, StoreConflict):
        return Conflict()
    if isinstance(exc, (CapabilityUnavailable, VectorStoreError)):
        return ServiceUnavailable()
    if isinstance(exc, ConfigError):
        # Raised by Extractor.extract for a request it cannot honour (e.g. prior= with
        # the continuation step disabled); the message names the parameter, nothing else.
        return InvalidRequest(str(exc))
    if user_facing_value_errors and isinstance(exc, ValueError):
        return InvalidRequest(str(exc))
    # Anything else is a fault whose details are internal; the traceback goes to the
    # log under the request's trace id.
    logger.error("request failed with an unexpected error", exc_info=exc)
    return ServingError()
