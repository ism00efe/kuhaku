"""The transport-agnostic serving core: authentication from headers, authorization,
access filtering through to RAG, error mapping, wire shapes. Needs neither extra."""

from __future__ import annotations

import time

import pytest

from kuhaku.core.auth import (
    AuthContext,
    ConfigAuthorizationPolicy,
    JWTAuthProvider,
)
from kuhaku.core.exceptions import ConfigError, StoreConflict
from kuhaku.core.llm.base import LLMError, LLMUnavailableError
from kuhaku.serving import (
    AuthenticationRequired,
    Conflict,
    InvalidRequest,
    NotFound,
    PermissionDenied,
    RequestAuthenticator,
    ServiceUnavailable,
    ServingError,
    ToolService,
)
from kuhaku.serving.errors import to_serving_error
from kuhaku.serving.schemas import AskRequest, ExtractRequest, IngestRequest
from tests.extraction.conftest import down

from .conftest import ENG_KEY, HR_KEY, order_extractor

GOOD = {"data": {"product": "tea", "quantity": 5}}


def _jwt(secret: str, payload: dict) -> str:
    import base64
    import hashlib
    import hmac
    import json

    def enc(obj: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    signing_input = f"{enc({'alg': 'HS256', 'typ': 'JWT'})}.{enc(payload)}"
    sig = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{base64.urlsafe_b64encode(sig).rstrip(b'=').decode()}"


# --- RequestAuthenticator -----------------------------------------------------------


def test_bearer_api_key_yields_the_providers_identity(keys):
    auth = RequestAuthenticator(keys)
    ctx = auth.authenticate({"authorization": f"Bearer {HR_KEY}"})
    assert ctx.identity == "hana"
    assert ctx.roles == ("people_ops",)
    assert ctx.is_authenticated


def test_missing_credential_is_rejected_when_required(keys):
    with pytest.raises(AuthenticationRequired):
        RequestAuthenticator(keys).authenticate({})


def test_missing_credential_is_anonymous_when_optional(keys):
    ctx = RequestAuthenticator(keys, required=False).authenticate({})
    assert ctx == AuthContext.anonymous()


@pytest.mark.parametrize(
    "header", ["Bearer nope", "Basic " + HR_KEY, HR_KEY, "Bearer "]
)
def test_a_bad_credential_is_rejected_even_when_optional(keys, header):
    auth = RequestAuthenticator(keys, required=False)
    with pytest.raises(AuthenticationRequired):
        auth.authenticate({"Authorization": header})


def test_custom_header_without_scheme(keys):
    auth = RequestAuthenticator(keys, header="X-API-Key", scheme=None)
    assert auth.authenticate({"x-api-key": ENG_KEY}).identity == "efe"
    assert auth.challenge == "ApiKey"


def test_jwt_provider_through_the_authenticator():
    provider = JWTAuthProvider("s3cret")
    auth = RequestAuthenticator(provider)
    token = _jwt("s3cret", {"sub": "ada", "roles": ["people_ops"], "exp": time.time() + 60})
    assert auth.authenticate({"Authorization": f"Bearer {token}"}).roles == ("people_ops",)
    forged = _jwt("wrong", {"sub": "ada", "roles": ["admin"]})
    with pytest.raises(AuthenticationRequired):
        auth.authenticate({"Authorization": f"Bearer {forged}"})


def test_a_provider_that_raises_is_a_401_not_a_500():
    class Exploding:
        def authenticate(self, credential):
            raise RuntimeError("boom")

    with pytest.raises(AuthenticationRequired):
        RequestAuthenticator(Exploding()).authenticate({"Authorization": "Bearer x"})


@pytest.mark.parametrize("token", ["é.é.é", "a.b.c", "....", "x" * 50])
def test_hostile_jwt_through_the_authenticator_is_a_401(token):
    auth = RequestAuthenticator(JWTAuthProvider("s3cret"))
    with pytest.raises(AuthenticationRequired):
        auth.authenticate({"Authorization": f"Bearer {token}"})


# --- ToolService: access filtering and audit ------------------------------------------


def _hr() -> AuthContext:
    return AuthContext(identity="hana", is_authenticated=True, roles=("people_ops",))


def _eng() -> AuthContext:
    return AuthContext(identity="efe", is_authenticated=True, roles=("engineering",))


def test_tagged_documents_follow_the_callers_roles(served):
    service = ToolService(rag=served.rag)

    eng = service.ask(_eng(), "what are the salary bands")
    assert all(c.title != "Salary Policy" for c in eng.citations)
    assert "Salary bands range" not in (served.llm.last_user or "")

    hr = service.ask(_hr(), "what are the salary bands")
    assert "Salary bands range" in (served.llm.last_user or "")
    assert hr.trace_id


def test_anonymous_sees_only_untagged_documents(served):
    ToolService(rag=served.rag).ask(AuthContext.anonymous(), "salary bands")
    assert "Salary bands range" not in (served.llm.last_user or "")
    assert "Run deploy.sh" in (served.llm.last_user or "")


def test_the_audit_record_carries_the_authenticated_identity(served):
    response = ToolService(rag=served.rag).ask(_hr(), "salary bands")
    records = [r for r in served.audit_records() if r["trace_id"] == response.trace_id]
    assert records
    assert records[-1]["identity"] == "hana"
    assert records[-1]["roles"] == ["people_ops"]
    assert records[-1]["is_authenticated"] is True


def test_ask_response_omits_server_paths(served):
    payload = ToolService(rag=served.rag).ask(_hr(), "salary").model_dump()
    assert payload["citations"]
    assert all("source_path" not in c for c in payload["citations"])


# --- ToolService: authorization ------------------------------------------------------


def _policy() -> ConfigAuthorizationPolicy:
    policy = ConfigAuthorizationPolicy()
    policy.set_rule("rag", "ask", ["*"])
    policy.set_rule("rag", "ingest", ["people_ops"])
    policy.set_rule("extractor:order", "extract", ["engineering"])
    return policy


def test_policy_denial_is_forbidden_for_an_identity(served):
    service = ToolService(
        rag=served.rag, extractors={"order": order_extractor(GOOD)}, policy=_policy(),
        enable_ingest=True,
    )
    with pytest.raises(PermissionDenied):
        service.ingest(_eng(), "text", "x.md")
    with pytest.raises(PermissionDenied):
        service.extract("order", _hr(), "5 tea")
    assert service.extract("order", _eng(), "5 tea")["status"] == "complete"


def test_policy_denial_for_anonymous_asks_for_authentication(served):
    service = ToolService(extractors={"order": order_extractor()}, policy=_policy())
    with pytest.raises(AuthenticationRequired):
        service.extract("order", AuthContext.anonymous(), "5 tea")


def test_a_crashing_policy_denies(served):
    class Broken:
        def check(self, context, resource, action):
            raise RuntimeError("boom")

    with pytest.raises(ServingError) as info:
        ToolService(rag=served.rag, policy=Broken()).ask(_hr(), "q")
    assert info.value.status_code == 500


def test_ingest_is_off_by_default(served):
    with pytest.raises(NotFound):
        ToolService(rag=served.rag).ingest(_hr(), "text", "x.md")


def test_anonymous_can_never_ingest_even_with_an_allow_all_policy(served):
    policy = ConfigAuthorizationPolicy()
    policy.set_rule("rag", "ingest", ["*"])
    service = ToolService(rag=served.rag, policy=policy, enable_ingest=True)
    with pytest.raises(AuthenticationRequired):
        service.ingest(AuthContext.anonymous(), "text", "x.md")


def test_ingested_document_is_visible_to_the_next_ask(served):
    service = ToolService(rag=served.rag, enable_ingest=True)
    result = service.ingest(_hr(), "# Holidays\n\nThe office closes in August.", "holidays.md",
                            ["people_ops"])
    assert result.chunks_added >= 1
    service.ask(_hr(), "when does the office close")
    assert "office closes in August" in (served.llm.last_user or "")
    service.ask(_eng(), "when does the office close")
    assert "office closes in August" not in (served.llm.last_user or "")


def test_ingest_user_facing_errors_are_invalid_requests(served):
    service = ToolService(rag=served.rag, enable_ingest=True)
    with pytest.raises(InvalidRequest) as info:
        service.ingest(_hr(), "some text", "x.md", ["  "])
    assert "tag" in info.value.message


# --- ToolService: extraction ------------------------------------------------------------


def test_extraction_payload_shape():
    service = ToolService(extractors=[order_extractor(GOOD)])
    payload = service.extract("Order", AuthContext.anonymous(), "5 boxes of tea")
    assert payload["status"] == "complete"
    assert payload["value"] == {"product": "tea", "quantity": 5.0}
    assert payload["trace_id"]
    assert "metadata" not in payload


def test_needs_clarification_is_a_result_not_an_error():
    extractor = order_extractor({"data": {"product": "tea", "quantity": None}})
    payload = ToolService(extractors={"order": extractor}).extract(
        "order", AuthContext.anonymous(), "tea please"
    )
    assert payload["status"] == "needs_clarification"
    assert payload["missing"] == ["/quantity"]
    assert payload["value"] is None


def test_unknown_extractor_is_not_found():
    with pytest.raises(NotFound):
        ToolService(extractors={"order": order_extractor()}).extract(
            "nope", AuthContext.anonymous(), "x"
        )


def test_llm_outage_is_unavailable_without_the_providers_message():
    extractor = order_extractor(down("connect to 10.0.0.7:11434 refused"))
    with pytest.raises(ServiceUnavailable) as info:
        ToolService(extractors={"order": extractor}).extract(
            "order", AuthContext.anonymous(), "5 tea"
        )
    assert "10.0.0.7" not in info.value.message


@pytest.mark.parametrize(
    "name", ["has space", "a/b", "", "x" * 65]
)
def test_extractor_names_must_be_url_and_tool_safe(name):
    with pytest.raises(ConfigError):
        ToolService(extractors={name: order_extractor()})


def test_nothing_to_serve_is_a_config_error():
    with pytest.raises(ConfigError):
        ToolService()
    with pytest.raises(ConfigError):
        ToolService(extractors={"order": order_extractor()}, enable_ingest=True)


# --- error mapping and wire schemas ---------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (LLMUnavailableError("secret-host:1234 down"), ServiceUnavailable),
        (StoreConflict("/srv/data locked by pid 42"), Conflict),
        (RuntimeError("/etc/passwd"), ServingError),
        (ValueError("internal detail"), ServingError),
    ],
)
def test_error_mapping_never_leaks_exception_text(exc, expected):
    mapped = to_serving_error(exc)
    assert type(mapped) is expected
    assert str(exc) not in mapped.message


def test_plain_llm_error_is_internal_not_unavailable():
    # RAGEngine/ExtractStep translate provider failures to LLMUnavailableError at their
    # boundary; anything else reaching here is unexpected.
    assert type(to_serving_error(LLMError("x"))) is ServingError


@pytest.mark.parametrize("model", [AskRequest, IngestRequest, ExtractRequest])
def test_request_bodies_cannot_carry_identity(model):
    from pydantic import ValidationError

    base = {"question": "q", "text": "t"}
    fields = {k: v for k, v in base.items() if k in model.model_fields}
    with pytest.raises(ValidationError):
        model.model_validate({**fields, "roles": ["admin"]})
    with pytest.raises(ValidationError):
        model.model_validate({**fields, "auth_context": {"identity": "root"}})
