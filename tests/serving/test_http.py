"""The HTTP API through FastAPI's TestClient. Skipped without the ``http`` extra."""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from kuhaku.core.auth import ConfigAuthorizationPolicy
from kuhaku.serving import RequestAuthenticator, create_app
from tests.extraction.conftest import down

from .conftest import ENG_KEY, HR_KEY, order_extractor

GOOD = {"data": {"product": "tea", "quantity": 5}}


def _bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def client(served, keys):
    app = create_app(
        rag=served.rag,
        extractors={"order": order_extractor(GOOD)},
        auth=RequestAuthenticator(keys),
        enable_ingest=True,
    )
    return TestClient(app)


def test_healthz_needs_no_credential(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["X-Trace-Id"]


def test_ask_without_credential_is_401_with_challenge(client):
    response = client.post("/rag/ask", json={"question": "salary bands"})
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    body = response.json()
    assert body["error"]["code"] == "unauthenticated"
    assert body["trace_id"] == response.headers["X-Trace-Id"]


def test_ask_uses_the_header_identity_and_returns_trace_id(client, served):
    response = client.post("/rag/ask", json={"question": "salary bands"}, headers=_bearer(HR_KEY))
    assert response.status_code == 200
    body = response.json()
    assert body["trace_id"] == response.headers["X-Trace-Id"]
    assert "Salary bands range" in served.llm.last_user
    audit = [r for r in served.audit_records() if r["trace_id"] == body["trace_id"]]
    assert audit and audit[-1]["identity"] == "hana"


def test_roles_in_the_body_are_rejected_not_trusted(client, served):
    response = client.post(
        "/rag/ask",
        json={"question": "salary bands", "roles": ["people_ops"]},
        headers=_bearer(ENG_KEY),
    )
    assert response.status_code == 422
    assert "Salary bands range" not in (served.llm.last_user or "")


def test_validation_error_does_not_echo_the_input(client):
    secret = "my card is 4111 1111 1111 1111"
    response = client.post(
        "/rag/ask", json={"question": "", "context_text": secret, "extra": secret},
        headers=_bearer(HR_KEY),
    )
    assert response.status_code == 422
    assert secret not in response.text
    assert response.json()["error"]["code"] == "invalid_request"


def test_engineer_cannot_see_tagged_documents(client, served):
    response = client.post("/rag/ask", json={"question": "salary"}, headers=_bearer(ENG_KEY))
    assert response.status_code == 200
    assert "Salary bands range" not in served.llm.last_user


def test_optional_auth_lets_anonymous_ask_untagged_only(served, keys):
    app = create_app(rag=served.rag, auth=RequestAuthenticator(keys, required=False))
    client = TestClient(app)
    assert client.post("/rag/ask", json={"question": "salary"}).status_code == 200
    assert "Salary bands range" not in served.llm.last_user
    bad = client.post("/rag/ask", json={"question": "salary"}, headers=_bearer("wrong"))
    assert bad.status_code == 401


def test_ingest_then_ask(client, served):
    response = client.post(
        "/rag/ingest",
        json={"text": "# Holidays\n\nThe office closes in August.", "filename": "h.md",
              "access_tags": ["people_ops"]},
        headers=_bearer(HR_KEY),
    )
    assert response.status_code == 200
    assert response.json()["chunks_added"] >= 1
    client.post("/rag/ask", json={"question": "office close"}, headers=_bearer(HR_KEY))
    assert "office closes in August" in served.llm.last_user


def test_ingest_route_absent_unless_enabled(served, keys):
    client = TestClient(create_app(rag=served.rag, auth=RequestAuthenticator(keys)))
    assert client.post("/rag/ingest", json={"text": "x"}, headers=_bearer(HR_KEY)).status_code == 404


def test_policy_denial_is_403(served, keys):
    policy = ConfigAuthorizationPolicy()
    policy.set_rule("rag", "ask", ["people_ops"])
    client = TestClient(create_app(rag=served.rag, auth=RequestAuthenticator(keys), policy=policy))
    assert client.post("/rag/ask", json={"question": "q"}, headers=_bearer(ENG_KEY)).status_code == 403
    assert client.post("/rag/ask", json={"question": "q"}, headers=_bearer(HR_KEY)).status_code == 200


def test_extract_route_and_openapi_schema(client):
    response = client.post("/extract/order", json={"text": "5 boxes of tea"}, headers=_bearer(ENG_KEY))
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "complete"
    assert body["value"] == {"product": "tea", "quantity": 5.0}
    assert body["trace_id"] == response.headers["X-Trace-Id"]

    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    assert "Order" in schemas
    assert set(schemas["Order"]["properties"]) == {"product", "quantity"}


def test_unknown_extractor_is_404_in_the_common_error_shape(client):
    response = client.post("/extract/nope", json={"text": "x"}, headers=_bearer(ENG_KEY))
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    assert response.json()["trace_id"] == response.headers["X-Trace-Id"]


def test_llm_outage_is_503_without_internals(served, keys, caplog):
    app = create_app(
        extractors={"order": order_extractor(down("dial tcp 10.1.2.3:443: refused"))},
        auth=RequestAuthenticator(keys),
    )
    response = TestClient(app).post("/extract/order", json={"text": "5 tea"}, headers=_bearer(ENG_KEY))
    assert response.status_code == 503
    assert "10.1.2.3" not in response.text
    assert response.json()["error"]["code"] == "unavailable"


def test_unexpected_error_is_500_without_internals(served, keys, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("password=hunter2 at /srv/kuhaku")

    monkeypatch.setattr(served.rag, "ask", boom)
    client = TestClient(create_app(rag=served.rag, auth=RequestAuthenticator(keys)))
    response = client.post("/rag/ask", json={"question": "q"}, headers=_bearer(HR_KEY))
    assert response.status_code == 500
    assert "hunter2" not in response.text
    assert response.json()["trace_id"] == response.headers["X-Trace-Id"]


def test_request_bodies_never_reach_the_logs(client, caplog):
    secret_question = "zebra-unicorn-salary question"
    with caplog.at_level(logging.DEBUG):
        client.post("/rag/ask", json={"question": secret_question}, headers=_bearer(HR_KEY))
        client.post(
            "/rag/ingest", json={"text": "zebra-unicorn body", "filename": "z.md"},
            headers=_bearer(HR_KEY),
        )
    for record in caplog.records:
        assert "zebra-unicorn" not in record.getMessage()
        assert all("zebra-unicorn" not in str(v) for v in record.__dict__.values())


def test_no_authenticator_means_anonymous_and_warns(served, caplog):
    with caplog.at_level(logging.WARNING, logger="kuhaku.serving.http"):
        client = TestClient(create_app(rag=served.rag, enable_ingest=True))
    assert any("anonymous" in r.getMessage() for r in caplog.records)
    assert client.post("/rag/ask", json={"question": "salary"}).status_code == 200
    assert "Salary bands range" not in served.llm.last_user
    assert client.post("/rag/ingest", json={"text": "x"}).status_code == 401
