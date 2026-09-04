from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from test_features import evaluation_input

from razortrust.api import create_app
from razortrust.settings import Settings


@pytest.mark.asyncio
async def test_hold_creation_is_idempotent(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            policy_mode="local",
            ledger_path=str(tmp_path / "audit.jsonl"),
        )
    )
    request = {
        "request_id": str(uuid4()),
        "merchant_id": "merchant_001",
        "source_event_id": "settlement_001",
        "triggered_at": datetime(2026, 8, 28, 12, tzinfo=UTC).isoformat(),
        "reason_code": "CAMPAIGN",
    }
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.post("/v1/holds", json=request)
        second = await client.post("/v1/holds", json=request)
        assert first.status_code == 201
        assert second.status_code == 200
        assert first.json()["hold_id"] == second.json()["hold_id"]
        ready = await client.get("/health/ready")
        assert ready.status_code == 200
        assert len(ready.headers["X-Trace-ID"]) == 32
        frontend = await client.get("/")
        assert frontend.status_code == 200
        assert "Live Case Processing Workflow" in frontend.text
        assert "HUMAN_ONLY" in frontend.text
        assert "AUTO RELEASE OFF" in frontend.text
        holds = await client.get("/v1/holds")
        assert holds.status_code == 200
        assert len(holds.json()) == 1
        summary = await client.get("/v1/operator/summary")
        assert summary.status_code == 200
        assert summary.json()["case_count"] == 1
        assert summary.json()["research"]["stopping_decision"] == "STOP_ML"
        assert summary.json()["research"]["sealed_accessed"] is False
        assert summary.json()["risk_family_counts"] == {"CAMPAIGN": 1}
        evaluated = await client.post(
            f"/v1/holds/{first.json()['hold_id']}/evaluate",
            json=evaluation_input().model_dump(mode="json"),
        )
        assert evaluated.status_code == 200
        evidence_case = await client.get(f"/v1/holds/{first.json()['hold_id']}/evidence-case")
        assert evidence_case.status_code == 200
        assert len(evidence_case.json()["transactions"]) == 4
        assert evidence_case.json()["supporting_document"] is None


@pytest.mark.asyncio
async def test_missing_configured_model_fails_readiness(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            policy_mode="local",
            ledger_path=str(tmp_path / "audit.jsonl"),
            model_release_path=str(tmp_path / "missing-release"),
            model_public_key="invalid",
        )
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["detail"] == "configured model release is unavailable or unverified"


@pytest.mark.asyncio
async def test_production_never_falls_back_to_development_model(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            environment="production",
            policy_mode="local",
            authorization_mode="local",
            authorization_required=True,
            trusted_hosts=("test",),
            evidence_attestation_keys={"evidence-1": "invalid-until-used"},
            ledger_path=str(tmp_path / "audit.jsonl"),
        )
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["detail"] == "configured model release is unavailable or unverified"


@pytest.mark.asyncio
async def test_authenticated_merchant_is_scoped_and_cannot_view_audit(tmp_path: Path) -> None:
    token = "merchant-secret-token"
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    app = create_app(
        Settings(
            policy_mode="local",
            ledger_path=str(tmp_path / "audit.jsonl"),
            authorization_required=True,
            api_principals={
                token_hash: {
                    "id": "merchant-user-1",
                    "role": "MERCHANT",
                    "merchant_id": "merchant_001",
                }
            },
        )
    )
    payload = {
        "request_id": str(uuid4()),
        "merchant_id": "merchant_001",
        "source_event_id": "settlement_auth_001",
        "triggered_at": datetime(2026, 8, 28, 12, tzinfo=UTC).isoformat(),
        "reason_code": "CAMPAIGN",
    }
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.post("/v1/holds", json=payload)).status_code == 401
        headers = {"Authorization": f"Bearer {token}"}
        created = await client.post("/v1/holds", json=payload, headers=headers)
        assert created.status_code == 201
        hold_id = created.json()["hold_id"]
        assert (await client.get(f"/v1/holds/{hold_id}", headers=headers)).status_code == 200
        assert (await client.get(f"/v1/holds/{hold_id}/audit", headers=headers)).status_code == 403
        assert (
            await client.get(f"/v1/holds/{hold_id}/evidence-case", headers=headers)
        ).status_code == 403
        assert (await client.get("/v1/operator/summary", headers=headers)).status_code == 403

        wrong_merchant = {**payload, "request_id": str(uuid4()), "merchant_id": "merchant_002"}
        assert (
            await client.post("/v1/holds", json=wrong_merchant, headers=headers)
        ).status_code == 403
        own_holds = await client.get("/v1/holds?merchant_id=merchant_001", headers=headers)
        assert own_holds.status_code == 200
        assert len(own_holds.json()) == 1
        assert (
            await client.get("/v1/holds?merchant_id=merchant_002", headers=headers)
        ).status_code == 403


@pytest.mark.asyncio
async def test_readiness_fails_when_opa_is_unavailable(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            policy_mode="opa",
            opa_url="http://127.0.0.1:1",
            ledger_path=str(tmp_path / "audit.jsonl"),
        )
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["detail"] == "policy service is unavailable"


def test_invalid_authorization_boolean_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAZORTRUST_ENV", "production")
    monkeypatch.setenv("RAZORTRUST_AUTHORIZATION_REQUIRED", "tru")

    with pytest.raises(ValueError, match="must be 'true' or 'false'"):
        Settings.from_env()


@pytest.mark.asyncio
async def test_human_only_mode_is_ready_without_model_and_routes_to_human_review(
    tmp_path: Path,
) -> None:
    app = create_app(
        Settings(
            environment="production",
            decision_mode="human_only",
            policy_mode="local",
            authorization_mode="local",
            authorization_required=False,
            trusted_hosts=("test",),
            evidence_attestation_keys={"evidence-1": "configured-key-placeholder"},
            ledger_path=str(tmp_path / "audit.jsonl"),
        )
    )
    request = {
        "request_id": str(uuid4()),
        "merchant_id": "merchant_human_only",
        "source_event_id": "settlement_human_only",
        "triggered_at": datetime(2026, 8, 28, 12, tzinfo=UTC).isoformat(),
        "reason_code": "CAMPAIGN",
    }
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        ready = await client.get("/health/ready")
        assert ready.status_code == 200
        assert ready.json()["decision_mode"] == "human_only"
        assert ready.json()["risk_runtime"] == "human-only@1"

        created = await client.post("/v1/holds", json=request)
        assert created.status_code == 201
        hold_id = created.json()["hold_id"]
        evaluated = await client.post(
            f"/v1/holds/{hold_id}/evaluate",
            json=evaluation_input().model_dump(mode="json"),
        )
        assert evaluated.status_code == 200
        payload = evaluated.json()
        assert payload["decision"] == "ESCALATE"
        assert payload["reason_code"] == "HUMAN_ONLY"
        assert payload["model_version"] == "human-only@1"
        assert payload["probabilities"] == {
            "release": 0.0,
            "evidence_needed": 0.0,
            "escalate": 1.0,
        }
        hold = await client.get(f"/v1/holds/{hold_id}")
        assert hold.status_code == 200
        assert hold.json()["state"] == "HUMAN_REVIEW"


def test_human_only_mode_rejects_model_release_configuration(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be configured in human_only decision mode"):
        create_app(
            Settings(
                environment="production",
                decision_mode="human_only",
                policy_mode="local",
                model_release_path=str(tmp_path / "model-release"),
                model_public_key="unused",
                ledger_path=str(tmp_path / "audit.jsonl"),
            )
        )
