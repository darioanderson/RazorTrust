from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from razortrust.api import create_app
from razortrust.database import Base
from razortrust.integrations.razorpay import (
    RazorpayWebhookEnvelope,
    build_event_summary,
    payload_sha256,
    verify_webhook_signature,
)
from razortrust.razorpay_store import SqlRazorpayEventStore
from razortrust.settings import Settings

WEBHOOK_SECRET = "local-test-webhook-secret"


def webhook_body() -> bytes:
    payload = {
        "entity": "event",
        "account_id": "acc_test_merchant",
        "event": "payment.captured",
        "contains": ["payment"],
        "created_at": 1788200000,
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_test_001",
                    "entity": "payment",
                    "amount": 45218000,
                    "currency": "INR",
                    "status": "captured",
                    "method": "upi",
                    "email": "must-not-be-persisted@example.com",
                    "contact": "+919999999999",
                    "vpa": "must-not-be-persisted@exampleupi",
                    "card": {"id": "card_sensitive"},
                }
            }
        },
    }
    return json.dumps(payload, separators=(",", ":")).encode()


def sign(body: bytes, secret: str = WEBHOOK_SECRET) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


@pytest.mark.asyncio
async def test_valid_webhook_is_verified_minimised_and_idempotent(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            policy_mode="local",
            ledger_path=str(tmp_path / "audit.jsonl"),
            razorpay_enabled=True,
            razorpay_webhook_secret=WEBHOOK_SECRET,
        )
    )
    body = webhook_body()
    headers = {
        "content-type": "application/json",
        "X-Razorpay-Signature": sign(body),
        "x-razorpay-event-id": "evt_test_001",
    }
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.post(
            "/v1/integrations/razorpay/webhook", content=body, headers=headers
        )
        second = await client.post(
            "/v1/integrations/razorpay/webhook", content=body, headers=headers
        )
        status = await client.get("/v1/integrations/razorpay/status")

    assert first.status_code == 200
    assert first.json()["duplicate"] is False
    assert second.status_code == 200
    assert second.json()["duplicate"] is True
    assert status.status_code == 200
    assert status.json()["total_events"] == 1
    assert status.json()["last_event_type"] == "payment.captured"

    envelope = RazorpayWebhookEnvelope.model_validate_json(body)
    summary_text = json.dumps(build_event_summary(envelope))
    assert "must-not-be-persisted" not in summary_text
    assert "card_sensitive" not in summary_text


@pytest.mark.asyncio
async def test_webhook_rejects_invalid_signature_before_accepting_payload(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            policy_mode="local",
            ledger_path=str(tmp_path / "audit.jsonl"),
            razorpay_enabled=True,
            razorpay_webhook_secret=WEBHOOK_SECRET,
        )
    )
    body = webhook_body()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/integrations/razorpay/webhook",
            content=body,
            headers={
                "content-type": "application/json",
                "X-Razorpay-Signature": "0" * 64,
                "x-razorpay-event-id": "evt_bad_signature",
            },
        )
        status_response = await client.get("/v1/integrations/razorpay/status")

    assert response.status_code == 401
    assert status_response.json()["total_events"] == 0


@pytest.mark.asyncio
async def test_enabled_integration_without_secret_fails_closed(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            policy_mode="local",
            ledger_path=str(tmp_path / "audit.jsonl"),
            razorpay_enabled=True,
        )
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        ready = await client.get("/health/ready")
        webhook = await client.post("/v1/integrations/razorpay/webhook", content=b"{}")

    assert ready.status_code == 503
    assert ready.json()["detail"] == "Razorpay webhook secret is not configured"
    assert webhook.status_code == 503


def test_webhook_signature_uses_raw_body_exactly() -> None:
    compact = b'{"event":"payment.captured"}'
    spaced = b'{ "event": "payment.captured" }'
    signature = sign(compact)
    assert verify_webhook_signature(compact, signature, WEBHOOK_SECRET) is True
    assert verify_webhook_signature(spaced, signature, WEBHOOK_SECRET) is False


@pytest.mark.asyncio
async def test_sql_event_store_deduplicates_provider_event_ids() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        store = SqlRazorpayEventStore(async_sessionmaker(engine, expire_on_commit=False))
        body = webhook_body()
        envelope = RazorpayWebhookEnvelope.model_validate_json(body)
        kwargs = {
            "provider_event_id": "evt_sql_001",
            "envelope": envelope,
            "payload_sha256": payload_sha256(body),
            "summary": build_event_summary(envelope),
            "received_at": datetime.now(UTC),
        }
        first, created = await store.store(**kwargs)
        second, duplicate_created = await store.store(**kwargs)
        stats = await store.stats()

        assert created is True
        assert duplicate_created is False
        assert first.event_id == second.event_id
        assert stats.total_events == 1
    finally:
        await engine.dispose()
