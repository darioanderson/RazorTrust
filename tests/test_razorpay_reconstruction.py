from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from razortrust.integrations.razorpay import (
    RazorpayApiError,
    RazorpayStoredEvent,
    RazorpayWebhookEnvelope,
    build_event_summary,
)
from razortrust.razorpay_processor import (
    InMemoryRazorpayReconstructionStore,
    RazorpayPaymentReconstructor,
    extract_payment_id,
    normalize_payment,
)


class FakePaymentClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[str] = []

    async def fetch_payment(self, payment_id: str) -> dict[str, object]:
        self.calls.append(payment_id)
        return dict(self.payload)


class PermanentFailurePaymentClient:
    def __init__(self) -> None:
        self.calls = 0

    async def fetch_payment(self, payment_id: str) -> dict[str, object]:
        self.calls += 1
        raise RazorpayApiError("unauthorized", status_code=401, retryable=False)


class FailingPaymentClient:
    def __init__(self) -> None:
        self.calls = 0

    async def fetch_payment(self, payment_id: str) -> dict[str, object]:
        self.calls += 1
        raise RuntimeError("provider unavailable")


def stored_event(
    *, summary: dict[str, object], event_type: str = "payment.captured"
) -> RazorpayStoredEvent:
    now = datetime.now(UTC)
    return RazorpayStoredEvent(
        event_id=uuid4(),
        provider_event_id=f"evt_{uuid4().hex}",
        event_type=event_type,
        account_id="acc_test_merchant",
        event_created_at=now,
        payload_sha256="a" * 64,
        summary=summary,
        received_at=now,
        processing_status="RECEIVED",
        processing_attempts=0,
    )


def authoritative_payment() -> dict[str, object]:
    return {
        "id": "pay_test_001",
        "amount": 1000,
        "currency": "INR",
        "status": "captured",
        "method": "wallet",
        "order_id": "order_test_001",
        "captured": True,
        "amount_refunded": 0,
        "refund_status": None,
        "international": False,
        "created_at": 1788200000,
        "email": "must-not-be-stored@example.com",
        "contact": "+919999999999",
        "vpa": "sensitive@upi",
        "card": {"id": "card_sensitive"},
    }


def test_event_summary_preserves_only_safe_entity_references() -> None:
    envelope = RazorpayWebhookEnvelope.model_validate(
        {
            "entity": "event",
            "account_id": "acc_test_merchant",
            "event": "order.paid",
            "contains": ["payment", "order"],
            "created_at": 1788200000,
            "payload": {
                "payment": {
                    "entity": {
                        "id": "pay_test_001",
                        "order_id": "order_test_001",
                        "email": "sensitive@example.com",
                    }
                },
                "order": {"entity": {"id": "order_test_001"}},
            },
        }
    )
    summary = build_event_summary(envelope)
    assert summary["entity_refs"] == {
        "payment": {"id": "pay_test_001", "order_id": "order_test_001"},
        "order": {"id": "order_test_001"},
    }
    assert "sensitive@example.com" not in str(summary)


def test_extract_payment_id_supports_refund_relationship() -> None:
    event = stored_event(
        event_type="refund.processed",
        summary={
            "primary_entity_type": "refund",
            "primary_entity": {"id": "rfnd_1", "payment_id": "pay_test_001"},
        },
    )
    assert extract_payment_id(event) == "pay_test_001"


def test_normalized_payment_excludes_customer_and_instrument_data() -> None:
    event = stored_event(
        summary={
            "primary_entity_type": "payment",
            "primary_entity": {"id": "pay_test_001"},
        }
    )
    payment = normalize_payment(authoritative_payment(), event)
    dumped = payment.model_dump()
    text = str(dumped)
    assert payment.amount == 1000
    assert payment.status == "captured"
    assert "must-not-be-stored" not in text
    assert "sensitive@upi" not in text
    assert "card_sensitive" not in text


@pytest.mark.asyncio
async def test_reconstructor_enriches_and_processes_event() -> None:
    store = InMemoryRazorpayReconstructionStore()
    event = stored_event(
        summary={
            "primary_entity_type": "payment",
            "primary_entity": {"id": "pay_test_001"},
        }
    )
    store.add_event(event)
    client = FakePaymentClient(authoritative_payment())
    reconstructor = RazorpayPaymentReconstructor(store, client, retry_base_seconds=0)

    await reconstructor.process_event(event.event_id)

    processed = await store.get_event(event.event_id)
    stats = await store.processing_stats()
    assert processed is not None
    assert processed.processing_status == "PROCESSED"
    assert client.calls == ["pay_test_001"]
    assert stats.processed_events == 1
    assert stats.normalized_payments == 1
    assert store.payments["pay_test_001"].status == "captured"


@pytest.mark.asyncio
async def test_reconstructor_is_idempotent_across_multiple_events_for_same_payment() -> None:
    store = InMemoryRazorpayReconstructionStore()
    events = [
        stored_event(
            event_type=event_type,
            summary={
                "primary_entity_type": "payment",
                "primary_entity": {"id": "pay_test_001"},
            },
        )
        for event_type in ("payment.authorized", "order.paid", "payment.captured")
    ]
    for event in events:
        store.add_event(event)
    reconstructor = RazorpayPaymentReconstructor(
        store, FakePaymentClient(authoritative_payment()), retry_base_seconds=0
    )

    await reconstructor.process_pending(limit=10)

    stats = await store.processing_stats()
    assert stats.processed_events == 3
    assert stats.normalized_payments == 1


@pytest.mark.asyncio
async def test_reconstructor_fails_closed_after_bounded_retries() -> None:
    store = InMemoryRazorpayReconstructionStore()
    event = stored_event(
        summary={
            "primary_entity_type": "payment",
            "primary_entity": {"id": "pay_test_001"},
        }
    )
    store.add_event(event)
    client = FailingPaymentClient()
    reconstructor = RazorpayPaymentReconstructor(
        store, client, max_attempts=3, retry_base_seconds=0, retry_jitter_seconds=0
    )

    await reconstructor.process_event(event.event_id)

    failed = await store.get_event(event.event_id)
    assert failed is not None
    assert failed.processing_status == "FAILED"
    assert client.calls == 3
    assert "pay_test_001" not in store.payments


@pytest.mark.asyncio
async def test_non_payment_event_is_explicitly_skipped_not_left_pending() -> None:
    store = InMemoryRazorpayReconstructionStore()
    event = stored_event(
        event_type="invoice.paid",
        summary={
            "primary_entity_type": "invoice",
            "primary_entity": {"id": "inv_test_001"},
        },
    )
    store.add_event(event)
    reconstructor = RazorpayPaymentReconstructor(
        store, FakePaymentClient(authoritative_payment()), retry_base_seconds=0
    )

    await reconstructor.process_event(event.event_id)

    skipped = await store.get_event(event.event_id)
    assert skipped is not None
    assert skipped.processing_status == "SKIPPED"
    assert (await store.processing_stats()).skipped_events == 1


@pytest.mark.asyncio
async def test_permanent_api_failure_is_not_retried() -> None:
    store = InMemoryRazorpayReconstructionStore()
    event = stored_event(
        summary={
            "primary_entity_type": "payment",
            "primary_entity": {"id": "pay_test_001"},
        }
    )
    store.add_event(event)
    client = PermanentFailurePaymentClient()
    reconstructor = RazorpayPaymentReconstructor(
        store, client, max_attempts=3, retry_base_seconds=0, retry_jitter_seconds=0
    )

    await reconstructor.process_event(event.event_id)

    failed = await store.get_event(event.event_id)
    assert failed is not None
    assert failed.processing_status == "FAILED"
    assert client.calls == 1


@pytest.mark.asyncio
async def test_retry_budget_is_persistently_bounded_after_restart() -> None:
    store = InMemoryRazorpayReconstructionStore()
    event = stored_event(
        summary={
            "primary_entity_type": "payment",
            "primary_entity": {"id": "pay_test_001"},
        }
    ).model_copy(update={"processing_status": "RETRY", "processing_attempts": 3})
    store.add_event(event)
    client = FakePaymentClient(authoritative_payment())
    reconstructor = RazorpayPaymentReconstructor(
        store, client, max_attempts=3, retry_base_seconds=0, retry_jitter_seconds=0
    )

    await reconstructor.process_event(event.event_id)

    failed = await store.get_event(event.event_id)
    assert failed is not None
    assert failed.processing_status == "FAILED"
    assert client.calls == []
