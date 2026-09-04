from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from razortrust.checkout_bridge import derive_checkout_lifecycle
from razortrust.integrations.razorpay import RazorpayStoredEvent
from razortrust.razorpay_processor import (
    InMemoryRazorpayReconstructionStore,
    RazorpayPaymentReconstructor,
)


def test_checkout_lifecycle_matches_razorpay_order_and_payment_semantics() -> None:
    assert derive_checkout_lifecycle(payment_status=None, order_status="created") == "OPEN"
    assert (
        derive_checkout_lifecycle(payment_status="failed", order_status="attempted")
        == "ATTEMPT_FAILED"
    )
    assert (
        derive_checkout_lifecycle(payment_status="authorized", order_status="attempted")
        == "AUTHORIZED"
    )
    assert derive_checkout_lifecycle(payment_status="captured", order_status="attempted") == "PAID"
    assert derive_checkout_lifecycle(payment_status="refunded", order_status="paid") == "PAID"


class _FakePaymentClient:
    async def fetch_payment(self, payment_id: str) -> dict[str, object]:
        return {
            "id": payment_id,
            "order_id": "order_lifecycle_1",
            "amount": 1000,
            "currency": "INR",
            "status": "failed",
            "method": "upi",
            "captured": False,
            "amount_refunded": 0,
            "refund_status": None,
            "international": False,
            "created_at": 1788330000,
        }

    async def fetch_refund(self, refund_id: str):  # pragma: no cover - protocol only
        raise NotImplementedError

    async def fetch_settlement(self, settlement_id: str):  # pragma: no cover
        raise NotImplementedError

    async def fetch_dispute(self, dispute_id: str):  # pragma: no cover
        raise NotImplementedError

    async def fetch_settlement_recon(self, **kwargs):  # pragma: no cover
        raise NotImplementedError


class _Observer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def observe_payment_event(self, **kwargs) -> None:
        self.calls.append(kwargs)


@pytest.mark.asyncio
async def test_payment_failed_event_is_forwarded_to_checkout_lifecycle() -> None:
    store = InMemoryRazorpayReconstructionStore()
    event = RazorpayStoredEvent(
        event_id=uuid4(),
        provider_event_id="evt_lifecycle_failed_1",
        event_type="payment.failed",
        account_id="acc_test",
        event_created_at=datetime(2026, 9, 2, 8, tzinfo=UTC),
        payload_sha256="a" * 64,
        summary={
            "primary_entity_type": "payment",
            "primary_entity": {"id": "pay_lifecycle_1"},
        },
        received_at=datetime(2026, 9, 2, 8, 0, 1, tzinfo=UTC),
        processing_status="RECEIVED",
        processing_attempts=0,
    )
    store.add_event(event)
    observer = _Observer()
    reconstructor = RazorpayPaymentReconstructor(
        store,
        _FakePaymentClient(),
        checkout_lifecycle=observer,
        retry_base_seconds=0,
    )

    await reconstructor.process_event(event.event_id)

    assert len(observer.calls) == 1
    call = observer.calls[0]
    assert call["event_type"] == "payment.failed"
    assert call["order_id"] == "order_lifecycle_1"
    assert call["payment_status"] == "failed"
    processed = await store.get_event(event.event_id)
    assert processed is not None
    assert processed.processing_status == "PROCESSED"
