from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from razortrust.integrations.razorpay import RazorpayStoredEvent
from razortrust.razorpay_processor import (
    InMemoryRazorpayReconstructionStore,
    RazorpayPaymentReconstructor,
    normalize_reconciliation_item,
)


def event(event_type: str, entity_type: str, entity_id: str) -> RazorpayStoredEvent:
    now = datetime.now(UTC)
    return RazorpayStoredEvent(
        event_id=uuid4(),
        provider_event_id=f"evt_{uuid4().hex}",
        event_type=event_type,
        account_id="acc_test_merchant",
        event_created_at=now,
        payload_sha256="b" * 64,
        summary={
            "primary_entity_type": entity_type,
            "primary_entity": {"id": entity_id},
            "entity_refs": {entity_type: {"id": entity_id}},
        },
        received_at=now,
        processing_status="RECEIVED",
        processing_attempts=0,
    )


def payment_payload(
    amount_refunded: int = 0, refund_status: str | None = None
) -> dict[str, object]:
    return {
        "id": "pay_test_001",
        "amount": 10000,
        "currency": "INR",
        "status": "captured",
        "method": "upi",
        "order_id": "order_test_001",
        "captured": True,
        "amount_refunded": amount_refunded,
        "refund_status": refund_status,
        "international": False,
        "created_at": 1788200000,
        "email": "private@example.com",
        "contact": "+919999999999",
        "vpa": "private@upi",
    }


class FakeR3BClient:
    async def fetch_payment(self, payment_id: str) -> dict[str, object]:
        assert payment_id == "pay_test_001"
        return payment_payload(amount_refunded=2500, refund_status="partial")

    async def fetch_refund(self, refund_id: str) -> dict[str, object]:
        assert refund_id == "rfnd_test_001"
        return {
            "id": refund_id,
            "amount": 2500,
            "currency": "INR",
            "payment_id": "pay_test_001",
            "status": "processed",
            "speed_requested": "normal",
            "speed_processed": "normal",
            "created_at": 1788200100,
            "notes": {"customer": "must-not-persist"},
            "acquirer_data": {"arn": "sensitive"},
        }

    async def fetch_settlement(self, settlement_id: str) -> dict[str, object]:
        assert settlement_id == "setl_test_001"
        return {
            "id": settlement_id,
            "amount": 7250,
            "currency": "INR",
            "status": "processed",
            "fees": 250,
            "tax": 45,
            "utr": "UTR_TEST_001",
            "created_at": 1788200200,
        }

    async def fetch_dispute(self, dispute_id: str) -> dict[str, object]:
        assert dispute_id == "disp_test_001"
        return {
            "id": dispute_id,
            "payment_id": "pay_test_001",
            "amount": 1000,
            "currency": "INR",
            "amount_deducted": 0,
            "reason_code": "chargeback",
            "respond_by": 1788800000,
            "status": "open",
            "phase": "chargeback",
            "created_at": 1788200300,
            "evidence": {"summary": "must-not-persist"},
        }

    async def fetch_settlement_recon(
        self,
        *,
        year: int,
        month: int,
        day: int | None = None,
        count: int = 100,
        skip: int = 0,
    ) -> dict[str, object]:
        assert year == 2026 and month == 9 and day == 1
        if skip > 0:
            return {"entity": "collection", "count": 0, "items": []}
        return {
            "entity": "collection",
            "count": 1,
            "items": [
                {
                    "entity_id": "pay_test_001",
                    "type": "payment",
                    "debit": 0,
                    "credit": 9700,
                    "amount": 10000,
                    "currency": "INR",
                    "fee": 300,
                    "tax": 45,
                    "on_hold": False,
                    "settled": True,
                    "created_at": 1788200000,
                    "settled_at": 1788200400,
                    "settlement_id": "setl_test_001",
                    "payment_id": None,
                    "order_id": "order_test_001",
                    "dispute_id": None,
                    "method": "upi",
                    "settlement_utr": "UTR_TEST_001",
                    "card_network": "must-not-persist",
                }
            ],
        }


@pytest.mark.asyncio
async def test_refund_event_reconstructs_refund_and_refreshes_payment() -> None:
    store = InMemoryRazorpayReconstructionStore()
    item = event("refund.processed", "refund", "rfnd_test_001")
    store.add_event(item)
    reconstructor = RazorpayPaymentReconstructor(store, FakeR3BClient(), retry_base_seconds=0)

    await reconstructor.process_event(item.event_id)

    processed = await store.get_event(item.event_id)
    assert processed is not None and processed.processing_status == "PROCESSED"
    assert store.refunds["rfnd_test_001"].status == "processed"
    assert store.payments["pay_test_001"].amount_refunded == 2500
    assert "must-not-persist" not in str(store.refunds["rfnd_test_001"].model_dump())


@pytest.mark.asyncio
async def test_settlement_event_reconstructs_settlement() -> None:
    store = InMemoryRazorpayReconstructionStore()
    item = event("settlement.processed", "settlement", "setl_test_001")
    store.add_event(item)
    reconstructor = RazorpayPaymentReconstructor(store, FakeR3BClient(), retry_base_seconds=0)

    await reconstructor.process_event(item.event_id)

    assert store.settlements["setl_test_001"].utr == "UTR_TEST_001"
    assert (await store.get_event(item.event_id)).processing_status == "PROCESSED"


@pytest.mark.asyncio
async def test_dispute_event_reconstructs_dispute_and_refreshes_payment() -> None:
    store = InMemoryRazorpayReconstructionStore()
    item = event("payment.dispute.created", "dispute", "disp_test_001")
    store.add_event(item)
    reconstructor = RazorpayPaymentReconstructor(store, FakeR3BClient(), retry_base_seconds=0)

    await reconstructor.process_event(item.event_id)

    dispute = store.disputes["disp_test_001"]
    assert dispute.payment_id == "pay_test_001"
    assert dispute.status == "open"
    assert "must-not-persist" not in str(dispute.model_dump())
    assert (await store.get_event(item.event_id)).processing_status == "PROCESSED"


@pytest.mark.asyncio
async def test_reconciliation_sync_is_idempotent_and_privacy_minimised() -> None:
    store = InMemoryRazorpayReconstructionStore()
    reconstructor = RazorpayPaymentReconstructor(store, FakeR3BClient(), retry_base_seconds=0)

    first = await reconstructor.sync_reconciliation(
        year=2026, month=9, day=1, page_size=100, max_pages=2
    )
    second = await reconstructor.sync_reconciliation(
        year=2026, month=9, day=1, page_size=100, max_pages=2
    )

    assert first.fetched_items == 1
    assert second.fetched_items == 1
    assert len(store.reconciliation_items) == 1
    text = str(next(iter(store.reconciliation_items.values())).model_dump())
    assert "card_network" not in text
    assert "must-not-persist" not in text


def test_reconciliation_identity_is_stable() -> None:
    payload = {
        "entity_id": "pay_test_001",
        "type": "payment",
        "credit": 9700,
        "debit": 0,
        "amount": 10000,
        "currency": "INR",
        "settlement_id": "setl_test_001",
        "settled_at": 1788200400,
    }
    now = datetime.now(UTC)
    first = normalize_reconciliation_item(payload, year=2026, month=9, day=1, fetched_at=now)
    second = normalize_reconciliation_item(payload, year=2026, month=9, day=1, fetched_at=now)
    assert first.reconciliation_id == second.reconciliation_id
