from __future__ import annotations

import asyncio
import hashlib
import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
from uuid import UUID

from pydantic import BaseModel, Field

from .integrations.razorpay import RazorpayApiError, RazorpayStoredEvent


class NormalizedProcessorPayment(BaseModel):
    """Privacy-minimised authoritative payment state used by the risk pipeline."""

    payment_id: str = Field(min_length=1, max_length=128)
    account_id: str | None = Field(default=None, max_length=128)
    order_id: str | None = Field(default=None, max_length=128)
    amount: int = Field(ge=0)
    currency: str = Field(min_length=3, max_length=8)
    status: str = Field(min_length=1, max_length=32)
    method: str | None = Field(default=None, max_length=64)
    captured: bool = False
    amount_refunded: int = Field(default=0, ge=0)
    refund_status: str | None = Field(default=None, max_length=32)
    international: bool | None = None
    provider_created_at: datetime | None = None
    source_event_id: str = Field(min_length=1, max_length=128)
    source_event_created_at: datetime | None = None
    authoritative_sha256: str = Field(min_length=64, max_length=64)
    enriched_at: datetime


class NormalizedProcessorRefund(BaseModel):
    refund_id: str = Field(min_length=1, max_length=128)
    account_id: str | None = Field(default=None, max_length=128)
    payment_id: str = Field(min_length=1, max_length=128)
    amount: int = Field(ge=0)
    currency: str = Field(min_length=3, max_length=8)
    status: str = Field(min_length=1, max_length=32)
    speed_requested: str | None = Field(default=None, max_length=32)
    speed_processed: str | None = Field(default=None, max_length=32)
    provider_created_at: datetime | None = None
    source_event_id: str = Field(min_length=1, max_length=128)
    source_event_created_at: datetime | None = None
    authoritative_sha256: str = Field(min_length=64, max_length=64)
    enriched_at: datetime


class NormalizedProcessorSettlement(BaseModel):
    settlement_id: str = Field(min_length=1, max_length=128)
    account_id: str | None = Field(default=None, max_length=128)
    amount: int = Field(ge=0)
    currency: str | None = Field(default=None, min_length=3, max_length=8)
    status: str = Field(min_length=1, max_length=32)
    fees: int = Field(default=0, ge=0)
    tax: int = Field(default=0, ge=0)
    utr: str | None = Field(default=None, max_length=128)
    provider_created_at: datetime | None = None
    source_event_id: str = Field(min_length=1, max_length=128)
    source_event_created_at: datetime | None = None
    authoritative_sha256: str = Field(min_length=64, max_length=64)
    enriched_at: datetime


class NormalizedProcessorDispute(BaseModel):
    dispute_id: str = Field(min_length=1, max_length=128)
    account_id: str | None = Field(default=None, max_length=128)
    payment_id: str = Field(min_length=1, max_length=128)
    amount: int = Field(ge=0)
    currency: str = Field(min_length=3, max_length=8)
    amount_deducted: int = Field(default=0, ge=0)
    reason_code: str | None = Field(default=None, max_length=128)
    status: str = Field(min_length=1, max_length=32)
    phase: str | None = Field(default=None, max_length=64)
    respond_by: datetime | None = None
    provider_created_at: datetime | None = None
    source_event_id: str = Field(min_length=1, max_length=128)
    source_event_created_at: datetime | None = None
    authoritative_sha256: str = Field(min_length=64, max_length=64)
    enriched_at: datetime


class NormalizedReconciliationItem(BaseModel):
    reconciliation_id: str = Field(min_length=64, max_length=64)
    entity_id: str = Field(min_length=1, max_length=128)
    entity_type: str = Field(min_length=1, max_length=32)
    debit: int = Field(default=0, ge=0)
    credit: int = Field(default=0, ge=0)
    amount: int = Field(default=0, ge=0)
    currency: str = Field(min_length=3, max_length=8)
    fee: int = Field(default=0, ge=0)
    tax: int = Field(default=0, ge=0)
    on_hold: bool | None = None
    settled: bool | None = None
    provider_created_at: datetime | None = None
    settled_at: datetime | None = None
    settlement_id: str | None = Field(default=None, max_length=128)
    payment_id: str | None = Field(default=None, max_length=128)
    order_id: str | None = Field(default=None, max_length=128)
    dispute_id: str | None = Field(default=None, max_length=128)
    method: str | None = Field(default=None, max_length=64)
    settlement_utr: str | None = Field(default=None, max_length=128)
    fetched_year: int
    fetched_month: int
    fetched_day: int | None = None
    authoritative_sha256: str = Field(min_length=64, max_length=64)
    fetched_at: datetime


class RazorpayProcessingStats(BaseModel):
    pending_events: int = 0
    processing_events: int = 0
    processed_events: int = 0
    retry_events: int = 0
    failed_events: int = 0
    skipped_events: int = 0
    normalized_payments: int = 0
    normalized_refunds: int = 0
    normalized_settlements: int = 0
    normalized_disputes: int = 0
    reconciliation_items: int = 0
    last_processed_at: datetime | None = None


class ReconciliationSyncResult(BaseModel):
    fetched_items: int = 0
    stored_items: int = 0
    pages: int = 0


class RazorpayReconstructionStore(Protocol):
    async def healthcheck(self) -> None: ...
    async def get_event(self, event_id: UUID) -> RazorpayStoredEvent | None: ...
    async def claim_event(self, event_id: UUID) -> RazorpayStoredEvent | None: ...
    async def complete_payment_event(
        self, event_id: UUID, payment: NormalizedProcessorPayment
    ) -> None: ...
    async def complete_refund_event(
        self,
        event_id: UUID,
        refund: NormalizedProcessorRefund,
        payment: NormalizedProcessorPayment,
    ) -> None: ...
    async def complete_settlement_event(
        self, event_id: UUID, settlement: NormalizedProcessorSettlement
    ) -> None: ...
    async def complete_dispute_event(
        self,
        event_id: UUID,
        dispute: NormalizedProcessorDispute,
        payment: NormalizedProcessorPayment,
    ) -> None: ...
    async def upsert_reconciliation_items(
        self, items: list[NormalizedReconciliationItem]
    ) -> int: ...
    async def mark_skipped(self, event_id: UUID, reason: str) -> None: ...
    async def mark_retry(self, event_id: UUID, error: str) -> None: ...
    async def mark_failed(self, event_id: UUID, error: str) -> None: ...
    async def pending_event_ids(self, *, limit: int) -> list[UUID]: ...
    async def recover_stale_processing(self, *, stale_before: datetime) -> int: ...
    async def processing_stats(self) -> RazorpayProcessingStats: ...


class InMemoryRazorpayReconstructionStore:
    """Development/test store for R3 reconstruction mechanics."""

    def __init__(self) -> None:
        self.events: dict[UUID, RazorpayStoredEvent] = {}
        self.payments: dict[str, NormalizedProcessorPayment] = {}
        self.refunds: dict[str, NormalizedProcessorRefund] = {}
        self.settlements: dict[str, NormalizedProcessorSettlement] = {}
        self.disputes: dict[str, NormalizedProcessorDispute] = {}
        self.reconciliation_items: dict[str, NormalizedReconciliationItem] = {}
        self.last_errors: dict[UUID, str] = {}
        self.processed_at: dict[UUID, datetime] = {}

    async def healthcheck(self) -> None:
        return None

    def add_event(self, event: RazorpayStoredEvent) -> None:
        self.events[event.event_id] = event.model_copy(deep=True)

    async def get_event(self, event_id: UUID) -> RazorpayStoredEvent | None:
        event = self.events.get(event_id)
        return event.model_copy(deep=True) if event is not None else None

    async def claim_event(self, event_id: UUID) -> RazorpayStoredEvent | None:
        event = self.events.get(event_id)
        if event is None or event.processing_status not in {"RECEIVED", "RETRY"}:
            return None
        event = event.model_copy(
            update={
                "processing_status": "PROCESSING",
                "processing_attempts": event.processing_attempts + 1,
            }
        )
        self.events[event_id] = event
        return event.model_copy(deep=True)

    async def complete_payment_event(
        self, event_id: UUID, payment: NormalizedProcessorPayment
    ) -> None:
        self.payments[payment.payment_id] = payment.model_copy(deep=True)
        self._mark(event_id, "PROCESSED")

    async def complete_refund_event(
        self,
        event_id: UUID,
        refund: NormalizedProcessorRefund,
        payment: NormalizedProcessorPayment,
    ) -> None:
        self.refunds[refund.refund_id] = refund.model_copy(deep=True)
        self.payments[payment.payment_id] = payment.model_copy(deep=True)
        self._mark(event_id, "PROCESSED")

    async def complete_settlement_event(
        self, event_id: UUID, settlement: NormalizedProcessorSettlement
    ) -> None:
        self.settlements[settlement.settlement_id] = settlement.model_copy(deep=True)
        self._mark(event_id, "PROCESSED")

    async def complete_dispute_event(
        self,
        event_id: UUID,
        dispute: NormalizedProcessorDispute,
        payment: NormalizedProcessorPayment,
    ) -> None:
        self.disputes[dispute.dispute_id] = dispute.model_copy(deep=True)
        self.payments[payment.payment_id] = payment.model_copy(deep=True)
        self._mark(event_id, "PROCESSED")

    async def upsert_reconciliation_items(self, items: list[NormalizedReconciliationItem]) -> int:
        for item in items:
            self.reconciliation_items[item.reconciliation_id] = item.model_copy(deep=True)
        return len(items)

    async def mark_skipped(self, event_id: UUID, reason: str) -> None:
        self.last_errors[event_id] = reason
        self._mark(event_id, "SKIPPED")

    async def mark_retry(self, event_id: UUID, error: str) -> None:
        self.last_errors[event_id] = error
        self._mark(event_id, "RETRY", processed=False)

    async def mark_failed(self, event_id: UUID, error: str) -> None:
        self.last_errors[event_id] = error
        self._mark(event_id, "FAILED")

    async def pending_event_ids(self, *, limit: int) -> list[UUID]:
        return [
            event_id
            for event_id, event in self.events.items()
            if event.processing_status in {"RECEIVED", "RETRY"}
        ][:limit]

    async def recover_stale_processing(self, *, stale_before: datetime) -> int:
        return 0

    async def processing_stats(self) -> RazorpayProcessingStats:
        statuses = [event.processing_status for event in self.events.values()]
        last_processed_at = max(self.processed_at.values(), default=None)
        return RazorpayProcessingStats(
            pending_events=statuses.count("RECEIVED"),
            processing_events=statuses.count("PROCESSING"),
            processed_events=statuses.count("PROCESSED"),
            retry_events=statuses.count("RETRY"),
            failed_events=statuses.count("FAILED"),
            skipped_events=statuses.count("SKIPPED"),
            normalized_payments=len(self.payments),
            normalized_refunds=len(self.refunds),
            normalized_settlements=len(self.settlements),
            normalized_disputes=len(self.disputes),
            reconciliation_items=len(self.reconciliation_items),
            last_processed_at=last_processed_at,
        )

    def _mark(self, event_id: UUID, status: str, *, processed: bool = True) -> None:
        event = self.events[event_id]
        self.events[event_id] = event.model_copy(update={"processing_status": status})
        if processed:
            self.processed_at[event_id] = datetime.now(UTC)


PAYMENT_RECONSTRUCTION_EVENTS = {
    "payment.authorized",
    "payment.captured",
    "payment.failed",
    "payment.refunded",
    "order.paid",
}
REFUND_RECONSTRUCTION_EVENTS = {
    "refund.created",
    "refund.processed",
    "refund.failed",
    "refund.speed_changed",
}
SETTLEMENT_RECONSTRUCTION_EVENTS = {"settlement.processed"}
DISPUTE_RECONSTRUCTION_EVENTS = {
    "payment.dispute.created",
    "payment.dispute.won",
    "payment.dispute.lost",
    "payment.dispute.closed",
    "payment.dispute.under_review",
    "payment.dispute.action_required",
}


class CheckoutLifecycleObserver(Protocol):
    async def observe_payment_event(
        self,
        *,
        event_type: str,
        source_event_id: str,
        order_id: str | None,
        payment_id: str,
        payment_status: str,
        captured: bool,
        observed_at: datetime,
    ) -> None: ...


class RazorpayPaymentClient(Protocol):
    async def fetch_payment(self, payment_id: str) -> dict[str, Any]: ...
    async def fetch_refund(self, refund_id: str) -> dict[str, Any]: ...
    async def fetch_settlement(self, settlement_id: str) -> dict[str, Any]: ...
    async def fetch_dispute(self, dispute_id: str) -> dict[str, Any]: ...
    async def fetch_settlement_recon(
        self,
        *,
        year: int,
        month: int,
        day: int | None = None,
        count: int = 100,
        skip: int = 0,
    ) -> dict[str, Any]: ...


@dataclass(slots=True)
class RazorpayPaymentReconstructor:
    """Authoritative multi-entity Razorpay reconstruction engine.

    The class name is preserved for compatibility with R3A imports. R3B routes
    payments, refunds, settlements and disputes to entity-specific normalizers.
    """

    store: RazorpayReconstructionStore
    client: RazorpayPaymentClient
    checkout_lifecycle: CheckoutLifecycleObserver | None = None
    max_attempts: int = 3
    retry_base_seconds: float = 0.5
    retry_jitter_seconds: float = 0.25
    stale_claim_seconds: int = 120

    async def process_event(self, event_id: UUID) -> None:
        while True:
            event = await self.store.claim_event(event_id)
            if event is None:
                return
            if event.processing_attempts > self.max_attempts:
                await self.store.mark_failed(event_id, "retry_budget_exhausted")
                return

            try:
                await self._process_claimed_event(event)
                return
            except RazorpayApiError as exc:
                safe_error = (
                    f"RazorpayApiError:{exc.status_code}"
                    if exc.status_code is not None
                    else "RazorpayApiError"
                )
                if not exc.retryable or event.processing_attempts >= self.max_attempts:
                    await self.store.mark_failed(event_id, safe_error)
                    return
            except (TypeError, ValueError) as exc:
                await self.store.mark_failed(event_id, type(exc).__name__)
                return
            except Exception as exc:
                safe_error = type(exc).__name__
                if event.processing_attempts >= self.max_attempts:
                    await self.store.mark_failed(event_id, safe_error)
                    return

            await self.store.mark_retry(event_id, safe_error)
            delay = self.retry_base_seconds * (2 ** (event.processing_attempts - 1))
            if self.retry_jitter_seconds > 0:
                delay += random.uniform(0.0, self.retry_jitter_seconds)
            await asyncio.sleep(delay)

    async def _process_claimed_event(self, event: RazorpayStoredEvent) -> None:
        if event.event_type in PAYMENT_RECONSTRUCTION_EVENTS:
            payment_id = extract_payment_id(event)
            if payment_id is None:
                await self.store.mark_skipped(event.event_id, "no_payment_reference")
                return
            payload = await self.client.fetch_payment(payment_id)
            payment = normalize_payment(payload, event)
            if self.checkout_lifecycle is not None:
                await self.checkout_lifecycle.observe_payment_event(
                    event_type=event.event_type,
                    source_event_id=event.provider_event_id,
                    order_id=payment.order_id,
                    payment_id=payment.payment_id,
                    payment_status=payment.status,
                    captured=payment.captured,
                    observed_at=payment.enriched_at,
                )
            await self.store.complete_payment_event(event.event_id, payment)
            return

        if event.event_type in REFUND_RECONSTRUCTION_EVENTS:
            refund_id = extract_refund_id(event)
            if refund_id is None:
                await self.store.mark_skipped(event.event_id, "no_refund_reference")
                return
            refund_payload = await self.client.fetch_refund(refund_id)
            payment_id = _required_text(refund_payload, "payment_id", prefix="pay_")
            payment_payload = await self.client.fetch_payment(payment_id)
            payment = normalize_payment(payment_payload, event)
            refund = normalize_refund(refund_payload, event, payment_currency=payment.currency)
            await self.store.complete_refund_event(event.event_id, refund, payment)
            return

        if event.event_type in SETTLEMENT_RECONSTRUCTION_EVENTS:
            settlement_id = extract_settlement_id(event)
            if settlement_id is None:
                await self.store.mark_skipped(event.event_id, "no_settlement_reference")
                return
            payload = await self.client.fetch_settlement(settlement_id)
            settlement = normalize_settlement(payload, event)
            await self.store.complete_settlement_event(event.event_id, settlement)
            return

        if event.event_type in DISPUTE_RECONSTRUCTION_EVENTS:
            dispute_id = extract_dispute_id(event)
            if dispute_id is None:
                await self.store.mark_skipped(event.event_id, "no_dispute_reference")
                return
            dispute_payload = await self.client.fetch_dispute(dispute_id)
            payment_id = _required_text(dispute_payload, "payment_id", prefix="pay_")
            payment_payload = await self.client.fetch_payment(payment_id)
            payment = normalize_payment(payment_payload, event)
            dispute = normalize_dispute(dispute_payload, event)
            await self.store.complete_dispute_event(event.event_id, dispute, payment)
            return

        await self.store.mark_skipped(event.event_id, "unsupported_r3b_event_type")

    async def process_pending(self, *, limit: int = 100) -> int:
        stale_before = datetime.now(UTC) - timedelta(seconds=self.stale_claim_seconds)
        await self.store.recover_stale_processing(stale_before=stale_before)
        event_ids = await self.store.pending_event_ids(limit=limit)
        for event_id in event_ids:
            await self.process_event(event_id)
        return len(event_ids)

    async def sync_reconciliation(
        self,
        *,
        year: int,
        month: int,
        day: int | None = None,
        page_size: int = 200,
        max_pages: int = 20,
    ) -> ReconciliationSyncResult:
        if not 1 <= page_size <= 1000:
            raise ValueError("page_size must be between 1 and 1000")
        if not 1 <= max_pages <= 100:
            raise ValueError("max_pages must be between 1 and 100")
        fetched = 0
        stored = 0
        pages = 0
        skip = 0
        for _ in range(max_pages):
            payload = await self.client.fetch_settlement_recon(
                year=year,
                month=month,
                day=day,
                count=page_size,
                skip=skip,
            )
            raw_items = payload.get("items", [])
            if not isinstance(raw_items, list):
                raise ValueError("Razorpay reconciliation items must be a list")
            if not raw_items:
                break
            now = datetime.now(UTC)
            items = [
                normalize_reconciliation_item(
                    item,
                    year=year,
                    month=month,
                    day=day,
                    fetched_at=now,
                )
                for item in raw_items
            ]
            stored += await self.store.upsert_reconciliation_items(items)
            fetched += len(items)
            pages += 1
            skip += len(raw_items)
            if len(raw_items) < page_size:
                break
        return ReconciliationSyncResult(
            fetched_items=fetched,
            stored_items=stored,
            pages=pages,
        )


def extract_payment_id(event: RazorpayStoredEvent) -> str | None:
    refs = event.summary.get("entity_refs")
    if isinstance(refs, dict):
        payment = refs.get("payment")
        if isinstance(payment, dict):
            identifier = payment.get("id")
            if isinstance(identifier, str) and identifier.startswith("pay_"):
                return identifier
    primary = event.summary.get("primary_entity")
    if isinstance(primary, dict):
        if event.summary.get("primary_entity_type") == "payment":
            identifier = primary.get("id")
            if isinstance(identifier, str) and identifier.startswith("pay_"):
                return identifier
        related = primary.get("payment_id")
        if isinstance(related, str) and related.startswith("pay_"):
            return related
    return None


def extract_refund_id(event: RazorpayStoredEvent) -> str | None:
    return _extract_entity_id(event, "refund", "rfnd_")


def extract_settlement_id(event: RazorpayStoredEvent) -> str | None:
    return _extract_entity_id(event, "settlement", "setl_")


def extract_dispute_id(event: RazorpayStoredEvent) -> str | None:
    return _extract_entity_id(event, "dispute", "disp_")


def _extract_entity_id(event: RazorpayStoredEvent, entity_type: str, prefix: str) -> str | None:
    refs = event.summary.get("entity_refs")
    if isinstance(refs, dict):
        entity = refs.get(entity_type)
        if isinstance(entity, dict):
            identifier = entity.get("id")
            if isinstance(identifier, str) and identifier.startswith(prefix):
                return identifier
    primary = event.summary.get("primary_entity")
    if isinstance(primary, dict) and event.summary.get("primary_entity_type") == entity_type:
        identifier = primary.get("id")
        if isinstance(identifier, str) and identifier.startswith(prefix):
            return identifier
    return None


def normalize_payment(
    payload: dict[str, Any], event: RazorpayStoredEvent
) -> NormalizedProcessorPayment:
    payment_id = _required_text(payload, "id", prefix="pay_")
    amount = _non_negative_int(payload.get("amount"), default=-1)
    if amount < 0:
        raise ValueError("Razorpay payment amount is invalid")
    currency = _required_text(payload, "currency").upper()
    status = _required_text(payload, "status")
    created_at = _provider_datetime(payload.get("created_at"))
    safe_payload = {
        key: payload.get(key)
        for key in (
            "id",
            "amount",
            "currency",
            "status",
            "method",
            "order_id",
            "captured",
            "amount_refunded",
            "refund_status",
            "international",
            "created_at",
        )
    }
    return NormalizedProcessorPayment(
        payment_id=payment_id,
        account_id=event.account_id,
        order_id=_optional_text(payload.get("order_id")),
        amount=amount,
        currency=currency,
        status=status,
        method=_optional_text(payload.get("method")),
        captured=bool(payload.get("captured", status == "captured")),
        amount_refunded=_non_negative_int(payload.get("amount_refunded"), default=0),
        refund_status=_optional_text(payload.get("refund_status")),
        international=payload.get("international")
        if isinstance(payload.get("international"), bool)
        else None,
        provider_created_at=created_at,
        source_event_id=event.provider_event_id,
        source_event_created_at=event.event_created_at,
        authoritative_sha256=_hash_safe(safe_payload),
        enriched_at=datetime.now(UTC),
    )


def normalize_refund(
    payload: dict[str, Any],
    event: RazorpayStoredEvent,
    *,
    payment_currency: str,
) -> NormalizedProcessorRefund:
    refund_id = _required_text(payload, "id", prefix="rfnd_")
    payment_id = _required_text(payload, "payment_id", prefix="pay_")
    amount = _non_negative_int(payload.get("amount"), default=-1)
    if amount < 0:
        raise ValueError("Razorpay refund amount is invalid")
    raw_currency = payload.get("currency")
    currency = (
        raw_currency.upper()
        if isinstance(raw_currency, str) and len(raw_currency) >= 3
        else payment_currency.upper()
    )
    status = _required_text(payload, "status")
    safe_payload = {
        key: payload.get(key)
        for key in (
            "id",
            "amount",
            "currency",
            "payment_id",
            "created_at",
            "status",
            "speed_processed",
            "speed_requested",
        )
    }
    return NormalizedProcessorRefund(
        refund_id=refund_id,
        account_id=event.account_id,
        payment_id=payment_id,
        amount=amount,
        currency=currency,
        status=status,
        speed_requested=_optional_text(payload.get("speed_requested")),
        speed_processed=_optional_text(payload.get("speed_processed")),
        provider_created_at=_provider_datetime(payload.get("created_at")),
        source_event_id=event.provider_event_id,
        source_event_created_at=event.event_created_at,
        authoritative_sha256=_hash_safe(safe_payload),
        enriched_at=datetime.now(UTC),
    )


def normalize_settlement(
    payload: dict[str, Any], event: RazorpayStoredEvent
) -> NormalizedProcessorSettlement:
    settlement_id = _required_text(payload, "id", prefix="setl_")
    amount = _non_negative_int(payload.get("amount"), default=-1)
    if amount < 0:
        raise ValueError("Razorpay settlement amount is invalid")
    status = _required_text(payload, "status")
    raw_currency = payload.get("currency")
    currency = (
        raw_currency.upper() if isinstance(raw_currency, str) and len(raw_currency) >= 3 else None
    )
    safe_payload = {
        key: payload.get(key)
        for key in ("id", "amount", "currency", "status", "fees", "tax", "utr", "created_at")
    }
    return NormalizedProcessorSettlement(
        settlement_id=settlement_id,
        account_id=event.account_id,
        amount=amount,
        currency=currency,
        status=status,
        fees=_non_negative_int(payload.get("fees"), default=0),
        tax=_non_negative_int(payload.get("tax"), default=0),
        utr=_optional_text(payload.get("utr")),
        provider_created_at=_provider_datetime(payload.get("created_at")),
        source_event_id=event.provider_event_id,
        source_event_created_at=event.event_created_at,
        authoritative_sha256=_hash_safe(safe_payload),
        enriched_at=datetime.now(UTC),
    )


def normalize_dispute(
    payload: dict[str, Any], event: RazorpayStoredEvent
) -> NormalizedProcessorDispute:
    dispute_id = _required_text(payload, "id", prefix="disp_")
    payment_id = _required_text(payload, "payment_id", prefix="pay_")
    amount = _non_negative_int(payload.get("amount"), default=-1)
    if amount < 0:
        raise ValueError("Razorpay dispute amount is invalid")
    currency = _required_text(payload, "currency").upper()
    status = _required_text(payload, "status")
    safe_payload = {
        key: payload.get(key)
        for key in (
            "id",
            "payment_id",
            "amount",
            "currency",
            "amount_deducted",
            "reason_code",
            "respond_by",
            "status",
            "phase",
            "created_at",
        )
    }
    return NormalizedProcessorDispute(
        dispute_id=dispute_id,
        account_id=event.account_id,
        payment_id=payment_id,
        amount=amount,
        currency=currency,
        amount_deducted=_non_negative_int(payload.get("amount_deducted"), default=0),
        reason_code=_optional_text(payload.get("reason_code")),
        status=status,
        phase=_optional_text(payload.get("phase")),
        respond_by=_provider_datetime(payload.get("respond_by")),
        provider_created_at=_provider_datetime(payload.get("created_at")),
        source_event_id=event.provider_event_id,
        source_event_created_at=event.event_created_at,
        authoritative_sha256=_hash_safe(safe_payload),
        enriched_at=datetime.now(UTC),
    )


def normalize_reconciliation_item(
    payload: Any,
    *,
    year: int,
    month: int,
    day: int | None,
    fetched_at: datetime,
) -> NormalizedReconciliationItem:
    if not isinstance(payload, dict):
        raise ValueError("Razorpay reconciliation item must be an object")
    entity_id = _required_text(payload, "entity_id")
    entity_type = _required_text(payload, "type")
    currency = _required_text(payload, "currency").upper()
    safe_payload = {
        key: payload.get(key)
        for key in (
            "entity_id",
            "type",
            "debit",
            "credit",
            "amount",
            "currency",
            "fee",
            "tax",
            "on_hold",
            "settled",
            "created_at",
            "settled_at",
            "settlement_id",
            "payment_id",
            "order_id",
            "dispute_id",
            "method",
            "settlement_utr",
        )
    }
    authoritative_sha256 = _hash_safe(safe_payload)
    stable_key = {
        "entity_id": entity_id,
        "type": entity_type,
        "settlement_id": payload.get("settlement_id"),
        "settled_at": payload.get("settled_at"),
        "debit": payload.get("debit"),
        "credit": payload.get("credit"),
    }
    return NormalizedReconciliationItem(
        reconciliation_id=_hash_safe(stable_key),
        entity_id=entity_id,
        entity_type=entity_type,
        debit=_non_negative_int(payload.get("debit"), default=0),
        credit=_non_negative_int(payload.get("credit"), default=0),
        amount=_non_negative_int(payload.get("amount"), default=0),
        currency=currency,
        fee=_non_negative_int(payload.get("fee"), default=0),
        tax=_non_negative_int(payload.get("tax"), default=0),
        on_hold=payload.get("on_hold") if isinstance(payload.get("on_hold"), bool) else None,
        settled=payload.get("settled") if isinstance(payload.get("settled"), bool) else None,
        provider_created_at=_provider_datetime(payload.get("created_at")),
        settled_at=_provider_datetime(payload.get("settled_at")),
        settlement_id=_optional_text(payload.get("settlement_id")),
        payment_id=_optional_text(payload.get("payment_id")),
        order_id=_optional_text(payload.get("order_id")),
        dispute_id=_optional_text(payload.get("dispute_id")),
        method=_optional_text(payload.get("method")),
        settlement_utr=_optional_text(payload.get("settlement_utr")),
        fetched_year=year,
        fetched_month=month,
        fetched_day=day,
        authoritative_sha256=authoritative_sha256,
        fetched_at=fetched_at,
    )


def _required_text(payload: dict[str, Any], key: str, *, prefix: str | None = None) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValueError(f"Razorpay {key} is invalid")
    if prefix is not None and not value.startswith(prefix):
        raise ValueError(f"Razorpay {key} has an unexpected format")
    return value


def _optional_text(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or len(value) > 128:
        raise ValueError("Razorpay text field is invalid")
    return value


def _non_negative_int(value: Any, *, default: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("Razorpay numeric field is invalid")
    return value


def _provider_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("Razorpay timestamp is invalid")
    return datetime.fromtimestamp(value, tz=UTC)


def _hash_safe(payload: dict[str, Any]) -> str:
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
