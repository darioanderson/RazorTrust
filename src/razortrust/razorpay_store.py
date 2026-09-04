from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Integer,
    String,
    UniqueConstraint,
    Uuid,
    func,
    select,
    update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base
from .integrations.razorpay import (
    RazorpayIngestionStats,
    RazorpayStoredEvent,
    RazorpayWebhookEnvelope,
)
from .razorpay_processor import (
    NormalizedProcessorDispute,
    NormalizedProcessorPayment,
    NormalizedProcessorRefund,
    NormalizedProcessorSettlement,
    NormalizedReconciliationItem,
    RazorpayProcessingStats,
)


class ProcessorWebhookEventRow(Base):
    __tablename__ = "processor_webhook_events"
    __table_args__ = (
        UniqueConstraint("provider", "provider_event_id", name="uq_processor_provider_event"),
    )

    event_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    provider_event_id: Mapped[str] = mapped_column(String(128), nullable=False)
    event_type: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    account_id: Mapped[str | None] = mapped_column(String(128), index=True)
    event_created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    payload_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    summary: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    processing_status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    processing_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processing_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    last_error: Mapped[str | None] = mapped_column(String(128))


class ProcessorPaymentRow(Base):
    __tablename__ = "processor_payments"

    payment_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False, default="RAZORPAY")
    account_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    order_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    amount: Mapped[int] = mapped_column(BigInteger, nullable=False)
    currency: Mapped[str] = mapped_column(String(8), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    method: Mapped[str | None] = mapped_column(String(64))
    captured: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    amount_refunded: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    refund_status: Mapped[str | None] = mapped_column(String(32))
    international: Mapped[bool | None] = mapped_column(Boolean)
    provider_created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), index=True
    )
    source_event_id: Mapped[str] = mapped_column(String(128), nullable=False)
    source_event_created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    authoritative_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    enriched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ProcessorPaymentObservationRow(Base):
    __tablename__ = "processor_payment_observations"
    __table_args__ = (
        UniqueConstraint("source_event_id", name="uq_payment_observation_source_event"),
    )

    observation_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    payment_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    account_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    order_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    amount: Mapped[int] = mapped_column(BigInteger, nullable=False)
    currency: Mapped[str] = mapped_column(String(8), nullable=False)
    method: Mapped[str | None] = mapped_column(String(64))
    captured: Mapped[bool] = mapped_column(Boolean, nullable=False)
    amount_refunded: Mapped[int] = mapped_column(BigInteger, nullable=False)
    refund_status: Mapped[str | None] = mapped_column(String(32))
    international: Mapped[bool | None] = mapped_column(Boolean)
    provider_created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_event_id: Mapped[str] = mapped_column(String(128), nullable=False)
    source_event_created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    authoritative_sha256: Mapped[str] = mapped_column(String(64), nullable=False)


class ProcessorRefundRow(Base):
    __tablename__ = "processor_refunds"

    refund_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False, default="RAZORPAY")
    account_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    payment_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    amount: Mapped[int] = mapped_column(BigInteger, nullable=False)
    currency: Mapped[str] = mapped_column(String(8), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    speed_requested: Mapped[str | None] = mapped_column(String(32))
    speed_processed: Mapped[str | None] = mapped_column(String(32))
    provider_created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), index=True
    )
    source_event_id: Mapped[str] = mapped_column(String(128), nullable=False)
    source_event_created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    authoritative_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    enriched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ProcessorRefundObservationRow(Base):
    __tablename__ = "processor_refund_observations"
    __table_args__ = (
        UniqueConstraint("source_event_id", name="uq_refund_observation_source_event"),
    )

    observation_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    refund_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    payment_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    account_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    amount: Mapped[int] = mapped_column(BigInteger, nullable=False)
    currency: Mapped[str] = mapped_column(String(8), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    speed_requested: Mapped[str | None] = mapped_column(String(32))
    speed_processed: Mapped[str | None] = mapped_column(String(32))
    provider_created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_event_id: Mapped[str] = mapped_column(String(128), nullable=False)
    source_event_created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    authoritative_sha256: Mapped[str] = mapped_column(String(64), nullable=False)


class ProcessorSettlementRow(Base):
    __tablename__ = "processor_settlements"

    settlement_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False, default="RAZORPAY")
    account_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    amount: Mapped[int] = mapped_column(BigInteger, nullable=False)
    currency: Mapped[str | None] = mapped_column(String(8))
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    fees: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    tax: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    utr: Mapped[str | None] = mapped_column(String(128), index=True)
    provider_created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), index=True
    )
    source_event_id: Mapped[str] = mapped_column(String(128), nullable=False)
    source_event_created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    authoritative_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    enriched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ProcessorSettlementObservationRow(Base):
    __tablename__ = "processor_settlement_observations"
    __table_args__ = (
        UniqueConstraint("source_event_id", name="uq_settlement_observation_source_event"),
    )

    observation_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    settlement_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    account_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    amount: Mapped[int] = mapped_column(BigInteger, nullable=False)
    currency: Mapped[str | None] = mapped_column(String(8))
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    fees: Mapped[int] = mapped_column(BigInteger, nullable=False)
    tax: Mapped[int] = mapped_column(BigInteger, nullable=False)
    utr: Mapped[str | None] = mapped_column(String(128))
    provider_created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_event_id: Mapped[str] = mapped_column(String(128), nullable=False)
    source_event_created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    authoritative_sha256: Mapped[str] = mapped_column(String(64), nullable=False)


class ProcessorDisputeRow(Base):
    __tablename__ = "processor_disputes"

    dispute_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False, default="RAZORPAY")
    account_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    payment_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    amount: Mapped[int] = mapped_column(BigInteger, nullable=False)
    currency: Mapped[str] = mapped_column(String(8), nullable=False)
    amount_deducted: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    reason_code: Mapped[str | None] = mapped_column(String(128), index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    phase: Mapped[str | None] = mapped_column(String(64), index=True)
    respond_by: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    provider_created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), index=True
    )
    source_event_id: Mapped[str] = mapped_column(String(128), nullable=False)
    source_event_created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    authoritative_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    enriched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ProcessorDisputeObservationRow(Base):
    __tablename__ = "processor_dispute_observations"
    __table_args__ = (
        UniqueConstraint("source_event_id", name="uq_dispute_observation_source_event"),
    )

    observation_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    dispute_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    payment_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    account_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    amount: Mapped[int] = mapped_column(BigInteger, nullable=False)
    currency: Mapped[str] = mapped_column(String(8), nullable=False)
    amount_deducted: Mapped[int] = mapped_column(BigInteger, nullable=False)
    reason_code: Mapped[str | None] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    phase: Mapped[str | None] = mapped_column(String(64))
    respond_by: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    provider_created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_event_id: Mapped[str] = mapped_column(String(128), nullable=False)
    source_event_created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    authoritative_sha256: Mapped[str] = mapped_column(String(64), nullable=False)


class ProcessorReconciliationItemRow(Base):
    __tablename__ = "processor_reconciliation_items"

    reconciliation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False, default="RAZORPAY")
    entity_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    debit: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    credit: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    amount: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    currency: Mapped[str] = mapped_column(String(8), nullable=False)
    fee: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    tax: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    on_hold: Mapped[bool | None] = mapped_column(Boolean)
    settled: Mapped[bool | None] = mapped_column(Boolean)
    provider_created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), index=True
    )
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    settlement_id: Mapped[str | None] = mapped_column(String(128), index=True)
    payment_id: Mapped[str | None] = mapped_column(String(128), index=True)
    order_id: Mapped[str | None] = mapped_column(String(128), index=True)
    dispute_id: Mapped[str | None] = mapped_column(String(128), index=True)
    method: Mapped[str | None] = mapped_column(String(64))
    settlement_utr: Mapped[str | None] = mapped_column(String(128), index=True)
    fetched_year: Mapped[int] = mapped_column(Integer, nullable=False)
    fetched_month: Mapped[int] = mapped_column(Integer, nullable=False)
    fetched_day: Mapped[int | None] = mapped_column(Integer)
    authoritative_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class MerchantProcessorAccountRow(Base):
    __tablename__ = "merchant_processor_accounts"
    __table_args__ = (
        UniqueConstraint("provider", "provider_account_id", name="uq_processor_merchant_account"),
    )

    account_row_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    provider_account_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_payment_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_refund_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_settlement_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_dispute_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_event_type: Mapped[str | None] = mapped_column(String(128))


class SqlRazorpayEventStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def healthcheck(self) -> None:
        async with self._sessions() as session:
            await session.execute(select(func.count()).select_from(ProcessorWebhookEventRow))

    async def store(
        self,
        *,
        provider_event_id: str,
        envelope: RazorpayWebhookEnvelope,
        payload_sha256: str,
        summary: dict[str, Any],
        received_at: datetime,
    ) -> tuple[RazorpayStoredEvent, bool]:
        async with self._sessions() as session:
            existing = await session.scalar(
                select(ProcessorWebhookEventRow).where(
                    ProcessorWebhookEventRow.provider == "RAZORPAY",
                    ProcessorWebhookEventRow.provider_event_id == provider_event_id,
                )
            )
            if existing is not None:
                return _from_row(existing), False
            row = ProcessorWebhookEventRow(
                provider="RAZORPAY",
                provider_event_id=provider_event_id,
                event_type=envelope.event,
                account_id=envelope.account_id,
                event_created_at=envelope.event_created_at(),
                payload_sha256=payload_sha256,
                summary=summary,
                received_at=received_at,
                processing_status="RECEIVED",
                processing_attempts=0,
            )
            session.add(row)
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                existing = await session.scalar(
                    select(ProcessorWebhookEventRow).where(
                        ProcessorWebhookEventRow.provider == "RAZORPAY",
                        ProcessorWebhookEventRow.provider_event_id == provider_event_id,
                    )
                )
                if existing is None:
                    raise
                return _from_row(existing), False
            return _from_row(row), True

    async def stats(self) -> RazorpayIngestionStats:
        async with self._sessions() as session:
            total = int(
                await session.scalar(
                    select(func.count())
                    .select_from(ProcessorWebhookEventRow)
                    .where(ProcessorWebhookEventRow.provider == "RAZORPAY")
                )
                or 0
            )
            latest = await session.scalar(
                select(ProcessorWebhookEventRow)
                .where(ProcessorWebhookEventRow.provider == "RAZORPAY")
                .order_by(ProcessorWebhookEventRow.received_at.desc())
                .limit(1)
            )
        if latest is None:
            return RazorpayIngestionStats(total_events=total)
        return RazorpayIngestionStats(
            total_events=total,
            last_event_at=latest.received_at,
            last_event_type=latest.event_type,
        )


class SqlRazorpayReconstructionStore:
    """Transactional multi-entity event reconstruction persistence."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def healthcheck(self) -> None:
        async with self._sessions() as session:
            await session.execute(select(func.count()).select_from(ProcessorPaymentRow))
            await session.execute(select(func.count()).select_from(ProcessorRefundRow))
            await session.execute(select(func.count()).select_from(ProcessorSettlementRow))
            await session.execute(select(func.count()).select_from(ProcessorDisputeRow))

    async def get_event(self, event_id: UUID) -> RazorpayStoredEvent | None:
        async with self._sessions() as session:
            row = await session.get(ProcessorWebhookEventRow, event_id)
            return _from_row(row) if row is not None else None

    async def claim_event(self, event_id: UUID) -> RazorpayStoredEvent | None:
        async with self._sessions() as session:
            row = await session.scalar(
                select(ProcessorWebhookEventRow)
                .where(
                    ProcessorWebhookEventRow.event_id == event_id,
                    ProcessorWebhookEventRow.processing_status.in_(["RECEIVED", "RETRY"]),
                )
                .with_for_update()
            )
            if row is None:
                return None
            row.processing_status = "PROCESSING"
            row.processing_attempts += 1
            row.processing_started_at = datetime.now(UTC)
            row.last_error = None
            await session.commit()
            return _from_row(row)

    async def complete_payment_event(
        self, event_id: UUID, payment: NormalizedProcessorPayment
    ) -> None:
        async with self._sessions() as session:
            now = datetime.now(UTC)
            await self._upsert_payment(session, payment, now)
            await self._insert_payment_observation(session, payment)
            await self._touch_account_activity(
                session,
                account_id=payment.account_id,
                event_time=payment.source_event_created_at or payment.enriched_at,
                activity_time=payment.provider_created_at or payment.enriched_at,
                activity="payment",
            )
            await self._complete_event(session, event_id, now)
            await session.commit()

    async def complete_refund_event(
        self,
        event_id: UUID,
        refund: NormalizedProcessorRefund,
        payment: NormalizedProcessorPayment,
    ) -> None:
        async with self._sessions() as session:
            now = datetime.now(UTC)
            await self._upsert_payment(session, payment, now)
            await self._insert_payment_observation(session, payment)
            row = await session.get(ProcessorRefundRow, refund.refund_id)
            if row is None:
                row = ProcessorRefundRow(refund_id=refund.refund_id, provider="RAZORPAY")
                session.add(row)
            _assign_refund(row, refund, now)
            observation = await session.scalar(
                select(ProcessorRefundObservationRow).where(
                    ProcessorRefundObservationRow.source_event_id == refund.source_event_id
                )
            )
            if observation is None:
                session.add(_refund_observation(refund))
            await self._touch_account_activity(
                session,
                account_id=refund.account_id,
                event_time=refund.source_event_created_at or refund.enriched_at,
                activity_time=refund.provider_created_at or refund.enriched_at,
                activity="refund",
            )
            await self._complete_event(session, event_id, now)
            await session.commit()

    async def complete_settlement_event(
        self, event_id: UUID, settlement: NormalizedProcessorSettlement
    ) -> None:
        async with self._sessions() as session:
            now = datetime.now(UTC)
            row = await session.get(ProcessorSettlementRow, settlement.settlement_id)
            if row is None:
                row = ProcessorSettlementRow(
                    settlement_id=settlement.settlement_id, provider="RAZORPAY"
                )
                session.add(row)
            _assign_settlement(row, settlement, now)
            observation = await session.scalar(
                select(ProcessorSettlementObservationRow).where(
                    ProcessorSettlementObservationRow.source_event_id == settlement.source_event_id
                )
            )
            if observation is None:
                session.add(_settlement_observation(settlement))
            await self._touch_account_activity(
                session,
                account_id=settlement.account_id,
                event_time=settlement.source_event_created_at or settlement.enriched_at,
                activity_time=settlement.provider_created_at or settlement.enriched_at,
                activity="settlement",
            )
            await self._complete_event(session, event_id, now)
            await session.commit()

    async def complete_dispute_event(
        self,
        event_id: UUID,
        dispute: NormalizedProcessorDispute,
        payment: NormalizedProcessorPayment,
    ) -> None:
        async with self._sessions() as session:
            now = datetime.now(UTC)
            await self._upsert_payment(session, payment, now)
            await self._insert_payment_observation(session, payment)
            row = await session.get(ProcessorDisputeRow, dispute.dispute_id)
            if row is None:
                row = ProcessorDisputeRow(dispute_id=dispute.dispute_id, provider="RAZORPAY")
                session.add(row)
            _assign_dispute(row, dispute, now)
            observation = await session.scalar(
                select(ProcessorDisputeObservationRow).where(
                    ProcessorDisputeObservationRow.source_event_id == dispute.source_event_id
                )
            )
            if observation is None:
                session.add(_dispute_observation(dispute))
            await self._touch_account_activity(
                session,
                account_id=dispute.account_id,
                event_time=dispute.source_event_created_at or dispute.enriched_at,
                activity_time=dispute.provider_created_at or dispute.enriched_at,
                activity="dispute",
            )
            await self._complete_event(session, event_id, now)
            await session.commit()

    async def upsert_reconciliation_items(self, items: list[NormalizedReconciliationItem]) -> int:
        async with self._sessions() as session:
            now = datetime.now(UTC)
            for item in items:
                row = await session.get(ProcessorReconciliationItemRow, item.reconciliation_id)
                if row is None:
                    row = ProcessorReconciliationItemRow(
                        reconciliation_id=item.reconciliation_id,
                        provider="RAZORPAY",
                    )
                    session.add(row)
                _assign_reconciliation(row, item, now)
            await session.commit()
        return len(items)

    async def mark_skipped(self, event_id: UUID, reason: str) -> None:
        await self._mark(event_id, "SKIPPED", error=reason, processed=True)

    async def mark_retry(self, event_id: UUID, error: str) -> None:
        await self._mark(event_id, "RETRY", error=error, processed=False)

    async def mark_failed(self, event_id: UUID, error: str) -> None:
        await self._mark(event_id, "FAILED", error=error, processed=True)

    async def pending_event_ids(self, *, limit: int) -> list[UUID]:
        async with self._sessions() as session:
            rows = (
                await session.scalars(
                    select(ProcessorWebhookEventRow.event_id)
                    .where(
                        ProcessorWebhookEventRow.provider == "RAZORPAY",
                        ProcessorWebhookEventRow.processing_status.in_(["RECEIVED", "RETRY"]),
                    )
                    .order_by(ProcessorWebhookEventRow.received_at.asc())
                    .limit(limit)
                )
            ).all()
            return list(rows)

    async def recover_stale_processing(self, *, stale_before: datetime) -> int:
        async with self._sessions() as session:
            result = await session.scalars(
                update(ProcessorWebhookEventRow)
                .where(
                    ProcessorWebhookEventRow.provider == "RAZORPAY",
                    ProcessorWebhookEventRow.processing_status == "PROCESSING",
                    ProcessorWebhookEventRow.processing_started_at.is_not(None),
                    ProcessorWebhookEventRow.processing_started_at < stale_before,
                )
                .values(
                    processing_status="RETRY",
                    processing_started_at=None,
                    last_error="stale_processing_recovered",
                )
                .returning(ProcessorWebhookEventRow.event_id)
            )
            recovered = len(result.all())
            await session.commit()
            return recovered

    async def processing_stats(self) -> RazorpayProcessingStats:
        async with self._sessions() as session:
            status_rows = (
                await session.execute(
                    select(ProcessorWebhookEventRow.processing_status, func.count())
                    .where(ProcessorWebhookEventRow.provider == "RAZORPAY")
                    .group_by(ProcessorWebhookEventRow.processing_status)
                )
            ).all()
            by_status = {str(status): int(count) for status, count in status_rows}
            payments = int(
                await session.scalar(select(func.count()).select_from(ProcessorPaymentRow)) or 0
            )
            refunds = int(
                await session.scalar(select(func.count()).select_from(ProcessorRefundRow)) or 0
            )
            settlements = int(
                await session.scalar(select(func.count()).select_from(ProcessorSettlementRow)) or 0
            )
            disputes = int(
                await session.scalar(select(func.count()).select_from(ProcessorDisputeRow)) or 0
            )
            recon = int(
                await session.scalar(
                    select(func.count()).select_from(ProcessorReconciliationItemRow)
                )
                or 0
            )
            last_processed = await session.scalar(
                select(func.max(ProcessorWebhookEventRow.processed_at)).where(
                    ProcessorWebhookEventRow.provider == "RAZORPAY"
                )
            )
        return RazorpayProcessingStats(
            pending_events=by_status.get("RECEIVED", 0),
            processing_events=by_status.get("PROCESSING", 0),
            processed_events=by_status.get("PROCESSED", 0),
            retry_events=by_status.get("RETRY", 0),
            failed_events=by_status.get("FAILED", 0),
            skipped_events=by_status.get("SKIPPED", 0),
            normalized_payments=payments,
            normalized_refunds=refunds,
            normalized_settlements=settlements,
            normalized_disputes=disputes,
            reconciliation_items=recon,
            last_processed_at=last_processed,
        )

    async def _mark(
        self,
        event_id: UUID,
        status: str,
        *,
        error: str | None = None,
        processed: bool,
    ) -> None:
        async with self._sessions() as session:
            values: dict[str, Any] = {
                "processing_status": status,
                "processing_started_at": None,
                "last_error": error[:128] if error else None,
            }
            if processed:
                values["processed_at"] = datetime.now(UTC)
            await session.execute(
                update(ProcessorWebhookEventRow)
                .where(ProcessorWebhookEventRow.event_id == event_id)
                .values(**values)
            )
            await session.commit()

    async def _upsert_payment(
        self,
        session: AsyncSession,
        payment: NormalizedProcessorPayment,
        now: datetime,
    ) -> None:
        row = await session.get(ProcessorPaymentRow, payment.payment_id)
        if row is None:
            row = ProcessorPaymentRow(payment_id=payment.payment_id, provider="RAZORPAY")
            session.add(row)
        row.account_id = payment.account_id
        row.order_id = payment.order_id
        row.amount = payment.amount
        row.currency = payment.currency
        row.status = payment.status
        row.method = payment.method
        row.captured = payment.captured
        row.amount_refunded = payment.amount_refunded
        row.refund_status = payment.refund_status
        row.international = payment.international
        row.provider_created_at = payment.provider_created_at
        row.source_event_id = payment.source_event_id
        row.source_event_created_at = payment.source_event_created_at
        row.authoritative_sha256 = payment.authoritative_sha256
        row.enriched_at = payment.enriched_at
        row.updated_at = now

    async def _insert_payment_observation(
        self, session: AsyncSession, payment: NormalizedProcessorPayment
    ) -> None:
        existing = await session.scalar(
            select(ProcessorPaymentObservationRow).where(
                ProcessorPaymentObservationRow.source_event_id == payment.source_event_id
            )
        )
        if existing is not None:
            return
        session.add(
            ProcessorPaymentObservationRow(
                payment_id=payment.payment_id,
                account_id=payment.account_id,
                order_id=payment.order_id,
                status=payment.status,
                amount=payment.amount,
                currency=payment.currency,
                method=payment.method,
                captured=payment.captured,
                amount_refunded=payment.amount_refunded,
                refund_status=payment.refund_status,
                international=payment.international,
                provider_created_at=payment.provider_created_at,
                source_event_id=payment.source_event_id,
                source_event_created_at=payment.source_event_created_at,
                observed_at=payment.enriched_at,
                authoritative_sha256=payment.authoritative_sha256,
            )
        )

    async def _complete_event(self, session: AsyncSession, event_id: UUID, now: datetime) -> None:
        event = await session.get(ProcessorWebhookEventRow, event_id)
        if event is None:
            raise ValueError("processor webhook event disappeared during completion")
        event.processing_status = "PROCESSED"
        event.processing_started_at = None
        event.processed_at = now
        event.last_error = None

    async def _touch_account_activity(
        self,
        session: AsyncSession,
        *,
        account_id: str | None,
        event_time: datetime,
        activity_time: datetime,
        activity: str,
    ) -> None:
        if account_id is None:
            return
        existing = await session.scalar(
            select(MerchantProcessorAccountRow).where(
                MerchantProcessorAccountRow.provider == "RAZORPAY",
                MerchantProcessorAccountRow.provider_account_id == account_id,
            )
        )
        if existing is None:
            existing = MerchantProcessorAccountRow(
                provider="RAZORPAY",
                provider_account_id=account_id,
                first_seen_at=event_time,
                last_seen_at=event_time,
                last_event_type=activity,
            )
            session.add(existing)
        else:
            existing.last_seen_at = max(existing.last_seen_at, event_time)
            existing.last_event_type = activity
        field_name = f"last_{activity}_at"
        if hasattr(existing, field_name):
            previous = getattr(existing, field_name)
            if previous is None or activity_time > previous:
                setattr(existing, field_name, activity_time)


def _assign_refund(row: ProcessorRefundRow, item: NormalizedProcessorRefund, now: datetime) -> None:
    row.account_id = item.account_id
    row.payment_id = item.payment_id
    row.amount = item.amount
    row.currency = item.currency
    row.status = item.status
    row.speed_requested = item.speed_requested
    row.speed_processed = item.speed_processed
    row.provider_created_at = item.provider_created_at
    row.source_event_id = item.source_event_id
    row.source_event_created_at = item.source_event_created_at
    row.authoritative_sha256 = item.authoritative_sha256
    row.enriched_at = item.enriched_at
    row.updated_at = now


def _refund_observation(item: NormalizedProcessorRefund) -> ProcessorRefundObservationRow:
    return ProcessorRefundObservationRow(
        refund_id=item.refund_id,
        payment_id=item.payment_id,
        account_id=item.account_id,
        amount=item.amount,
        currency=item.currency,
        status=item.status,
        speed_requested=item.speed_requested,
        speed_processed=item.speed_processed,
        provider_created_at=item.provider_created_at,
        source_event_id=item.source_event_id,
        source_event_created_at=item.source_event_created_at,
        observed_at=item.enriched_at,
        authoritative_sha256=item.authoritative_sha256,
    )


def _assign_settlement(
    row: ProcessorSettlementRow, item: NormalizedProcessorSettlement, now: datetime
) -> None:
    row.account_id = item.account_id
    row.amount = item.amount
    row.currency = item.currency
    row.status = item.status
    row.fees = item.fees
    row.tax = item.tax
    row.utr = item.utr
    row.provider_created_at = item.provider_created_at
    row.source_event_id = item.source_event_id
    row.source_event_created_at = item.source_event_created_at
    row.authoritative_sha256 = item.authoritative_sha256
    row.enriched_at = item.enriched_at
    row.updated_at = now


def _settlement_observation(
    item: NormalizedProcessorSettlement,
) -> ProcessorSettlementObservationRow:
    return ProcessorSettlementObservationRow(
        settlement_id=item.settlement_id,
        account_id=item.account_id,
        amount=item.amount,
        currency=item.currency,
        status=item.status,
        fees=item.fees,
        tax=item.tax,
        utr=item.utr,
        provider_created_at=item.provider_created_at,
        source_event_id=item.source_event_id,
        source_event_created_at=item.source_event_created_at,
        observed_at=item.enriched_at,
        authoritative_sha256=item.authoritative_sha256,
    )


def _assign_dispute(
    row: ProcessorDisputeRow, item: NormalizedProcessorDispute, now: datetime
) -> None:
    row.account_id = item.account_id
    row.payment_id = item.payment_id
    row.amount = item.amount
    row.currency = item.currency
    row.amount_deducted = item.amount_deducted
    row.reason_code = item.reason_code
    row.status = item.status
    row.phase = item.phase
    row.respond_by = item.respond_by
    row.provider_created_at = item.provider_created_at
    row.source_event_id = item.source_event_id
    row.source_event_created_at = item.source_event_created_at
    row.authoritative_sha256 = item.authoritative_sha256
    row.enriched_at = item.enriched_at
    row.updated_at = now


def _dispute_observation(item: NormalizedProcessorDispute) -> ProcessorDisputeObservationRow:
    return ProcessorDisputeObservationRow(
        dispute_id=item.dispute_id,
        payment_id=item.payment_id,
        account_id=item.account_id,
        amount=item.amount,
        currency=item.currency,
        amount_deducted=item.amount_deducted,
        reason_code=item.reason_code,
        status=item.status,
        phase=item.phase,
        respond_by=item.respond_by,
        provider_created_at=item.provider_created_at,
        source_event_id=item.source_event_id,
        source_event_created_at=item.source_event_created_at,
        observed_at=item.enriched_at,
        authoritative_sha256=item.authoritative_sha256,
    )


def _assign_reconciliation(
    row: ProcessorReconciliationItemRow, item: NormalizedReconciliationItem, now: datetime
) -> None:
    row.entity_id = item.entity_id
    row.entity_type = item.entity_type
    row.debit = item.debit
    row.credit = item.credit
    row.amount = item.amount
    row.currency = item.currency
    row.fee = item.fee
    row.tax = item.tax
    row.on_hold = item.on_hold
    row.settled = item.settled
    row.provider_created_at = item.provider_created_at
    row.settled_at = item.settled_at
    row.settlement_id = item.settlement_id
    row.payment_id = item.payment_id
    row.order_id = item.order_id
    row.dispute_id = item.dispute_id
    row.method = item.method
    row.settlement_utr = item.settlement_utr
    row.fetched_year = item.fetched_year
    row.fetched_month = item.fetched_month
    row.fetched_day = item.fetched_day
    row.authoritative_sha256 = item.authoritative_sha256
    row.fetched_at = item.fetched_at
    row.updated_at = now


def _from_row(row: ProcessorWebhookEventRow) -> RazorpayStoredEvent:
    return RazorpayStoredEvent(
        event_id=row.event_id,
        provider_event_id=row.provider_event_id,
        event_type=row.event_type,
        account_id=row.account_id,
        event_created_at=row.event_created_at,
        payload_sha256=row.payload_sha256,
        summary=row.summary,
        received_at=row.received_at,
        processing_status=row.processing_status,
        processing_attempts=row.processing_attempts,
    )
