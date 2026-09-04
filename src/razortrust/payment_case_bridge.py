from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .domain import HoldCase, HoldCreate, StrictModel
from .workflow import HoldService


class PaymentCandidate(StrictModel):
    provider: str = "RAZORPAY"
    payment_id: str
    account_id: str
    order_id: str | None
    amount: int
    currency: str
    status: str
    method: str | None
    captured: bool
    amount_refunded: int
    refund_status: str | None
    provider_created_at: datetime | None
    source_event_created_at: datetime | None
    updated_at: datetime
    decision_time: datetime
    reason_code: str
    authoritative_store_backed: bool
    authoritative_sha256: str | None


class PaymentCaseReceipt(StrictModel):
    provider: str = "RAZORPAY"
    payment: PaymentCandidate
    hold: HoldCase
    created: bool
    idempotency_key: UUID
    automatic_release_enabled: bool = False
    human_authorization_required: bool = True


class PaymentCaseBridgeUnavailable(RuntimeError):
    pass


def payment_reason_code(
    *,
    status: str,
    amount_refunded: int,
) -> str:
    normalized = status.strip().lower()
    if amount_refunded > 0 or normalized == "refunded":
        return "RAZORPAY_REFUND_OBSERVED"
    if normalized == "failed":
        return "RAZORPAY_PAYMENT_FAILED"
    if normalized == "captured":
        return "RAZORPAY_PAYMENT_CAPTURED"
    if normalized == "authorized":
        return "RAZORPAY_PAYMENT_AUTHORIZED"
    return "RAZORPAY_PAYMENT_OBSERVED"


def payment_case_request_id(account_id: str, payment_id: str) -> UUID:
    material = f"razortrust:razorpay-payment:{account_id}:{payment_id}"
    return uuid5(NAMESPACE_URL, material)


def payment_decision_time(row: dict[str, Any]) -> datetime:
    for name in (
        "source_event_created_at",
        "provider_created_at",
        "updated_at",
    ):
        value = row.get(name)
        if isinstance(value, datetime):
            if value.tzinfo is None or value.utcoffset() is None:
                return value.replace(tzinfo=UTC)
            return value.astimezone(UTC)
    raise ValueError("canonical payment does not contain a usable decision timestamp")


def _candidate(row: dict[str, Any]) -> PaymentCandidate:
    return PaymentCandidate(
        payment_id=str(row["payment_id"]),
        account_id=str(row["account_id"]),
        order_id=str(row["order_id"]) if row.get("order_id") else None,
        amount=int(row["amount"]),
        currency=str(row["currency"]),
        status=str(row["status"]),
        method=str(row["method"]) if row.get("method") else None,
        captured=bool(row["captured"]),
        amount_refunded=int(row.get("amount_refunded") or 0),
        refund_status=(
            str(row["refund_status"]) if row.get("refund_status") else None
        ),
        provider_created_at=row.get("provider_created_at"),
        source_event_created_at=row.get("source_event_created_at"),
        updated_at=row["updated_at"],
        decision_time=payment_decision_time(row),
        reason_code=payment_reason_code(
            status=str(row["status"]),
            amount_refunded=int(row.get("amount_refunded") or 0),
        ),
        authoritative_store_backed=bool(row.get("authoritative_sha256")),
        authoritative_sha256=(
            str(row["authoritative_sha256"])
            if row.get("authoritative_sha256")
            else None
        ),
    )


class SqlRazorpayPaymentCaseReader:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def list_candidates(self, *, limit: int = 100) -> list[PaymentCandidate]:
        if limit < 1 or limit > 500:
            raise ValueError("limit must be between 1 and 500")
        try:
            async with self._sessions() as session:
                result = await session.execute(
                    text(
                        """
                        SELECT
                            payment_id,
                            account_id,
                            order_id,
                            amount,
                            currency,
                            status,
                            method,
                            captured,
                            amount_refunded,
                            refund_status,
                            provider_created_at,
                            source_event_created_at,
                            authoritative_sha256,
                            updated_at
                        FROM processor_payments
                        WHERE provider = 'RAZORPAY'
                        ORDER BY COALESCE(
                            source_event_created_at,
                            provider_created_at,
                            updated_at
                        ) DESC
                        LIMIT :limit
                        """
                    ),
                    {"limit": limit},
                )
                return [_candidate(dict(row)) for row in result.mappings().all()]
        except SQLAlchemyError as exc:
            raise PaymentCaseBridgeUnavailable(
                f"payment case reader unavailable: {type(exc).__name__}"
            ) from exc

    async def get_candidate(self, payment_id: str) -> PaymentCandidate:
        try:
            async with self._sessions() as session:
                result = await session.execute(
                    text(
                        """
                        SELECT
                            payment_id,
                            account_id,
                            order_id,
                            amount,
                            currency,
                            status,
                            method,
                            captured,
                            amount_refunded,
                            refund_status,
                            provider_created_at,
                            source_event_created_at,
                            authoritative_sha256,
                            updated_at
                        FROM processor_payments
                        WHERE provider = 'RAZORPAY'
                          AND payment_id = :payment_id
                        LIMIT 1
                        """
                    ),
                    {"payment_id": payment_id},
                )
                row = result.mappings().first()
                if row is None:
                    raise LookupError("authoritative Razorpay payment was not found")
                return _candidate(dict(row))
        except SQLAlchemyError as exc:
            raise PaymentCaseBridgeUnavailable(
                f"payment case reader unavailable: {type(exc).__name__}"
            ) from exc


class RazorpayPaymentCaseBridge:
    def __init__(
        self,
        reader: SqlRazorpayPaymentCaseReader,
        hold_service: HoldService,
    ) -> None:
        self.reader = reader
        self.hold_service = hold_service

    async def create_or_get(self, payment_id: str) -> PaymentCaseReceipt:
        candidate = await self.reader.get_candidate(payment_id)
        request_id = payment_case_request_id(
            candidate.account_id,
            candidate.payment_id,
        )
        hold, created = await self.hold_service.create_hold(
            HoldCreate(
                request_id=request_id,
                merchant_id=candidate.account_id,
                source_event_id=candidate.payment_id,
                triggered_at=candidate.decision_time,
                reason_code=candidate.reason_code,
            )
        )
        return PaymentCaseReceipt(
            payment=candidate,
            hold=hold,
            created=created,
            idempotency_key=request_id,
        )
