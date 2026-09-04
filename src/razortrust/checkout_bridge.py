from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import BigInteger, Boolean, DateTime, Integer, String, Uuid, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base
from .live_features import (
    SOURCE_CONTRACT_VERSION,
    FirstPartyTelemetryRow,
    hash_device_pseudonym,
)
from .razorpay_store import MerchantProcessorAccountRow

CHECKOUT_CONTRACT_VERSION = "razorpay-standard-checkout-bridge@1"
CHECKOUT_LIFECYCLE_VERSION = "razorpay-checkout-lifecycle@1"
CHECKOUT_SIGNATURE_VERSION = "razorpay-order-payment-hmac-sha256@1"
DEFAULT_CHECKOUT_EXPIRY_HOURS = 24


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CheckoutOrderCreate(StrictModel):
    amount: int = Field(default=1000, ge=100, le=10_000_000)
    currency: str = Field(default="INR", pattern=r"^INR$")
    device_pseudonym: str = Field(min_length=16, max_length=256)
    customer_geo: str = Field(
        min_length=2,
        max_length=64,
        pattern=r"^[A-Z]{2}(?:-[A-Z0-9]{1,12}){0,2}$",
        description="Coarse geography only, for example IN-DL. Never raw GPS.",
    )
    client_event_at: datetime | None = None

    @field_validator("client_event_at")
    @classmethod
    def normalize_client_time(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("client_event_at must include a timezone")
        return value.astimezone(UTC)


class CheckoutOrderResponse(StrictModel):
    checkout_session_id: UUID
    razorpay_order_id: str = Field(pattern=r"^order_[A-Za-z0-9]+$")
    amount: int
    currency: str
    key_id: str
    provider_status: str
    checkout_contract_version: str = CHECKOUT_CONTRACT_VERSION


class CheckoutVerificationSubmission(StrictModel):
    checkout_session_id: UUID
    razorpay_payment_id: str = Field(pattern=r"^pay_[A-Za-z0-9]+$")
    razorpay_order_id: str = Field(pattern=r"^order_[A-Za-z0-9]+$")
    razorpay_signature: str = Field(min_length=64, max_length=128, pattern=r"^[0-9a-fA-F]+$")


class CheckoutVerificationResponse(StrictModel):
    status: str
    checkout_session_id: UUID
    razorpay_order_id: str
    razorpay_payment_id: str
    signature_verified: bool
    authoritative_payment_status: str
    telemetry_bound: bool
    fulfillment_eligible: bool
    shadow_score_only: bool = True
    production_action_eligible: bool = False
    checkout_signature_version: str = CHECKOUT_SIGNATURE_VERSION


class CheckoutOrderRow(Base):
    __tablename__ = "razorpay_checkout_orders"

    checkout_session_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    account_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    razorpay_order_id: Mapped[str | None] = mapped_column(String(128), unique=True, index=True)
    razorpay_payment_id: Mapped[str | None] = mapped_column(String(128), index=True)
    amount: Mapped[int] = mapped_column(BigInteger, nullable=False)
    currency: Mapped[str] = mapped_column(String(8), nullable=False)
    receipt: Mapped[str] = mapped_column(String(40), unique=True, nullable=False, index=True)
    provider_status: Mapped[str] = mapped_column(String(32), nullable=False)
    provider_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    lifecycle_status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    last_payment_status: Mapped[str | None] = mapped_column(String(32))
    device_fingerprint_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    customer_geo: Mapped[str] = mapped_column(String(64), nullable=False)
    client_event_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    telemetry_id: Mapped[UUID | None] = mapped_column(Uuid, index=True)
    authoritative_payment_status: Mapped[str | None] = mapped_column(String(32))
    signature_verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), index=True
    )
    webhook_confirmed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), index=True
    )
    last_provider_reconciled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    abandoned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    checkout_contract_version: Mapped[str] = mapped_column(String(64), nullable=False)


class CheckoutPaymentAttemptRow(Base):
    __tablename__ = "razorpay_checkout_payment_attempts"

    attempt_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    checkout_session_id: Mapped[UUID] = mapped_column(Uuid, nullable=False, index=True)
    account_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    razorpay_order_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    razorpay_payment_id: Mapped[str] = mapped_column(
        String(128), nullable=False, unique=True, index=True
    )
    payment_status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    captured: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    last_event_type: Mapped[str] = mapped_column(String(128), nullable=False)
    last_source_event_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class RazorpayOrderClient(Protocol):
    async def fetch_order(self, order_id: str) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class CheckoutReconcileResult:
    inspected: int = 0
    updated: int = 0
    paid: int = 0
    attempted: int = 0
    open: int = 0
    abandoned: int = 0
    errors: int = 0


def verify_checkout_signature(
    *, order_id: str, payment_id: str, signature: str, secret: str
) -> bool:
    if not order_id or not payment_id or not signature or not secret:
        return False
    material = f"{order_id}|{payment_id}".encode()
    expected = hmac.new(secret.encode("utf-8"), material, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature.strip().lower())


def validate_authoritative_payment(
    *,
    stored_order_id: str,
    stored_amount: int,
    stored_currency: str,
    payment_id: str,
    payment: dict[str, object],
) -> str:
    if payment.get("id") != payment_id:
        raise ValueError("authoritative payment id mismatch")
    if payment.get("order_id") != stored_order_id:
        raise ValueError("authoritative payment order mismatch")
    if payment.get("amount") != stored_amount:
        raise ValueError("authoritative payment amount mismatch")
    if payment.get("currency") != stored_currency:
        raise ValueError("authoritative payment currency mismatch")
    status = payment.get("status")
    if not isinstance(status, str) or status not in {"authorized", "captured", "refunded"}:
        raise ValueError("authoritative payment is not in a successful state")
    return status


def derive_checkout_lifecycle(*, payment_status: str | None, order_status: str | None) -> str:
    payment = (payment_status or "").strip().lower()
    order = (order_status or "").strip().lower()
    if order == "paid" or payment in {"captured", "refunded"}:
        return "PAID"
    if payment == "authorized":
        return "AUTHORIZED"
    if payment == "failed":
        return "ATTEMPT_FAILED"
    if order == "attempted":
        return "ATTEMPTED"
    if order == "created":
        return "OPEN"
    return "OPEN"


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


class SqlCheckoutBridgeStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def healthcheck(self) -> None:
        async with self._sessions() as session:
            await session.execute(select(func.count()).select_from(CheckoutOrderRow))

    async def stats(self) -> dict[str, int]:
        async with self._sessions() as session:
            total = int(
                await session.scalar(select(func.count()).select_from(CheckoutOrderRow)) or 0
            )
            verified = int(
                await session.scalar(
                    select(func.count())
                    .select_from(CheckoutOrderRow)
                    .where(CheckoutOrderRow.signature_verified_at.is_not(None))
                )
                or 0
            )
            attempts_total = int(
                await session.scalar(select(func.count()).select_from(CheckoutPaymentAttemptRow))
                or 0
            )
            lifecycle_counts: dict[str, int] = {}
            for lifecycle in (
                "OPEN",
                "AUTHORIZED",
                "ATTEMPTED",
                "ATTEMPT_FAILED",
                "PAID",
                "ABANDONED",
            ):
                lifecycle_counts[lifecycle] = int(
                    await session.scalar(
                        select(func.count())
                        .select_from(CheckoutOrderRow)
                        .where(CheckoutOrderRow.lifecycle_status == lifecycle)
                    )
                    or 0
                )
        return {
            "checkout_sessions": total,
            "verified_checkout_sessions": verified,
            "checkout_payment_attempts": attempts_total,
            "checkout_open_sessions": lifecycle_counts["OPEN"],
            "checkout_authorized_sessions": lifecycle_counts["AUTHORIZED"],
            "checkout_attempted_sessions": lifecycle_counts["ATTEMPTED"]
            + lifecycle_counts["ATTEMPT_FAILED"],
            "checkout_paid_sessions": lifecycle_counts["PAID"],
            "checkout_abandoned_sessions": lifecycle_counts["ABANDONED"],
        }

    async def resolve_account_id(self, *, preferred_account_id: str | None = None) -> str:
        async with self._sessions() as session:
            if preferred_account_id:
                account = await session.scalar(
                    select(MerchantProcessorAccountRow).where(
                        MerchantProcessorAccountRow.provider == "RAZORPAY",
                        MerchantProcessorAccountRow.provider_account_id == preferred_account_id,
                    )
                )
                if account is None:
                    raise LookupError(
                        "authenticated merchant is not a known Razorpay processor account"
                    )
                return account.provider_account_id
            accounts = list(
                (
                    await session.scalars(
                        select(MerchantProcessorAccountRow)
                        .where(MerchantProcessorAccountRow.provider == "RAZORPAY")
                        .order_by(MerchantProcessorAccountRow.first_seen_at)
                        .limit(2)
                    )
                ).all()
            )
            if not accounts:
                raise LookupError("no Razorpay processor account is known yet")
            if len(accounts) != 1:
                raise ValueError(
                    "multiple Razorpay accounts require authenticated merchant binding"
                )
            return accounts[0].provider_account_id

    async def create_pending(
        self,
        *,
        account_id: str,
        amount: int,
        currency: str,
        device_pseudonym: str,
        customer_geo: str,
        client_event_at: datetime | None,
    ) -> CheckoutOrderRow:
        now = datetime.now(UTC)
        if client_event_at is not None and client_event_at > now + timedelta(minutes=5):
            raise ValueError("client_event_at cannot be more than five minutes in the future")
        session_id = uuid4()
        receipt = f"rt-{session_id.hex[:24]}"
        row = CheckoutOrderRow(
            checkout_session_id=session_id,
            account_id=account_id,
            amount=amount,
            currency=currency,
            receipt=receipt,
            provider_status="PENDING_PROVIDER_ORDER",
            provider_attempts=0,
            lifecycle_status="PENDING_PROVIDER_ORDER",
            device_fingerprint_sha256=hash_device_pseudonym(account_id, device_pseudonym),
            customer_geo=customer_geo,
            client_event_at=client_event_at,
            expires_at=now + timedelta(hours=DEFAULT_CHECKOUT_EXPIRY_HOURS),
            created_at=now,
            updated_at=now,
            checkout_contract_version=CHECKOUT_CONTRACT_VERSION,
        )
        async with self._sessions() as session:
            session.add(row)
            await session.commit()
            return row

    async def mark_provider_failed(self, checkout_session_id: UUID) -> None:
        async with self._sessions() as session:
            row = await session.get(CheckoutOrderRow, checkout_session_id)
            if row is None:
                return
            row.provider_status = "PROVIDER_ORDER_FAILED"
            row.lifecycle_status = "PROVIDER_ORDER_FAILED"
            row.updated_at = datetime.now(UTC)
            await session.commit()

    async def finalize_provider_order(
        self,
        *,
        checkout_session_id: UUID,
        provider_order: dict[str, object],
    ) -> CheckoutOrderRow:
        order_id = provider_order.get("id")
        status = provider_order.get("status")
        amount = provider_order.get("amount")
        currency = provider_order.get("currency")
        receipt = provider_order.get("receipt")
        attempts = provider_order.get("attempts", 0)
        if not isinstance(order_id, str) or not order_id.startswith("order_"):
            raise ValueError("Razorpay order response is missing a valid id")
        if status != "created":
            raise ValueError("Razorpay order response is not in created state")
        if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 0:
            attempts = 0
        async with self._sessions() as session:
            row = await session.get(CheckoutOrderRow, checkout_session_id)
            if row is None:
                raise LookupError("checkout session was not found")
            if amount != row.amount or currency != row.currency or receipt != row.receipt:
                raise ValueError(
                    "Razorpay order response does not match the stored checkout session"
                )
            now = datetime.now(UTC)
            telemetry_id = uuid4()
            safe_payload = {
                "account_id": row.account_id,
                "order_id": order_id,
                "payment_id": None,
                "device_fingerprint_sha256": row.device_fingerprint_sha256,
                "customer_geo": row.customer_geo,
                "client_event_at": row.client_event_at.isoformat() if row.client_event_at else None,
                "observed_at": now.isoformat(),
                "source_contract_version": SOURCE_CONTRACT_VERSION,
            }
            telemetry = FirstPartyTelemetryRow(
                telemetry_id=telemetry_id,
                account_id=row.account_id,
                order_id=order_id,
                payment_id=None,
                device_fingerprint_sha256=row.device_fingerprint_sha256,
                customer_geo=row.customer_geo,
                client_event_at=row.client_event_at,
                observed_at=now,
                payload_sha256=_canonical_sha256(safe_payload),
                source_contract_version=SOURCE_CONTRACT_VERSION,
            )
            session.add(telemetry)
            row.razorpay_order_id = order_id
            row.provider_status = status
            row.provider_attempts = attempts
            row.lifecycle_status = "OPEN"
            row.telemetry_id = telemetry_id
            row.last_provider_reconciled_at = now
            row.updated_at = now
            await session.commit()
            return row

    async def get_session(self, checkout_session_id: UUID) -> CheckoutOrderRow:
        async with self._sessions() as session:
            row = await session.get(CheckoutOrderRow, checkout_session_id)
            if row is None:
                raise LookupError("checkout session was not found")
            return row

    async def bind_verified_payment(
        self,
        *,
        checkout_session_id: UUID,
        payment_id: str,
        authoritative_status: str,
    ) -> CheckoutOrderRow:
        async with self._sessions() as session:
            row = await session.get(CheckoutOrderRow, checkout_session_id)
            if row is None:
                raise LookupError("checkout session was not found")
            if row.razorpay_payment_id not in {None, payment_id}:
                raise ValueError("checkout session is already bound to another payment")
            if row.telemetry_id is None:
                raise ValueError("checkout telemetry was not created")
            telemetry = await session.get(FirstPartyTelemetryRow, row.telemetry_id)
            if telemetry is None:
                raise ValueError("checkout telemetry record is missing")
            if telemetry.payment_id not in {None, payment_id}:
                raise ValueError("checkout telemetry is already bound to another payment")
            now = datetime.now(UTC)
            row.razorpay_payment_id = payment_id
            row.provider_attempts = max(row.provider_attempts, 1)
            row.authoritative_payment_status = authoritative_status
            row.last_payment_status = authoritative_status
            row.lifecycle_status = derive_checkout_lifecycle(
                payment_status=authoritative_status,
                order_status=row.provider_status,
            )
            if row.lifecycle_status == "PAID":
                row.provider_status = "paid"
            row.signature_verified_at = now
            row.updated_at = now
            telemetry.payment_id = payment_id
            telemetry.payload_sha256 = _canonical_sha256(
                {
                    "account_id": telemetry.account_id,
                    "order_id": telemetry.order_id,
                    "payment_id": payment_id,
                    "device_fingerprint_sha256": telemetry.device_fingerprint_sha256,
                    "customer_geo": telemetry.customer_geo,
                    "client_event_at": telemetry.client_event_at.isoformat()
                    if telemetry.client_event_at
                    else None,
                    "observed_at": telemetry.observed_at.isoformat(),
                    "source_contract_version": telemetry.source_contract_version,
                }
            )
            await session.commit()
            return row

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
    ) -> None:
        if not order_id:
            return
        async with self._sessions() as session:
            row = await session.scalar(
                select(CheckoutOrderRow).where(CheckoutOrderRow.razorpay_order_id == order_id)
            )
            if row is None:
                return
            now = observed_at.astimezone(UTC)
            attempt = await session.scalar(
                select(CheckoutPaymentAttemptRow).where(
                    CheckoutPaymentAttemptRow.razorpay_payment_id == payment_id
                )
            )
            if attempt is None:
                attempt = CheckoutPaymentAttemptRow(
                    attempt_id=uuid4(),
                    checkout_session_id=row.checkout_session_id,
                    account_id=row.account_id,
                    razorpay_order_id=order_id,
                    razorpay_payment_id=payment_id,
                    payment_status=payment_status,
                    captured=captured,
                    last_event_type=event_type,
                    last_source_event_id=source_event_id,
                    first_seen_at=now,
                    last_seen_at=now,
                    created_at=now,
                    updated_at=now,
                )
                session.add(attempt)
            else:
                if (
                    attempt.checkout_session_id != row.checkout_session_id
                    or attempt.razorpay_order_id != order_id
                ):
                    raise ValueError(
                        "Razorpay payment attempt is already bound to another checkout order"
                    )
                attempt.payment_status = payment_status
                attempt.captured = captured
                attempt.last_event_type = event_type
                attempt.last_source_event_id = source_event_id
                attempt.last_seen_at = now
                attempt.updated_at = now
            await session.flush()
            distinct_attempts = int(
                await session.scalar(
                    select(func.count())
                    .select_from(CheckoutPaymentAttemptRow)
                    .where(CheckoutPaymentAttemptRow.checkout_session_id == row.checkout_session_id)
                )
                or 0
            )
            row.provider_attempts = max(row.provider_attempts, distinct_attempts)
            row.last_payment_status = payment_status
            row.authoritative_payment_status = payment_status
            lifecycle = derive_checkout_lifecycle(
                payment_status=payment_status,
                order_status=row.provider_status,
            )
            row.lifecycle_status = lifecycle
            if lifecycle == "PAID":
                row.provider_status = "paid"
                row.razorpay_payment_id = payment_id
                row.webhook_confirmed_at = now
                if row.telemetry_id is not None:
                    telemetry = await session.get(FirstPartyTelemetryRow, row.telemetry_id)
                    if telemetry is not None and telemetry.payment_id in {None, payment_id}:
                        telemetry.payment_id = payment_id
                        telemetry.payload_sha256 = _canonical_sha256(
                            {
                                "account_id": telemetry.account_id,
                                "order_id": telemetry.order_id,
                                "payment_id": payment_id,
                                "device_fingerprint_sha256": telemetry.device_fingerprint_sha256,
                                "customer_geo": telemetry.customer_geo,
                                "client_event_at": telemetry.client_event_at.isoformat()
                                if telemetry.client_event_at
                                else None,
                                "observed_at": telemetry.observed_at.isoformat(),
                                "source_contract_version": telemetry.source_contract_version,
                            }
                        )
            elif payment_status.strip().lower() == "failed":
                # A checkout order may have multiple failed payment IDs before one succeeds.
                # Preserve every failed attempt in CheckoutPaymentAttemptRow, while leaving
                # the checkout/telemetry canonical payment_id free for the eventual success.
                pass
            elif payment_status.strip().lower() == "authorized":
                row.razorpay_payment_id = payment_id
            row.updated_at = now
            await session.commit()

    async def reconcile_open_orders(
        self,
        client: RazorpayOrderClient,
        *,
        limit: int = 100,
        abandon_after_hours: int = DEFAULT_CHECKOUT_EXPIRY_HOURS,
    ) -> CheckoutReconcileResult:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        if not 1 <= abandon_after_hours <= 24 * 30:
            raise ValueError("abandon_after_hours must be between 1 and 720")
        now = datetime.now(UTC)
        async with self._sessions() as session:
            rows = list(
                (
                    await session.scalars(
                        select(CheckoutOrderRow)
                        .where(
                            CheckoutOrderRow.razorpay_order_id.is_not(None),
                            CheckoutOrderRow.lifecycle_status.not_in(
                                ("ABANDONED", "PROVIDER_ORDER_FAILED")
                            ),
                            or_(
                                CheckoutOrderRow.lifecycle_status != "PAID",
                                CheckoutOrderRow.provider_attempts == 0,
                            ),
                        )
                        .order_by(CheckoutOrderRow.created_at)
                        .limit(limit)
                    )
                ).all()
            )

        updated = paid = attempted = open_count = abandoned = errors = 0
        for detached in rows:
            try:
                provider_order = await client.fetch_order(detached.razorpay_order_id or "")
                order_status = provider_order.get("status")
                attempts_raw = provider_order.get("attempts", 0)
                if order_status not in {"created", "attempted", "paid"}:
                    raise ValueError("unexpected Razorpay order status")
                attempts = (
                    attempts_raw
                    if isinstance(attempts_raw, int)
                    and not isinstance(attempts_raw, bool)
                    and attempts_raw >= 0
                    else 0
                )
                async with self._sessions() as session:
                    row = await session.get(CheckoutOrderRow, detached.checkout_session_id)
                    if row is None:
                        continue
                    row.provider_status = order_status
                    row.provider_attempts = attempts
                    row.last_provider_reconciled_at = now
                    if order_status == "paid":
                        row.lifecycle_status = "PAID"
                        paid += 1
                    elif order_status == "attempted":
                        row.lifecycle_status = (
                            "ATTEMPT_FAILED"
                            if (row.last_payment_status or "").lower() == "failed"
                            else "ATTEMPTED"
                        )
                        attempted += 1
                    else:
                        row.lifecycle_status = "OPEN"
                        open_count += 1
                    expiry = row.expires_at or (
                        row.created_at + timedelta(hours=abandon_after_hours)
                    )
                    row.expires_at = expiry
                    if order_status != "paid" and now >= expiry:
                        row.lifecycle_status = "ABANDONED"
                        row.abandoned_at = row.abandoned_at or now
                        abandoned += 1
                    row.updated_at = now
                    await session.commit()
                    updated += 1
            except Exception:
                errors += 1
        return CheckoutReconcileResult(
            inspected=len(rows),
            updated=updated,
            paid=paid,
            attempted=attempted,
            open=open_count,
            abandoned=abandoned,
            errors=errors,
        )
