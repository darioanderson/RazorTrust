from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from statistics import fmean, median, pstdev
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import DateTime, String, Uuid, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base
from .domain import HoldCase, HoldEvaluationInput, MerchantBaseline, TransactionEvent
from .features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, build_point_in_time_features
from .razorpay_store import (
    MerchantProcessorAccountRow,
    ProcessorDisputeObservationRow,
    ProcessorPaymentObservationRow,
    ProcessorPaymentRow,
    ProcessorRefundObservationRow,
)

SOURCE_CONTRACT_VERSION = "razorpay-first-party-source-contract@1"
BASELINE_METHOD_VERSION = "aligned-daily-baseline@1"
AUTH_MAPPING_VERSION = "razorpay-payment-status-auth@1"
CHARGEBACK_MAPPING_VERSION = "razorpay-dispute-created-chargeback@1"
DEVICE_PSEUDONYM_VERSION = "sha256-account-scoped-device@1"
KNOWLEDGE_TIME_POLICY = "OBSERVED_AT"
CURRENT_WINDOW_HOURS = 24
BASELINE_DAYS = 30
MIN_BASELINE_TRANSACTIONS = 30
MIN_BASELINE_ACTIVE_DAYS = 7


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FirstPartyTelemetrySubmission(StrictModel):
    account_id: str = Field(min_length=1, max_length=128)
    order_id: str = Field(min_length=1, max_length=128, pattern=r"^order_[A-Za-z0-9]+$")
    payment_id: str | None = Field(default=None, max_length=128, pattern=r"^pay_[A-Za-z0-9]+$")
    device_pseudonym: str = Field(min_length=16, max_length=256)
    customer_geo: str = Field(
        min_length=2,
        max_length=64,
        pattern=r"^[A-Z]{2}(?:-[A-Z0-9]{1,12}){0,2}$",
        description="Coarse geography only, for example IN-DL or IN-KA-BLR. Never raw GPS.",
    )
    client_event_at: datetime | None = None

    @field_validator("client_event_at")
    @classmethod
    def _normalize_client_time(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("client_event_at must include a timezone")
        return value.astimezone(UTC)


class FirstPartyTelemetryReceipt(StrictModel):
    telemetry_id: UUID
    account_id: str
    order_id: str
    payment_id: str | None
    customer_geo: str
    observed_at: datetime
    device_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_contract_version: str = SOURCE_CONTRACT_VERSION


class FeatureContractPreview(StrictModel):
    provider: str = "RAZORPAY"
    account_id: str
    as_of: datetime
    current_window_start: datetime
    baseline_start: datetime
    baseline_end: datetime
    knowledge_time_policy: str = KNOWLEDGE_TIME_POLICY
    source_contract_version: str = SOURCE_CONTRACT_VERSION
    baseline_method_version: str = BASELINE_METHOD_VERSION
    auth_mapping_version: str = AUTH_MAPPING_VERSION
    chargeback_mapping_version: str = CHARGEBACK_MAPPING_VERSION
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    baseline_transactions: int
    baseline_active_days: int
    current_transactions: int
    telemetry_coverage_baseline: float | None = Field(default=None, ge=0, le=1)
    telemetry_coverage_current: float | None = Field(default=None, ge=0, le=1)
    blockers: list[str]
    feature_vector: dict[str, float] | None = None
    feature_vector_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    shadow_score_eligible: bool = False
    production_action_eligible: bool = False
    production_action_blocker: str = (
        "CHAMPION_IS_SYNTHETIC_MECHANICS_RELEASE; LIVE ACTION REQUIRES SHADOW VALIDATION"
    )


class FirstPartyTelemetryRow(Base):
    __tablename__ = "first_party_transaction_telemetry"

    telemetry_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    account_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    order_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    payment_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    device_fingerprint_sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    customer_geo: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    client_event_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    payload_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    source_contract_version: Mapped[str] = mapped_column(String(64), nullable=False)


def hash_device_pseudonym(account_id: str, pseudonym: str) -> str:
    material = f"{DEVICE_PSEUDONYM_VERSION}|{account_id}|{pseudonym}".encode()
    return hashlib.sha256(material).hexdigest()


def map_auth_status(payment_status: str) -> Literal["APPROVED", "FAILED"] | None:
    normalized = payment_status.strip().lower()
    if normalized in {"authorized", "captured", "refunded"}:
        return "APPROVED"
    if normalized == "failed":
        return "FAILED"
    return None


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


class SqlLiveFeatureStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def submit_telemetry(
        self, submission: FirstPartyTelemetrySubmission
    ) -> FirstPartyTelemetryReceipt:
        now = datetime.now(UTC)
        if submission.client_event_at is not None and submission.client_event_at > now + timedelta(
            minutes=5
        ):
            raise ValueError("client_event_at cannot be more than five minutes in the future")
        device_hash = hash_device_pseudonym(submission.account_id, submission.device_pseudonym)
        safe_payload = {
            "account_id": submission.account_id,
            "order_id": submission.order_id,
            "payment_id": submission.payment_id,
            "device_fingerprint_sha256": device_hash,
            "customer_geo": submission.customer_geo,
            "client_event_at": submission.client_event_at.isoformat()
            if submission.client_event_at
            else None,
            "observed_at": now.isoformat(),
            "source_contract_version": SOURCE_CONTRACT_VERSION,
        }
        async with self._sessions() as session:
            account = await session.get(MerchantProcessorAccountRow, submission.account_id)
            if account is None:
                raise LookupError("processor account is not known")
            if submission.payment_id is not None:
                payment = await session.get(ProcessorPaymentRow, submission.payment_id)
                if payment is not None:
                    if payment.account_id != submission.account_id:
                        raise ValueError("payment_id is not owned by account_id")
                    if payment.order_id and payment.order_id != submission.order_id:
                        raise ValueError("order_id does not match authoritative payment")
            row = FirstPartyTelemetryRow(
                telemetry_id=uuid4(),
                account_id=submission.account_id,
                order_id=submission.order_id,
                payment_id=submission.payment_id,
                device_fingerprint_sha256=device_hash,
                customer_geo=submission.customer_geo,
                client_event_at=submission.client_event_at,
                observed_at=now,
                payload_sha256=_canonical_sha256(safe_payload),
                source_contract_version=SOURCE_CONTRACT_VERSION,
            )
            session.add(row)
            await session.commit()
            return FirstPartyTelemetryReceipt(
                telemetry_id=row.telemetry_id,
                account_id=row.account_id,
                order_id=row.order_id,
                payment_id=row.payment_id,
                customer_geo=row.customer_geo,
                observed_at=row.observed_at,
                device_fingerprint_sha256=row.device_fingerprint_sha256,
            )

    async def feature_contract_preview(
        self, *, account_id: str, as_of: datetime
    ) -> FeatureContractPreview:
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("as_of must include a timezone")
        as_of = as_of.astimezone(UTC)
        current_start = as_of - timedelta(hours=CURRENT_WINDOW_HOURS)
        baseline_end = current_start
        baseline_start = baseline_end - timedelta(days=BASELINE_DAYS)

        async with self._sessions() as session:
            payment_rows = list(
                (
                    await session.scalars(
                        select(ProcessorPaymentObservationRow).where(
                            ProcessorPaymentObservationRow.account_id == account_id,
                            ProcessorPaymentObservationRow.observed_at <= as_of,
                            ProcessorPaymentObservationRow.provider_created_at >= baseline_start,
                            ProcessorPaymentObservationRow.provider_created_at <= as_of,
                        )
                    )
                ).all()
            )
            refund_rows = list(
                (
                    await session.scalars(
                        select(ProcessorRefundObservationRow).where(
                            ProcessorRefundObservationRow.account_id == account_id,
                            ProcessorRefundObservationRow.observed_at <= as_of,
                        )
                    )
                ).all()
            )
            dispute_rows = list(
                (
                    await session.scalars(
                        select(ProcessorDisputeObservationRow).where(
                            ProcessorDisputeObservationRow.account_id == account_id,
                            ProcessorDisputeObservationRow.observed_at <= as_of,
                        )
                    )
                ).all()
            )
            payment_current_rows = list(
                (
                    await session.scalars(
                        select(ProcessorPaymentRow).where(
                            ProcessorPaymentRow.account_id == account_id
                        )
                    )
                ).all()
            )
            telemetry_rows = list(
                (
                    await session.scalars(
                        select(FirstPartyTelemetryRow).where(
                            FirstPartyTelemetryRow.account_id == account_id,
                            FirstPartyTelemetryRow.observed_at <= as_of,
                        )
                    )
                ).all()
            )

        order_by_payment = {row.payment_id: row.order_id for row in payment_current_rows}
        return build_feature_contract_preview(
            account_id=account_id,
            as_of=as_of,
            payment_rows=payment_rows,
            refund_rows=refund_rows,
            dispute_rows=dispute_rows,
            telemetry_rows=telemetry_rows,
            order_by_payment=order_by_payment,
        )


def _latest_by(
    rows: Sequence[object], key_attr: str, observed_attr: str = "observed_at"
) -> list[object]:
    latest: dict[str, object] = {}
    for row in sorted(rows, key=lambda item: getattr(item, observed_attr)):
        latest[str(getattr(row, key_attr))] = row
    return list(latest.values())


def _telemetry_maps(rows: Sequence[object]) -> tuple[dict[str, object], dict[str, object]]:
    by_order: dict[str, object] = {}
    by_payment: dict[str, object] = {}
    for row in sorted(rows, key=lambda item: item.observed_at):
        by_order[row.order_id] = row
        if row.payment_id:
            by_payment[row.payment_id] = row
    return by_order, by_payment


def _status_timestamp_by_payment(
    rows: Sequence[object], *, status: str | None = None
) -> dict[str, datetime]:
    result: dict[str, datetime] = {}
    for row in sorted(rows, key=lambda item: item.observed_at):
        if status is not None and str(getattr(row, "status", "")).lower() != status:
            continue
        timestamp = getattr(row, "provider_created_at", None) or row.observed_at
        result.setdefault(row.payment_id, timestamp)
    return result


def _make_events(
    *,
    account_id: str,
    payment_rows: Sequence[object],
    telemetry_rows: Sequence[object],
    order_by_payment: Mapping[str, str | None],
    refund_rows: Sequence[object],
    dispute_rows: Sequence[object],
) -> tuple[list[TransactionEvent], int]:
    latest_payments = _latest_by(payment_rows, "payment_id")
    by_order, by_payment = _telemetry_maps(telemetry_rows)
    refund_timestamps = _status_timestamp_by_payment(refund_rows, status="processed")
    chargeback_timestamps = _status_timestamp_by_payment(dispute_rows)
    events: list[TransactionEvent] = []
    telemetry_matches = 0
    for row in latest_payments:
        if row.provider_created_at is None or row.currency != "INR":
            continue
        auth_status = map_auth_status(row.status)
        if auth_status is None:
            continue
        order_id = getattr(row, "order_id", None) or order_by_payment.get(row.payment_id)
        telemetry = by_payment.get(row.payment_id)
        if telemetry is None and order_id:
            telemetry = by_order.get(order_id)
        if telemetry is None:
            continue
        telemetry_matches += 1
        events.append(
            TransactionEvent(
                transaction_id=row.payment_id,
                merchant_id=account_id,
                timestamp=row.provider_created_at,
                amount=row.amount / 100.0,
                device_fingerprint=telemetry.device_fingerprint_sha256,
                customer_geo=telemetry.customer_geo,
                auth_status=auth_status,
                refund_timestamp=refund_timestamps.get(row.payment_id),
                chargeback_timestamp=chargeback_timestamps.get(row.payment_id),
            )
        )
    return events, telemetry_matches


def _build_baseline(
    events: list[TransactionEvent], *, start: datetime, end: datetime
) -> tuple[MerchantBaseline, int]:
    by_day: dict[datetime.date, list[TransactionEvent]] = defaultdict(list)
    for event in events:
        if start <= event.timestamp < end:
            by_day[event.timestamp.date()].append(event)
    baseline_events = [event for day_events in by_day.values() for event in day_events]
    if not baseline_events:
        raise ValueError("baseline has no transactions")

    day_count = BASELINE_DAYS
    volumes: list[float] = []
    gmvs: list[float] = []
    cursor = start.date()
    for offset in range(day_count):
        day = cursor + timedelta(days=offset)
        day_events = by_day.get(day, [])
        volumes.append(float(len(day_events)))
        gmvs.append(sum(event.amount for event in day_events))

    amounts = [event.amount for event in baseline_events]
    ticket_mean = fmean(amounts)
    refund_rates = [
        sum(
            event.refund_timestamp is not None and event.refund_timestamp < end
            for event in day_events
        )
        / len(day_events)
        for day_events in by_day.values()
        if day_events
    ]
    chargeback_rates = [
        sum(
            event.chargeback_timestamp is not None and event.chargeback_timestamp < end
            for event in day_events
        )
        / len(day_events)
        for day_events in by_day.values()
        if day_events
    ]
    median_amount = max(1.0, median(amounts))
    edges = [0.0, median_amount * 0.5, median_amount, median_amount * 2.0, median_amount * 6.0]
    counts = [0, 0, 0, 0]
    for amount in amounts:
        if amount < edges[1]:
            counts[0] += 1
        elif amount < edges[2]:
            counts[1] += 1
        elif amount < edges[3]:
            counts[2] += 1
        else:
            counts[3] += 1
    smoothing = 1e-6
    total = sum(counts) + smoothing * 4
    probabilities = [(count + smoothing) / total for count in counts]

    volume_mean = fmean(volumes)
    gmv_mean = fmean(gmvs)
    refund_mean = fmean(refund_rates) if refund_rates else 0.0
    chargeback_mean = fmean(chargeback_rates) if chargeback_rates else 0.0
    return (
        MerchantBaseline(
            volume_mean=max(volume_mean, 1e-6),
            volume_std=max(pstdev(volumes), 5.0, volume_mean * 0.15),
            gmv_mean=max(gmv_mean, 1e-6),
            gmv_std=max(pstdev(gmvs), 500.0, gmv_mean * 0.20),
            ticket_size_mean=ticket_mean,
            ticket_size_std=max(
                pstdev(amounts) if len(amounts) > 1 else 0.0, 50.0, ticket_mean * 0.25
            ),
            refund_rate_mean=refund_mean,
            refund_rate_std=max(pstdev(refund_rates) if len(refund_rates) > 1 else 0.0, 0.015),
            chargeback_rate_mean=chargeback_mean,
            chargeback_rate_std=max(
                pstdev(chargeback_rates) if len(chargeback_rates) > 1 else 0.0, 0.005
            ),
            known_devices={event.device_fingerprint for event in baseline_events},
            known_geos={event.customer_geo for event in baseline_events},
            amount_bin_edges=edges,
            amount_bin_probabilities=probabilities,
        ),
        len(by_day),
    )


def build_feature_contract_preview(
    *,
    account_id: str,
    as_of: datetime,
    payment_rows: Sequence[object],
    refund_rows: Sequence[object],
    dispute_rows: Sequence[object],
    telemetry_rows: Sequence[object],
    order_by_payment: Mapping[str, str | None],
) -> FeatureContractPreview:
    as_of = as_of.astimezone(UTC)
    current_start = as_of - timedelta(hours=CURRENT_WINDOW_HOURS)
    baseline_end = current_start
    baseline_start = baseline_end - timedelta(days=BASELINE_DAYS)

    relevant_payment_rows = [
        row
        for row in payment_rows
        if row.provider_created_at is not None
        and baseline_start <= row.provider_created_at <= as_of
    ]
    latest_all = _latest_by(relevant_payment_rows, "payment_id")
    baseline_payments = [
        row for row in latest_all if baseline_start <= row.provider_created_at < baseline_end
    ]
    current_payments = [
        row for row in latest_all if current_start <= row.provider_created_at <= as_of
    ]

    all_events, matched = _make_events(
        account_id=account_id,
        payment_rows=relevant_payment_rows,
        telemetry_rows=telemetry_rows,
        order_by_payment=order_by_payment,
        refund_rows=refund_rows,
        dispute_rows=dispute_rows,
    )
    baseline_events = [
        event for event in all_events if baseline_start <= event.timestamp < baseline_end
    ]
    current_events = [event for event in all_events if current_start <= event.timestamp <= as_of]

    baseline_coverage = (
        None if not baseline_payments else len(baseline_events) / len(baseline_payments)
    )
    current_coverage = None if not current_payments else len(current_events) / len(current_payments)
    blockers: list[str] = []
    if len(baseline_payments) < MIN_BASELINE_TRANSACTIONS:
        blockers.append("INSUFFICIENT_BASELINE_TRANSACTIONS")
    active_days = len({row.provider_created_at.date() for row in baseline_payments})
    if active_days < MIN_BASELINE_ACTIVE_DAYS:
        blockers.append("INSUFFICIENT_BASELINE_ACTIVE_DAYS")
    if baseline_coverage is not None and baseline_coverage < 1.0:
        blockers.append("BASELINE_TELEMETRY_INCOMPLETE")
    if current_coverage is not None and current_coverage < 1.0:
        blockers.append("CURRENT_WINDOW_TELEMETRY_INCOMPLETE")
    if not current_payments:
        blockers.append("NO_CURRENT_WINDOW_PAYMENTS")
    unsupported_auth = [row.payment_id for row in latest_all if map_auth_status(row.status) is None]
    if unsupported_auth:
        blockers.append("UNMAPPED_PAYMENT_AUTH_STATUS")

    feature_vector = None
    feature_hash = None
    if not blockers:
        baseline, active_days = _build_baseline(all_events, start=baseline_start, end=baseline_end)
        hold = HoldCase(
            hold_id=uuid4(),
            request_id=uuid4(),
            merchant_id=account_id,
            source_event_id=f"live-feature-preview:{as_of.isoformat()}",
            triggered_at=as_of,
            reason_code="LIVE_FEATURE_PREVIEW",
        )
        evaluation = HoldEvaluationInput(
            baseline=baseline,
            transactions=all_events,
            cohort="pooled",
            window_hours=CURRENT_WINDOW_HOURS,
        )
        vector = build_point_in_time_features(hold, evaluation)
        dumped = vector.model_dump()
        if tuple(dumped) != FEATURE_COLUMNS:
            raise RuntimeError("live feature builder violated the locked 13-feature contract")
        feature_vector = dumped
        feature_hash = _canonical_sha256(
            {
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "as_of": as_of.isoformat(),
                "features": dumped,
                "source_contract_version": SOURCE_CONTRACT_VERSION,
                "baseline_method_version": BASELINE_METHOD_VERSION,
            }
        )

    return FeatureContractPreview(
        account_id=account_id,
        as_of=as_of,
        current_window_start=current_start,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        baseline_transactions=len(baseline_payments),
        baseline_active_days=active_days,
        current_transactions=len(current_payments),
        telemetry_coverage_baseline=baseline_coverage,
        telemetry_coverage_current=current_coverage,
        blockers=blockers,
        feature_vector=feature_vector,
        feature_vector_sha256=feature_hash,
        shadow_score_eligible=feature_vector is not None,
        production_action_eligible=False,
    )
