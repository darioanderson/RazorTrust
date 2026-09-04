from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

TIMELINE_RULES_VERSION = "processor-timeline-quality@1"
LOCKED_FEATURE_SCHEMA_VERSION = "1.0.0"


class TimelineModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class PaymentObservation(TimelineModel):
    payment_id: str
    status: str
    amount: int = Field(ge=0)
    currency: str
    method: str | None = None
    captured: bool
    amount_refunded: int = Field(ge=0)
    refund_status: str | None = None
    international: bool | None = None
    provider_created_at: datetime | None = None
    source_event_id: str
    observed_at: datetime


class RefundObservation(TimelineModel):
    refund_id: str
    payment_id: str
    amount: int = Field(ge=0)
    currency: str
    status: str
    provider_created_at: datetime | None = None
    source_event_id: str
    observed_at: datetime


class DisputeObservation(TimelineModel):
    dispute_id: str
    payment_id: str
    amount: int = Field(ge=0)
    currency: str
    status: str
    phase: str | None = None
    provider_created_at: datetime | None = None
    source_event_id: str
    observed_at: datetime


class SettlementObservation(TimelineModel):
    settlement_id: str
    amount: int = Field(ge=0)
    currency: str | None = None
    status: str
    fees: int = Field(ge=0)
    tax: int = Field(ge=0)
    provider_created_at: datetime | None = None
    source_event_id: str
    observed_at: datetime


class DataQualityIssue(TimelineModel):
    code: str
    severity: Literal["CRITICAL", "WARNING"]
    message: str


class DataQualityReport(TimelineModel):
    rules_version: str = TIMELINE_RULES_VERSION
    status: Literal["PASS", "PASS_WITH_WARNINGS", "BLOCKED"]
    critical_count: int = Field(ge=0)
    warning_count: int = Field(ge=0)
    issues: list[DataQualityIssue]


class FeatureReadiness(TimelineModel):
    feature_schema_version: str = LOCKED_FEATURE_SCHEMA_VERSION
    model_input_ready: bool
    available_source_dimensions: list[str]
    missing_source_dimensions: list[str]
    blockers: list[str]


class TimelineMetrics(TimelineModel):
    payment_count: int = Field(ge=0)
    captured_payment_count: int = Field(ge=0)
    failed_payment_count: int = Field(ge=0)
    gross_captured_amount: int = Field(ge=0)
    amount_refunded: int = Field(ge=0)
    refund_count: int = Field(ge=0)
    processed_refund_count: int = Field(ge=0)
    dispute_count: int = Field(ge=0)
    open_dispute_count: int = Field(ge=0)
    settlement_count: int = Field(ge=0)
    methods: dict[str, int]
    currencies: list[str]
    unresolved_pipeline_events: int = Field(ge=0)


class PointInTimeMerchantSnapshot(TimelineModel):
    provider: Literal["RAZORPAY"] = "RAZORPAY"
    account_id: str
    as_of: datetime
    window_start: datetime
    lookback_hours: int
    knowledge_time_policy: Literal["OBSERVED_AT"] = "OBSERVED_AT"
    payments: list[PaymentObservation]
    refunds: list[RefundObservation]
    disputes: list[DisputeObservation]
    settlements: list[SettlementObservation]
    metrics: TimelineMetrics
    data_quality: DataQualityReport
    feature_readiness: FeatureReadiness
    snapshot_sha256: str

    @field_validator("as_of", "window_start")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("point-in-time timestamps must include a timezone")
        return value.astimezone(UTC)


class ProcessorAccountSummary(TimelineModel):
    account_id: str
    first_seen_at: datetime
    last_seen_at: datetime
    last_payment_at: datetime | None = None
    last_refund_at: datetime | None = None
    last_settlement_at: datetime | None = None
    last_dispute_at: datetime | None = None
    last_event_type: str | None = None


def _latest_as_of(items: list[TimelineModel], *, key: str, as_of: datetime) -> list[TimelineModel]:
    latest: dict[str, TimelineModel] = {}
    for item in sorted(items, key=lambda value: value.observed_at, reverse=True):  # type: ignore[attr-defined]
        if item.observed_at > as_of:  # type: ignore[attr-defined]
            continue
        entity_id = str(getattr(item, key))
        if entity_id not in latest:
            latest[entity_id] = item
    return sorted(latest.values(), key=lambda value: value.observed_at)  # type: ignore[attr-defined]


def _in_event_window(item: TimelineModel, *, window_start: datetime, as_of: datetime) -> bool:
    event_time = item.provider_created_at
    return event_time is not None and window_start <= event_time <= as_of


def _quality_report(
    *,
    payments: list[PaymentObservation],
    refunds: list[RefundObservation],
    disputes: list[DisputeObservation],
    settlements: list[SettlementObservation],
    unresolved_pipeline_events: int,
    as_of: datetime,
) -> DataQualityReport:
    issues: list[DataQualityIssue] = []

    if unresolved_pipeline_events:
        issues.append(
            DataQualityIssue(
                code="UNRESOLVED_PIPELINE_EVENTS",
                severity="CRITICAL",
                message=(
                    f"{unresolved_pipeline_events} processor event(s) known by the cutoff are not "
                    "in a terminal processing state. Automatic scoring must stay blocked."
                ),
            )
        )

    all_items: list[TimelineModel] = [*payments, *refunds, *disputes, *settlements]
    missing_event_time = sum(item.provider_created_at is None for item in all_items)
    if missing_event_time:
        issues.append(
            DataQualityIssue(
                code="MISSING_PROVIDER_EVENT_TIME",
                severity="CRITICAL",
                message=(
                    f"{missing_event_time} authoritative observation(s) lack provider_created_at; "
                    "event-time features cannot be reconstructed faithfully."
                ),
            )
        )

    clock_skewed = sum(
        item.provider_created_at is not None
        and item.provider_created_at > item.observed_at + timedelta(minutes=5)  # type: ignore[attr-defined]
        for item in all_items
    )
    if clock_skewed:
        issues.append(
            DataQualityIssue(
                code="PROVIDER_TIMESTAMP_IN_FUTURE",
                severity="CRITICAL",
                message=(
                    f"{clock_skewed} observation(s) have provider timestamps more than five "
                    "minutes after the system knowledge timestamp."
                ),
            )
        )

    over_refunded = [item.payment_id for item in payments if item.amount_refunded > item.amount]
    if over_refunded:
        issues.append(
            DataQualityIssue(
                code="REFUND_EXCEEDS_PAYMENT",
                severity="CRITICAL",
                message=(
                    "Authoritative payment state reports amount_refunded greater than amount for: "
                    + ", ".join(over_refunded[:5])
                ),
            )
        )

    currencies = sorted(
        {item.currency for item in [*payments, *refunds, *disputes] if item.currency}
        | {item.currency for item in settlements if item.currency}
    )
    if len(currencies) > 1:
        issues.append(
            DataQualityIssue(
                code="MIXED_CURRENCIES",
                severity="CRITICAL",
                message=(
                    "The current feature contract has no FX-normalization policy; mixed currencies "
                    f"cannot be aggregated safely: {', '.join(currencies)}."
                ),
            )
        )
    elif currencies and currencies != ["INR"]:
        issues.append(
            DataQualityIssue(
                code="UNVALIDATED_CURRENCY",
                severity="CRITICAL",
                message=(
                    f"Currency {currencies[0]} has not been validated against the current INR "
                    "training/economic-cost assumptions."
                ),
            )
        )

    if not payments:
        issues.append(
            DataQualityIssue(
                code="NO_PAYMENT_HISTORY_IN_WINDOW",
                severity="WARNING",
                message="No point-in-time payment observations exist in the requested window.",
            )
        )

    if as_of > datetime.now(UTC) + timedelta(minutes=5):
        issues.append(
            DataQualityIssue(
                code="AS_OF_IN_FUTURE",
                severity="WARNING",
                message="The requested as_of timestamp is in the future relative to this service.",
            )
        )

    critical = sum(issue.severity == "CRITICAL" for issue in issues)
    warnings = sum(issue.severity == "WARNING" for issue in issues)
    status: Literal["PASS", "PASS_WITH_WARNINGS", "BLOCKED"]
    if critical:
        status = "BLOCKED"
    elif warnings:
        status = "PASS_WITH_WARNINGS"
    else:
        status = "PASS"
    return DataQualityReport(
        status=status,
        critical_count=critical,
        warning_count=warnings,
        issues=issues,
    )


def _feature_readiness(quality: DataQualityReport) -> FeatureReadiness:
    available = [
        "merchant/account identity",
        "payment timestamp",
        "payment amount",
        "payment/refund state",
        "payment method",
        "dispute observations",
        "settlement observations",
    ]
    missing = [
        "merchant baseline for z-score features",
        "device_fingerprint",
        "customer_geo",
        "auth_status with APPROVED/FAILED semantics",
        "frozen dispute-to-chargeback feature mapping",
    ]
    blockers = [
        "MERCHANT_BASELINE_NOT_BUILT",
        "DEVICE_FINGERPRINT_SOURCE_NOT_CONNECTED",
        "CUSTOMER_GEO_SOURCE_NOT_CONNECTED",
        "AUTH_STATUS_SOURCE_NOT_CONNECTED",
        "CHARGEBACK_MAPPING_NOT_FROZEN",
        "LOCKED_13_FEATURE_VECTOR_INCOMPLETE",
    ]
    if quality.status == "BLOCKED":
        blockers.insert(0, "DATA_QUALITY_FIREWALL_BLOCKED")
    return FeatureReadiness(
        model_input_ready=False,
        available_source_dimensions=available,
        missing_source_dimensions=missing,
        blockers=blockers,
    )


def build_point_in_time_snapshot(
    *,
    account_id: str,
    as_of: datetime,
    lookback_hours: int,
    payment_observations: list[PaymentObservation],
    refund_observations: list[RefundObservation],
    dispute_observations: list[DisputeObservation],
    settlement_observations: list[SettlementObservation],
    unresolved_pipeline_events: int,
) -> PointInTimeMerchantSnapshot:
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must include a timezone")
    as_of = as_of.astimezone(UTC)
    if not 1 <= lookback_hours <= 720:
        raise ValueError("lookback_hours must be between 1 and 720")
    window_start = as_of - timedelta(hours=lookback_hours)

    payments = [
        item
        for item in _latest_as_of(payment_observations, key="payment_id", as_of=as_of)
        if _in_event_window(item, window_start=window_start, as_of=as_of)
    ]
    refunds = [
        item
        for item in _latest_as_of(refund_observations, key="refund_id", as_of=as_of)
        if _in_event_window(item, window_start=window_start, as_of=as_of)
    ]
    disputes = [
        item
        for item in _latest_as_of(dispute_observations, key="dispute_id", as_of=as_of)
        if _in_event_window(item, window_start=window_start, as_of=as_of)
    ]
    settlements = [
        item
        for item in _latest_as_of(settlement_observations, key="settlement_id", as_of=as_of)
        if _in_event_window(item, window_start=window_start, as_of=as_of)
    ]

    methods = Counter(item.method or "unknown" for item in payments)
    currencies = sorted(
        {item.currency for item in [*payments, *refunds, *disputes] if item.currency}
        | {item.currency for item in settlements if item.currency}
    )
    metrics = TimelineMetrics(
        payment_count=len(payments),
        captured_payment_count=sum(item.captured for item in payments),
        failed_payment_count=sum(item.status == "failed" for item in payments),
        gross_captured_amount=sum(item.amount for item in payments if item.captured),
        amount_refunded=sum(item.amount_refunded for item in payments),
        refund_count=len(refunds),
        processed_refund_count=sum(item.status == "processed" for item in refunds),
        dispute_count=len(disputes),
        open_dispute_count=sum(item.status not in {"won", "lost", "closed"} for item in disputes),
        settlement_count=len(settlements),
        methods=dict(sorted(methods.items())),
        currencies=currencies,
        unresolved_pipeline_events=unresolved_pipeline_events,
    )
    quality = _quality_report(
        payments=payments,
        refunds=refunds,
        disputes=disputes,
        settlements=settlements,
        unresolved_pipeline_events=unresolved_pipeline_events,
        as_of=as_of,
    )
    readiness = _feature_readiness(quality)

    canonical = {
        "account_id": account_id,
        "as_of": as_of.isoformat(),
        "window_start": window_start.isoformat(),
        "lookback_hours": lookback_hours,
        "payments": [item.model_dump(mode="json") for item in payments],
        "refunds": [item.model_dump(mode="json") for item in refunds],
        "disputes": [item.model_dump(mode="json") for item in disputes],
        "settlements": [item.model_dump(mode="json") for item in settlements],
        "metrics": metrics.model_dump(mode="json"),
        "data_quality": quality.model_dump(mode="json"),
        "feature_readiness": readiness.model_dump(mode="json"),
    }
    snapshot_sha256 = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return PointInTimeMerchantSnapshot(
        account_id=account_id,
        as_of=as_of,
        window_start=window_start,
        lookback_hours=lookback_hours,
        payments=payments,
        refunds=refunds,
        disputes=disputes,
        settlements=settlements,
        metrics=metrics,
        data_quality=quality,
        feature_readiness=readiness,
        snapshot_sha256=snapshot_sha256,
    )


class SqlPointInTimeTimelineStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def list_accounts(self, *, limit: int = 100) -> list[ProcessorAccountSummary]:
        from .razorpay_store import MerchantProcessorAccountRow

        async with self._sessions() as session:
            rows = list(
                await session.scalars(
                    select(MerchantProcessorAccountRow)
                    .order_by(MerchantProcessorAccountRow.last_seen_at.desc())
                    .limit(limit)
                )
            )
        return [
            ProcessorAccountSummary(
                account_id=row.provider_account_id,
                first_seen_at=row.first_seen_at,
                last_seen_at=row.last_seen_at,
                last_payment_at=row.last_payment_at,
                last_refund_at=row.last_refund_at,
                last_settlement_at=row.last_settlement_at,
                last_dispute_at=row.last_dispute_at,
                last_event_type=row.last_event_type,
            )
            for row in rows
        ]

    async def snapshot(
        self, *, account_id: str, as_of: datetime, lookback_hours: int
    ) -> PointInTimeMerchantSnapshot:
        as_of = as_of.astimezone(UTC)
        window_start = as_of - timedelta(hours=lookback_hours)
        from .razorpay_store import (
            ProcessorDisputeObservationRow,
            ProcessorPaymentObservationRow,
            ProcessorRefundObservationRow,
            ProcessorSettlementObservationRow,
            ProcessorWebhookEventRow,
        )

        # observed_at is the online knowledge-time boundary. We deliberately do not read
        # the mutable current-state tables because doing so would leak later refunds/disputes
        # into an earlier decision.
        async with self._sessions() as session:
            payment_rows = list(
                await session.scalars(
                    select(ProcessorPaymentObservationRow).where(
                        ProcessorPaymentObservationRow.account_id == account_id,
                        ProcessorPaymentObservationRow.observed_at <= as_of,
                        ProcessorPaymentObservationRow.observed_at >= window_start,
                    )
                )
            )
            refund_rows = list(
                await session.scalars(
                    select(ProcessorRefundObservationRow).where(
                        ProcessorRefundObservationRow.account_id == account_id,
                        ProcessorRefundObservationRow.observed_at <= as_of,
                        ProcessorRefundObservationRow.observed_at >= window_start,
                    )
                )
            )
            dispute_rows = list(
                await session.scalars(
                    select(ProcessorDisputeObservationRow).where(
                        ProcessorDisputeObservationRow.account_id == account_id,
                        ProcessorDisputeObservationRow.observed_at <= as_of,
                        ProcessorDisputeObservationRow.observed_at >= window_start,
                    )
                )
            )
            settlement_rows = list(
                await session.scalars(
                    select(ProcessorSettlementObservationRow).where(
                        ProcessorSettlementObservationRow.account_id == account_id,
                        ProcessorSettlementObservationRow.observed_at <= as_of,
                        ProcessorSettlementObservationRow.observed_at >= window_start,
                    )
                )
            )
            unresolved = list(
                await session.scalars(
                    select(ProcessorWebhookEventRow).where(
                        ProcessorWebhookEventRow.account_id == account_id,
                        ProcessorWebhookEventRow.received_at <= as_of,
                        ProcessorWebhookEventRow.received_at >= window_start,
                        ProcessorWebhookEventRow.processing_status.not_in(["PROCESSED", "SKIPPED"]),
                    )
                )
            )

        return build_point_in_time_snapshot(
            account_id=account_id,
            as_of=as_of,
            lookback_hours=lookback_hours,
            payment_observations=[
                PaymentObservation(
                    payment_id=row.payment_id,
                    status=row.status,
                    amount=row.amount,
                    currency=row.currency,
                    method=row.method,
                    captured=row.captured,
                    amount_refunded=row.amount_refunded,
                    refund_status=row.refund_status,
                    international=row.international,
                    provider_created_at=row.provider_created_at,
                    source_event_id=row.source_event_id,
                    observed_at=row.observed_at,
                )
                for row in payment_rows
            ],
            refund_observations=[
                RefundObservation(
                    refund_id=row.refund_id,
                    payment_id=row.payment_id,
                    amount=row.amount,
                    currency=row.currency,
                    status=row.status,
                    provider_created_at=row.provider_created_at,
                    source_event_id=row.source_event_id,
                    observed_at=row.observed_at,
                )
                for row in refund_rows
            ],
            dispute_observations=[
                DisputeObservation(
                    dispute_id=row.dispute_id,
                    payment_id=row.payment_id,
                    amount=row.amount,
                    currency=row.currency,
                    status=row.status,
                    phase=row.phase,
                    provider_created_at=row.provider_created_at,
                    source_event_id=row.source_event_id,
                    observed_at=row.observed_at,
                )
                for row in dispute_rows
            ],
            settlement_observations=[
                SettlementObservation(
                    settlement_id=row.settlement_id,
                    amount=row.amount,
                    currency=row.currency,
                    status=row.status,
                    fees=row.fees,
                    tax=row.tax,
                    provider_created_at=row.provider_created_at,
                    source_event_id=row.source_event_id,
                    observed_at=row.observed_at,
                )
                for row in settlement_rows
            ],
            unresolved_pipeline_events=len(unresolved),
        )
