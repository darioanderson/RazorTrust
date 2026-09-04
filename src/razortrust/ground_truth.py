from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any, Protocol

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .domain import HoldCase, StrictModel

_PAYMENT_RE = re.compile(r"^pay_[A-Za-z0-9]+$")
_ORDER_RE = re.compile(r"^order_[A-Za-z0-9]+$")
_REFUND_RE = re.compile(r"^rfnd_[A-Za-z0-9]+$")
_DISPUTE_RE = re.compile(r"^disp_[A-Za-z0-9]+$")


class ProviderGroundTruth(StrictModel):
    provider: str = "RAZORPAY"
    case_link_status: str
    provider_outcome: str
    payment_id: str | None = None
    order_id: str | None = None
    payment: dict[str, Any] | None = None
    refunds: list[dict[str, Any]] = []
    disputes: list[dict[str, Any]] = []
    refund_count: int = 0
    refund_amount_total: int = 0
    dispute_count: int = 0
    dispute_confirmed: bool = False
    fraud_confirmed: bool = False
    authoritative_store_backed: bool = False
    decision_time: datetime
    truth_observed_through: datetime
    evaluation_only: bool = True
    used_for_model_scoring: bool = False
    used_for_policy: bool = False
    ground_truth_sha256: str
    notes: list[str] = []


class ProviderGroundTruthStore(Protocol):
    async def for_hold(self, hold: HoldCase) -> ProviderGroundTruth: ...


class ProviderGroundTruthUnavailable(RuntimeError):
    pass


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        _jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _extract_ids(value: Any) -> tuple[set[str], set[str]]:
    payments: set[str] = set()
    orders: set[str] = set()

    def walk(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                key_lower = str(key).lower()
                if isinstance(child, str):
                    if key_lower in {"payment_id", "payment"} and _PAYMENT_RE.fullmatch(child):
                        payments.add(child)
                    if key_lower in {"order_id", "order"} and _ORDER_RE.fullmatch(child):
                        orders.add(child)
                walk(child)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                walk(child)
            return
        if isinstance(item, str):
            if _PAYMENT_RE.fullmatch(item):
                payments.add(item)
            elif _ORDER_RE.fullmatch(item):
                orders.add(item)

    walk(value)
    return payments, orders


def _classify_outcome(
    payment: dict[str, Any] | None,
    refunds: list[dict[str, Any]],
    disputes: list[dict[str, Any]],
) -> tuple[str, bool, bool]:
    if disputes:
        fraud = any(str(row.get("phase") or "").lower() == "fraud" for row in disputes)
        if fraud:
            return "FRAUD_DISPUTE_CONFIRMED", True, True
        return "DISPUTE_CONFIRMED", True, False

    if refunds:
        return "REFUND_OBSERVED", False, False

    if payment is None:
        return "NOT_YET_OBSERVED", False, False

    if int(payment.get("amount_refunded") or 0) > 0:
        return "REFUND_OBSERVED", False, False

    status = str(payment.get("status") or "").strip().lower()
    if status == "failed":
        return "PAYMENT_FAILED", False, False
    if status == "captured":
        return "PAYMENT_CAPTURED", False, False
    if status == "authorized":
        return "PAYMENT_AUTHORIZED", False, False
    if status == "refunded":
        return "REFUND_OBSERVED", False, False
    return "PAYMENT_OBSERVED", False, False


def _payment_summary(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    keys = (
        "payment_id",
        "order_id",
        "amount",
        "currency",
        "status",
        "method",
        "captured",
        "amount_refunded",
        "refund_status",
        "provider_created_at",
        "source_event_id",
        "source_event_created_at",
        "authoritative_sha256",
        "updated_at",
    )
    return {key: _jsonable(row.get(key)) for key in keys}


def _refund_summary(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "refund_id",
        "payment_id",
        "amount",
        "currency",
        "status",
        "speed_requested",
        "speed_processed",
        "provider_created_at",
        "source_event_id",
        "source_event_created_at",
        "authoritative_sha256",
        "updated_at",
    )
    return {key: _jsonable(row.get(key)) for key in keys}


def _dispute_summary(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "dispute_id",
        "payment_id",
        "amount",
        "currency",
        "amount_deducted",
        "reason_code",
        "status",
        "phase",
        "respond_by",
        "provider_created_at",
        "source_event_id",
        "source_event_created_at",
        "authoritative_sha256",
        "updated_at",
    )
    return {key: _jsonable(row.get(key)) for key in keys}


class SqlProviderGroundTruthStore:
    """Read post-decision provider outcomes without feeding them into scoring."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def for_hold(self, hold: HoldCase) -> ProviderGroundTruth:
        observed_through = datetime.now(UTC)
        try:
            async with self._sessions() as session:
                payment_id, order_id = await self._resolve_case_payment(
                    session,
                    account_id=hold.merchant_id,
                    source_event_id=hold.source_event_id,
                )

                if payment_id is None:
                    payload = {
                        "case_link_status": "UNLINKED",
                        "provider_outcome": "UNLINKED_CASE",
                        "account_id": hold.merchant_id,
                        "source_event_id": hold.source_event_id,
                        "decision_time": hold.triggered_at,
                    }
                    return ProviderGroundTruth(
                        case_link_status="UNLINKED",
                        provider_outcome="UNLINKED_CASE",
                        order_id=order_id,
                        decision_time=hold.triggered_at,
                        truth_observed_through=observed_through,
                        ground_truth_sha256=_canonical_sha256(payload),
                        notes=[
                            "No authoritative payment could be linked to this hold.",
                            "RazorTrust did not guess a merchant-level payment.",
                        ],
                    )

                payment = await self._payment(session, hold.merchant_id, payment_id)
                refunds = await self._refunds(session, hold.merchant_id, payment_id)
                disputes = await self._disputes(session, hold.merchant_id, payment_id)

                payment_view = _payment_summary(payment)
                refund_views = [_refund_summary(row) for row in refunds]
                dispute_views = [_dispute_summary(row) for row in disputes]
                outcome, dispute_confirmed, fraud_confirmed = _classify_outcome(
                    payment_view,
                    refund_views,
                    dispute_views,
                )
                if order_id is None and payment_view is not None:
                    raw_order = payment_view.get("order_id")
                    if isinstance(raw_order, str):
                        order_id = raw_order

                hashes = []
                for row in [payment_view, *refund_views, *dispute_views]:
                    if row is None:
                        continue
                    value = row.get("authoritative_sha256")
                    if isinstance(value, str) and value:
                        hashes.append(value)

                payload = {
                    "payment_id": payment_id,
                    "order_id": order_id,
                    "provider_outcome": outcome,
                    "payment": payment_view,
                    "refunds": refund_views,
                    "disputes": dispute_views,
                    "decision_time": hold.triggered_at,
                }
                return ProviderGroundTruth(
                    case_link_status="LINKED",
                    provider_outcome=outcome,
                    payment_id=payment_id,
                    order_id=order_id,
                    payment=payment_view,
                    refunds=refund_views,
                    disputes=dispute_views,
                    refund_count=len(refund_views),
                    refund_amount_total=sum(
                        int(row.get("amount") or 0) for row in refund_views
                    ),
                    dispute_count=len(dispute_views),
                    dispute_confirmed=dispute_confirmed,
                    fraud_confirmed=fraud_confirmed,
                    authoritative_store_backed=bool(hashes),
                    decision_time=hold.triggered_at,
                    truth_observed_through=observed_through,
                    ground_truth_sha256=_canonical_sha256(payload),
                    notes=[
                        "Post-decision evaluation only; never supplied to model or OPA.",
                        (
                            "Fraud is confirmed only when an authoritative dispute "
                            "has phase=fraud."
                        ),
                    ],
                )
        except SQLAlchemyError as exc:
            raise ProviderGroundTruthUnavailable(
                f"provider ground-truth store unavailable: {type(exc).__name__}"
            ) from exc

    async def _resolve_case_payment(
        self,
        session: AsyncSession,
        *,
        account_id: str,
        source_event_id: str,
    ) -> tuple[str | None, str | None]:
        if _PAYMENT_RE.fullmatch(source_event_id):
            return source_event_id, None

        direct = await session.execute(
            text(
                """
                SELECT payment_id, order_id
                FROM processor_payments
                WHERE account_id = :account_id
                  AND (
                    source_event_id = :source_event_id
                    OR order_id = :source_event_id
                    OR payment_id = :source_event_id
                  )
                ORDER BY updated_at DESC
                LIMIT 1
                """
            ),
            {"account_id": account_id, "source_event_id": source_event_id},
        )
        direct_row = direct.mappings().first()
        if direct_row is not None:
            return str(direct_row["payment_id"]), direct_row.get("order_id")

        for table_name, id_column, pattern in (
            ("processor_refunds", "refund_id", _REFUND_RE),
            ("processor_disputes", "dispute_id", _DISPUTE_RE),
        ):
            if not pattern.fullmatch(source_event_id):
                continue
            linked = await session.execute(
                text(
                    f"""
                    SELECT payment_id
                    FROM {table_name}
                    WHERE account_id = :account_id
                      AND {id_column} = :source_event_id
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """
                ),
                {"account_id": account_id, "source_event_id": source_event_id},
            )
            linked_row = linked.mappings().first()
            if linked_row is not None:
                return str(linked_row["payment_id"]), None

        webhook = await session.execute(
            text(
                """
                SELECT summary
                FROM processor_webhook_events
                WHERE account_id = :account_id
                  AND provider_event_id = :source_event_id
                ORDER BY received_at DESC
                LIMIT 1
                """
            ),
            {"account_id": account_id, "source_event_id": source_event_id},
        )
        webhook_row = webhook.mappings().first()
        if webhook_row is None:
            return None, source_event_id if _ORDER_RE.fullmatch(source_event_id) else None

        summary = webhook_row.get("summary")
        if isinstance(summary, str):
            try:
                summary = json.loads(summary)
            except json.JSONDecodeError:
                summary = None
        payment_ids, order_ids = _extract_ids(summary)
        if len(payment_ids) == 1:
            return next(iter(payment_ids)), next(iter(order_ids), None)

        if len(order_ids) == 1:
            order_id = next(iter(order_ids))
            linked = await session.execute(
                text(
                    """
                    SELECT payment_id
                    FROM processor_payments
                    WHERE account_id = :account_id
                      AND order_id = :order_id
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """
                ),
                {"account_id": account_id, "order_id": order_id},
            )
            linked_row = linked.mappings().first()
            if linked_row is not None:
                return str(linked_row["payment_id"]), order_id
            return None, order_id

        return None, None

    async def _payment(
        self,
        session: AsyncSession,
        account_id: str,
        payment_id: str,
    ) -> dict[str, Any] | None:
        result = await session.execute(
            text(
                """
                SELECT *
                FROM processor_payments
                WHERE account_id = :account_id
                  AND payment_id = :payment_id
                LIMIT 1
                """
            ),
            {"account_id": account_id, "payment_id": payment_id},
        )
        row = result.mappings().first()
        return dict(row) if row is not None else None

    async def _refunds(
        self,
        session: AsyncSession,
        account_id: str,
        payment_id: str,
    ) -> list[dict[str, Any]]:
        result = await session.execute(
            text(
                """
                SELECT *
                FROM processor_refunds
                WHERE account_id = :account_id
                  AND payment_id = :payment_id
                ORDER BY COALESCE(updated_at, provider_created_at) DESC
                """
            ),
            {"account_id": account_id, "payment_id": payment_id},
        )
        return [dict(row) for row in result.mappings().all()]

    async def _disputes(
        self,
        session: AsyncSession,
        account_id: str,
        payment_id: str,
    ) -> list[dict[str, Any]]:
        result = await session.execute(
            text(
                """
                SELECT *
                FROM processor_disputes
                WHERE account_id = :account_id
                  AND payment_id = :payment_id
                ORDER BY COALESCE(updated_at, provider_created_at) DESC
                """
            ),
            {"account_id": account_id, "payment_id": payment_id},
        )
        return [dict(row) for row in result.mappings().all()]
