from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import UUID, uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator


class RazorpayWebhookError(ValueError):
    """Raised when a Razorpay webhook envelope is malformed."""


class RazorpayApiError(RuntimeError):
    """Sanitised Razorpay API failure with explicit retry semantics."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class RazorpayWebhookEnvelope(BaseModel):
    """Minimal Razorpay webhook contract used by the ingestion gateway.

    The full provider payload can contain customer and instrument data that is not
    required by the settlement-risk pipeline. RazorTrust validates the envelope,
    hashes the raw body, and persists only a privacy-minimised event summary.
    """

    model_config = ConfigDict(extra="allow")

    entity: str = "event"
    account_id: str | None = Field(default=None, max_length=128)
    event: str = Field(min_length=1, max_length=128)
    contains: list[str] = Field(default_factory=list, max_length=20)
    created_at: int | None = Field(default=None, ge=0)
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("entity")
    @classmethod
    def validate_entity(cls, value: str) -> str:
        if value != "event":
            raise ValueError("Razorpay webhook entity must be 'event'")
        return value

    @field_validator("contains")
    @classmethod
    def validate_contains(cls, value: list[str]) -> list[str]:
        if any(not item or len(item) > 128 for item in value):
            raise ValueError("Razorpay contains entries must be non-empty and <= 128 chars")
        return value

    def event_created_at(self) -> datetime | None:
        if self.created_at is None:
            return None
        return datetime.fromtimestamp(self.created_at, tz=UTC)


class RazorpayStoredEvent(BaseModel):
    event_id: UUID
    provider_event_id: str
    event_type: str
    account_id: str | None
    event_created_at: datetime | None
    payload_sha256: str
    summary: dict[str, Any]
    received_at: datetime
    processing_status: str
    processing_attempts: int = Field(default=0, ge=0)


class RazorpayIngestionStats(BaseModel):
    total_events: int
    last_event_at: datetime | None = None
    last_event_type: str | None = None


class RazorpayEventStore(Protocol):
    async def healthcheck(self) -> None: ...

    async def store(
        self,
        *,
        provider_event_id: str,
        envelope: RazorpayWebhookEnvelope,
        payload_sha256: str,
        summary: dict[str, Any],
        received_at: datetime,
    ) -> tuple[RazorpayStoredEvent, bool]: ...

    async def stats(self) -> RazorpayIngestionStats: ...


class InMemoryRazorpayEventStore:
    """Development-only event store used when no database is configured."""

    def __init__(self) -> None:
        self._events: dict[str, RazorpayStoredEvent] = {}

    async def healthcheck(self) -> None:
        return None

    async def store(
        self,
        *,
        provider_event_id: str,
        envelope: RazorpayWebhookEnvelope,
        payload_sha256: str,
        summary: dict[str, Any],
        received_at: datetime,
    ) -> tuple[RazorpayStoredEvent, bool]:
        existing = self._events.get(provider_event_id)
        if existing is not None:
            return existing.model_copy(deep=True), False
        record = RazorpayStoredEvent(
            event_id=uuid4(),
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
        self._events[provider_event_id] = record
        return record.model_copy(deep=True), True

    async def stats(self) -> RazorpayIngestionStats:
        if not self._events:
            return RazorpayIngestionStats(total_events=0)
        latest = max(self._events.values(), key=lambda item: item.received_at)
        return RazorpayIngestionStats(
            total_events=len(self._events),
            last_event_at=latest.received_at,
            last_event_type=latest.event_type,
        )


def verify_webhook_signature(raw_body: bytes, signature: str, secret: str) -> bool:
    """Validate Razorpay's HMAC-SHA256 signature over the *raw* request body."""

    if not raw_body or not signature or not secret:
        return False
    expected = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature.strip())


def payload_sha256(raw_body: bytes) -> str:
    return hashlib.sha256(raw_body).hexdigest()


def build_event_summary(envelope: RazorpayWebhookEnvelope) -> dict[str, Any]:
    """Build a privacy-minimised summary without customer identifiers or card data."""

    summary: dict[str, Any] = {
        "account_id": envelope.account_id,
        "event": envelope.event,
        "contains": envelope.contains,
        "created_at": envelope.created_at,
    }
    primary_type, primary = _primary_entity(envelope)
    if primary_type is None or primary is None:
        return summary

    summary["primary_entity_type"] = primary_type
    safe_keys = (
        "id",
        "entity",
        "amount",
        "base_amount",
        "currency",
        "status",
        "method",
        "order_id",
        "payment_id",
        "captured",
        "amount_refunded",
        "refund_status",
        "international",
        "created_at",
        "settlement_id",
        "dispute_id",
        "fees",
        "tax",
        "utr",
        "amount_deducted",
        "reason_code",
        "respond_by",
        "phase",
        "speed_processed",
        "speed_requested",
    )
    summary["primary_entity"] = {key: primary.get(key) for key in safe_keys if key in primary}

    # Persist only provider entity references needed for deterministic reconstruction.
    # This deliberately excludes customer contact, VPA, card and acquirer payloads.
    entity_refs: dict[str, dict[str, str]] = {}
    for entity_type, wrapped in envelope.payload.items():
        if not isinstance(wrapped, dict):
            continue
        entity = wrapped.get("entity")
        if not isinstance(entity, dict):
            continue
        refs: dict[str, str] = {}
        for key in ("id", "payment_id", "order_id", "settlement_id", "dispute_id"):
            value = entity.get(key)
            if isinstance(value, str) and 0 < len(value) <= 128:
                refs[key] = value
        if refs:
            entity_refs[str(entity_type)] = refs
    if entity_refs:
        summary["entity_refs"] = entity_refs
    return summary


def _primary_entity(envelope: RazorpayWebhookEnvelope) -> tuple[str | None, dict[str, Any] | None]:
    candidates = list(envelope.contains) + [
        key for key in envelope.payload if key not in envelope.contains
    ]
    for entity_type in candidates:
        wrapped = envelope.payload.get(entity_type)
        if not isinstance(wrapped, dict):
            continue
        entity = wrapped.get("entity")
        if isinstance(entity, dict):
            return entity_type, entity
    return None, None


@dataclass(frozen=True, slots=True)
class RazorpayClient:
    """Small async Razorpay REST client for authoritative enrichment reads."""

    key_id: str
    key_secret: str
    base_url: str = "https://api.razorpay.com/v1"
    timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        if not self.key_id or not self.key_secret:
            raise ValueError("Razorpay API key id and secret are required")
        if not self.base_url.startswith("https://"):
            raise ValueError("Razorpay base URL must use HTTPS")

    async def create_order(
        self,
        *,
        amount: int,
        currency: str,
        receipt: str,
        notes: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if amount <= 0:
            raise ValueError("order amount must be positive")
        if currency != "INR":
            raise ValueError("R4C checkout currently supports INR only")
        if not receipt or len(receipt) > 40:
            raise ValueError("receipt must be between 1 and 40 characters")
        payload: dict[str, Any] = {
            "amount": amount,
            "currency": currency,
            "receipt": receipt,
        }
        if notes:
            payload["notes"] = notes
        return await self._post("/orders", json_body=payload)

    async def fetch_order(self, order_id: str) -> dict[str, Any]:
        return await self._get(f"/orders/{_safe_provider_id(order_id)}")

    async def fetch_payment(self, payment_id: str) -> dict[str, Any]:
        return await self._get(f"/payments/{_safe_provider_id(payment_id)}")

    async def fetch_refund(self, refund_id: str) -> dict[str, Any]:
        return await self._get(f"/refunds/{_safe_provider_id(refund_id)}")

    async def fetch_settlement(self, settlement_id: str) -> dict[str, Any]:
        return await self._get(f"/settlements/{_safe_provider_id(settlement_id)}")

    async def fetch_dispute(self, dispute_id: str) -> dict[str, Any]:
        return await self._get(f"/disputes/{_safe_provider_id(dispute_id)}")

    async def fetch_settlement_recon(
        self,
        *,
        year: int,
        month: int,
        day: int | None = None,
        count: int = 100,
        skip: int = 0,
    ) -> dict[str, Any]:
        if not 2000 <= year <= 2200:
            raise ValueError("year is outside the supported range")
        if not 1 <= month <= 12:
            raise ValueError("month must be between 1 and 12")
        if day is not None and not 1 <= day <= 31:
            raise ValueError("day must be between 1 and 31")
        if not 1 <= count <= 1000:
            raise ValueError("count must be between 1 and 1000")
        if skip < 0:
            raise ValueError("skip cannot be negative")
        params: dict[str, int] = {"year": year, "month": month, "count": count, "skip": skip}
        if day is not None:
            params["day"] = day
        return await self._get("/settlements/recon/combined", params=params)

    async def _get(self, path: str, params: dict[str, int] | None = None) -> dict[str, Any]:
        return await self._request("GET", path, params=params)

    async def _post(self, path: str, *, json_body: dict[str, Any]) -> dict[str, Any]:
        return await self._request("POST", path, json_body=json_body)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, int] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}{path}"
        try:
            async with httpx.AsyncClient(
                auth=(self.key_id, self.key_secret), timeout=self.timeout_seconds
            ) as client:
                response = await client.request(method, url, params=params, json=json_body)
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            retryable = status_code == 429 or status_code >= 500
            raise RazorpayApiError(
                "Razorpay API returned a non-success status",
                status_code=status_code,
                retryable=retryable,
            ) from exc
        except httpx.RequestError as exc:
            raise RazorpayApiError("Razorpay API transport failed", retryable=True) from exc
        except ValueError as exc:
            raise RazorpayApiError("Razorpay API returned invalid JSON", retryable=False) from exc
        if not isinstance(payload, dict):
            raise RazorpayApiError("Razorpay API response must be an object", retryable=False)
        return payload


def _safe_provider_id(value: str) -> str:
    if not value or len(value) > 128:
        raise ValueError("provider id must be between 1 and 128 characters")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    if any(character not in allowed for character in value):
        raise ValueError("provider id contains unsupported characters")
    return value
