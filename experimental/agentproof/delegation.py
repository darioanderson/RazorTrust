from __future__ import annotations

from datetime import datetime
from uuid import UUID

import httpx
from pydantic import Field, field_validator, model_validator

from razortrust.domain import StrictModel
from razortrust.security import sign_manifest, verify_manifest


class DelegationMandate(StrictModel):
    delegation_id: UUID
    agent_id: str = Field(min_length=1, max_length=128)
    merchant_id: str = Field(min_length=1, max_length=128)
    action: str = Field(min_length=1, max_length=128)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    maximum_amount: float = Field(gt=0)
    valid_from: datetime
    valid_until: datetime
    nonce: str = Field(min_length=16, max_length=128)

    @field_validator("valid_from", "valid_until")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("delegation timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def require_positive_window(self) -> DelegationMandate:
        if self.valid_until <= self.valid_from:
            raise ValueError("delegation valid_until must be later than valid_from")
        return self


class SignedDelegation(StrictModel):
    mandate: DelegationMandate
    key_id: str = Field(min_length=1, max_length=128)
    signature_b64: str = Field(min_length=1)


class DelegationContext(StrictModel):
    agent_id: str
    merchant_id: str
    action: str
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    amount: float = Field(gt=0)
    occurred_at: datetime

    @field_validator("occurred_at")
    @classmethod
    def occurred_at_requires_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must include a timezone")
        return value


class DelegationDecision(StrictModel):
    permitted: bool
    reasons: list[str]
    delegation_id: UUID
    key_id: str
    policy_version: str = "delegation-signature@1"


class DelegationCheckRequest(StrictModel):
    signed_delegation: SignedDelegation
    context: DelegationContext


def sign_delegation(
    mandate: DelegationMandate, *, key_id: str, private_key_b64: str
) -> SignedDelegation:
    return SignedDelegation(
        mandate=mandate,
        key_id=key_id,
        signature_b64=sign_manifest(mandate.model_dump(mode="json"), private_key_b64),
    )


def verify_delegation(
    signed: SignedDelegation,
    context: DelegationContext,
    *,
    trusted_public_keys: dict[str, str],
) -> DelegationDecision:
    public_key = trusted_public_keys.get(signed.key_id)
    reasons: list[str] = []
    if public_key is None or not verify_manifest(
        signed.mandate.model_dump(mode="json"), signed.signature_b64, public_key
    ):
        reasons.append("SIGNATURE_INVALID")
    mandate = signed.mandate
    comparisons = (
        (context.agent_id == mandate.agent_id, "AGENT_MISMATCH"),
        (context.merchant_id == mandate.merchant_id, "MERCHANT_MISMATCH"),
        (context.action == mandate.action, "ACTION_NOT_DELEGATED"),
        (context.currency == mandate.currency, "CURRENCY_MISMATCH"),
        (context.amount <= mandate.maximum_amount, "AMOUNT_LIMIT_EXCEEDED"),
        (
            mandate.valid_from <= context.occurred_at <= mandate.valid_until,
            "OUTSIDE_VALIDITY_WINDOW",
        ),
    )
    reasons.extend(reason for valid, reason in comparisons if not valid)
    return DelegationDecision(
        permitted=not reasons,
        reasons=reasons or ["SIGNED_MANDATE_PERMITS"],
        delegation_id=mandate.delegation_id,
        key_id=signed.key_id,
    )


async def verify_delegation_with_opa(
    signed: SignedDelegation,
    context: DelegationContext,
    *,
    trusted_public_keys: dict[str, str],
    opa_url: str,
    timeout_seconds: float = 2.0,
) -> DelegationDecision:
    cryptographic = verify_delegation(signed, context, trusted_public_keys=trusted_public_keys)
    if not cryptographic.permitted:
        return cryptographic
    mandate = signed.mandate
    opa_input = {
        "signature_verified": True,
        "mandate": {
            **mandate.model_dump(mode="json"),
            "valid_from_ns": int(mandate.valid_from.timestamp() * 1_000_000_000),
            "valid_until_ns": int(mandate.valid_until.timestamp() * 1_000_000_000),
        },
        "context": {
            **context.model_dump(mode="json"),
            "occurred_at_ns": int(context.occurred_at.timestamp() * 1_000_000_000),
        },
    }
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(
                f"{opa_url.rstrip('/')}/v1/data/razortrust/delegation/decision",
                json={"input": opa_input},
            )
            response.raise_for_status()
            result = response.json().get("result")
        if not isinstance(result, dict):
            raise ValueError("OPA delegation response is missing")
        return DelegationDecision(
            permitted=bool(result.get("permitted", False)),
            reasons=list(result.get("reasons", ["DELEGATION_DENIED"])),
            delegation_id=mandate.delegation_id,
            key_id=signed.key_id,
            policy_version=str(result.get("policy_version", "delegation-policy@1")),
        )
    except (httpx.HTTPError, ValueError, TypeError):
        return DelegationDecision(
            permitted=False,
            reasons=["POLICY_ENGINE_UNAVAILABLE"],
            delegation_id=mandate.delegation_id,
            key_id=signed.key_id,
            policy_version="delegation-fail-closed@1",
        )
