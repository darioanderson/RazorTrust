from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from experimental.agentproof.delegation import (
    DelegationContext,
    DelegationMandate,
    sign_delegation,
    verify_delegation,
)
from razortrust.security import generate_release_keypair


def test_signed_delegation_enforces_signature_scope_limit_and_time() -> None:
    private_key, public_key = generate_release_keypair()
    now = datetime(2026, 8, 29, tzinfo=UTC)
    mandate = DelegationMandate(
        delegation_id=uuid4(),
        agent_id="agent-1",
        merchant_id="merchant-1",
        action="SETTLE",
        currency="INR",
        maximum_amount=100_000,
        valid_from=now - timedelta(hours=1),
        valid_until=now + timedelta(hours=1),
        nonce="nonce-1234567890",
    )
    signed = sign_delegation(mandate, key_id="delegation-2026-08", private_key_b64=private_key)
    context = DelegationContext(
        agent_id="agent-1",
        merchant_id="merchant-1",
        action="SETTLE",
        currency="INR",
        amount=50_000,
        occurred_at=now,
    )

    allowed = verify_delegation(
        signed, context, trusted_public_keys={"delegation-2026-08": public_key}
    )
    denied = verify_delegation(
        signed,
        context.model_copy(update={"amount": 100_001}),
        trusted_public_keys={"delegation-2026-08": public_key},
    )

    assert allowed.permitted is True
    assert denied.permitted is False
    assert "AMOUNT_LIMIT_EXCEEDED" in denied.reasons


def test_tampered_delegation_is_denied() -> None:
    private_key, public_key = generate_release_keypair()
    now = datetime(2026, 8, 29, tzinfo=UTC)
    mandate = DelegationMandate(
        delegation_id=uuid4(),
        agent_id="agent-1",
        merchant_id="merchant-1",
        action="SETTLE",
        currency="INR",
        maximum_amount=100_000,
        valid_from=now - timedelta(hours=1),
        valid_until=now + timedelta(hours=1),
        nonce="nonce-1234567890",
    )
    signed = sign_delegation(mandate, key_id="key-1", private_key_b64=private_key)
    signed = signed.model_copy(
        update={"mandate": signed.mandate.model_copy(update={"maximum_amount": 1_000_000})}
    )
    context = DelegationContext(
        agent_id="agent-1",
        merchant_id="merchant-1",
        action="SETTLE",
        currency="INR",
        amount=50_000,
        occurred_at=now,
    )

    decision = verify_delegation(signed, context, trusted_public_keys={"key-1": public_key})

    assert decision.permitted is False
    assert "SIGNATURE_INVALID" in decision.reasons
