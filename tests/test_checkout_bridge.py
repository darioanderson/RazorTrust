from __future__ import annotations

import hashlib
import hmac
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from razortrust.checkout_bridge import (
    CheckoutOrderCreate,
    validate_authoritative_payment,
    verify_checkout_signature,
)
from razortrust.live_features import build_feature_contract_preview


def test_checkout_signature_uses_stored_order_and_constant_contract() -> None:
    secret = "test-secret"
    order_id = "order_ABC123"
    payment_id = "pay_XYZ789"
    signature = hmac.new(
        secret.encode(), f"{order_id}|{payment_id}".encode(), hashlib.sha256
    ).hexdigest()
    assert verify_checkout_signature(
        order_id=order_id,
        payment_id=payment_id,
        signature=signature,
        secret=secret,
    )
    assert not verify_checkout_signature(
        order_id="order_TAMPERED",
        payment_id=payment_id,
        signature=signature,
        secret=secret,
    )


def test_checkout_order_rejects_raw_coordinates() -> None:
    with pytest.raises(ValidationError):
        CheckoutOrderCreate(
            amount=1000,
            device_pseudonym="device-pseudonym-00000001",
            customer_geo="28.6139,77.2090",
        )


def test_authoritative_payment_must_match_server_order_amount_and_currency() -> None:
    payment = {
        "id": "pay_XYZ789",
        "order_id": "order_ABC123",
        "amount": 1000,
        "currency": "INR",
        "status": "captured",
        "captured": True,
    }
    assert (
        validate_authoritative_payment(
            stored_order_id="order_ABC123",
            stored_amount=1000,
            stored_currency="INR",
            payment_id="pay_XYZ789",
            payment=payment,
        )
        == "captured"
    )
    with pytest.raises(ValueError):
        validate_authoritative_payment(
            stored_order_id="order_OTHER",
            stored_amount=1000,
            stored_currency="INR",
            payment_id="pay_XYZ789",
            payment=payment,
        )


def test_zero_baseline_telemetry_coverage_is_not_reported_as_100_percent() -> None:
    preview = build_feature_contract_preview(
        account_id="acc_test",
        as_of=datetime(2026, 9, 2, 12, tzinfo=UTC),
        payment_rows=[],
        refund_rows=[],
        dispute_rows=[],
        telemetry_rows=[],
        order_by_payment={},
    )
    assert preview.telemetry_coverage_baseline is None
    assert preview.telemetry_coverage_current is None
    assert "INSUFFICIENT_BASELINE_TRANSACTIONS" in preview.blockers
    assert "NO_CURRENT_WINDOW_PAYMENTS" in preview.blockers
