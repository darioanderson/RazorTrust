from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from razortrust.features import FEATURE_COLUMNS
from razortrust.live_features import (
    FirstPartyTelemetrySubmission,
    build_feature_contract_preview,
    hash_device_pseudonym,
    map_auth_status,
)


def _payment(
    payment_id: str,
    when: datetime,
    *,
    order_id: str,
    status: str = "captured",
    amount: int = 100000,
):
    return SimpleNamespace(
        payment_id=payment_id,
        order_id=order_id,
        account_id="acc_test",
        status=status,
        amount=amount,
        currency="INR",
        method="card",
        captured=status in {"captured", "refunded"},
        amount_refunded=0,
        refund_status=None,
        provider_created_at=when,
        observed_at=when + timedelta(minutes=1),
    )


def _telemetry(
    order_id: str, when: datetime, *, device: str, geo: str, payment_id: str | None = None
):
    return SimpleNamespace(
        order_id=order_id,
        payment_id=payment_id,
        device_fingerprint_sha256=hash_device_pseudonym("acc_test", device),
        customer_geo=geo,
        observed_at=when,
    )


def test_auth_mapping_is_frozen_and_fail_closed() -> None:
    assert map_auth_status("authorized") == "APPROVED"
    assert map_auth_status("captured") == "APPROVED"
    assert map_auth_status("refunded") == "APPROVED"
    assert map_auth_status("failed") == "FAILED"
    assert map_auth_status("created") is None


def test_device_pseudonym_is_account_scoped_and_not_stored_raw() -> None:
    raw = "device-session-0000000001"
    hashed = hash_device_pseudonym("acc_test", raw)
    assert len(hashed) == 64
    assert raw not in hashed
    assert hashed != hash_device_pseudonym("acc_other", raw)


def test_geo_contract_rejects_raw_coordinate_style_values() -> None:
    with pytest.raises(ValidationError):
        FirstPartyTelemetrySubmission(
            account_id="acc_test",
            order_id="order_abc123",
            device_pseudonym="device-session-0000000001",
            customer_geo="28.6139,77.2090",
        )
    valid = FirstPartyTelemetrySubmission(
        account_id="acc_test",
        order_id="order_abc123",
        device_pseudonym="device-session-0000000001",
        customer_geo="IN-DL",
    )
    assert valid.customer_geo == "IN-DL"


def test_one_payment_is_correctly_blocked_from_live_feature_vector() -> None:
    as_of = datetime(2026, 9, 2, 12, tzinfo=UTC)
    when = as_of - timedelta(hours=2)
    preview = build_feature_contract_preview(
        account_id="acc_test",
        as_of=as_of,
        payment_rows=[_payment("pay_one", when, order_id="order_one")],
        refund_rows=[],
        dispute_rows=[],
        telemetry_rows=[
            _telemetry(
                "order_one",
                when + timedelta(seconds=10),
                device="device-0000000000000001",
                geo="IN-DL",
            )
        ],
        order_by_payment={"pay_one": "order_one"},
    )
    assert "INSUFFICIENT_BASELINE_TRANSACTIONS" in preview.blockers
    assert preview.feature_vector is None
    assert preview.shadow_score_eligible is False
    assert preview.production_action_eligible is False


def test_complete_history_builds_exact_locked_13_feature_vector() -> None:
    as_of = datetime(2026, 9, 2, 12, tzinfo=UTC)
    payments = []
    telemetry = []
    order_by_payment = {}
    # 35 baseline payments across ten active days, safely outside the current 24h window.
    for i in range(35):
        when = as_of - timedelta(days=2 + (i % 10), hours=i % 5)
        pid = f"pay_base_{i}"
        oid = f"order_base_{i}"
        payments.append(_payment(pid, when, order_id=oid, amount=80_000 + (i % 7) * 5_000))
        telemetry.append(
            _telemetry(
                oid,
                when + timedelta(minutes=2),
                payment_id=pid,
                device=f"baseline-device-{i % 5:02d}-0000000000",
                geo="IN-DL" if i % 2 else "IN-KA",
            )
        )
        order_by_payment[pid] = oid
    # Current window has two payments, one on a new device/geo.
    for i in range(2):
        when = as_of - timedelta(hours=4 - i)
        pid = f"pay_current_{i}"
        oid = f"order_current_{i}"
        payments.append(_payment(pid, when, order_id=oid, amount=120_000 + i * 20_000))
        telemetry.append(
            _telemetry(
                oid,
                when + timedelta(minutes=1),
                payment_id=pid,
                device=(
                    "current-new-device-000000001" if i == 0 else "baseline-device-01-0000000000"
                ),
                geo=("IN-MH" if i == 0 else "IN-DL"),
            )
        )
        order_by_payment[pid] = oid

    preview = build_feature_contract_preview(
        account_id="acc_test",
        as_of=as_of,
        payment_rows=payments,
        refund_rows=[],
        dispute_rows=[],
        telemetry_rows=telemetry,
        order_by_payment=order_by_payment,
    )
    assert preview.blockers == []
    assert preview.feature_vector is not None
    assert tuple(preview.feature_vector) == FEATURE_COLUMNS
    assert preview.feature_vector_sha256 is not None
    assert preview.shadow_score_eligible is True
    assert preview.production_action_eligible is False


def test_telemetry_observed_after_cutoff_cannot_complete_contract() -> None:
    as_of = datetime(2026, 9, 2, 12, tzinfo=UTC)
    baseline_payments = []
    baseline_telemetry = []
    order_by_payment = {}
    for i in range(35):
        when = as_of - timedelta(days=2 + (i % 10), hours=i % 3)
        pid = f"pay_base_{i}"
        oid = f"order_base_{i}"
        baseline_payments.append(_payment(pid, when, order_id=oid))
        baseline_telemetry.append(
            _telemetry(
                oid,
                when + timedelta(minutes=1),
                payment_id=pid,
                device=f"device-{i:020d}",
                geo="IN-DL",
            )
        )
        order_by_payment[pid] = oid
    current = _payment("pay_current", as_of - timedelta(hours=1), order_id="order_current")
    order_by_payment[current.payment_id] = "order_current"
    # Simulate the store's as-of filter: future telemetry is intentionally not passed
    # to the builder.
    preview = build_feature_contract_preview(
        account_id="acc_test",
        as_of=as_of,
        payment_rows=[*baseline_payments, current],
        refund_rows=[],
        dispute_rows=[],
        telemetry_rows=baseline_telemetry,
        order_by_payment=order_by_payment,
    )
    assert "CURRENT_WINDOW_TELEMETRY_INCOMPLETE" in preview.blockers
    assert preview.feature_vector is None
