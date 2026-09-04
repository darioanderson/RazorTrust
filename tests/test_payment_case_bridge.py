from __future__ import annotations

from datetime import UTC, datetime, timedelta

from razortrust.payment_case_bridge import (
    payment_case_request_id,
    payment_decision_time,
    payment_reason_code,
)


def test_failed_payment_maps_to_real_failure_reason() -> None:
    assert (
        payment_reason_code(status="failed", amount_refunded=0)
        == "RAZORPAY_PAYMENT_FAILED"
    )


def test_refund_takes_precedence_over_captured_status() -> None:
    assert (
        payment_reason_code(status="captured", amount_refunded=900)
        == "RAZORPAY_REFUND_OBSERVED"
    )


def test_case_request_id_is_deterministic_and_payment_scoped() -> None:
    first = payment_case_request_id("acc_one", "pay_one")
    second = payment_case_request_id("acc_one", "pay_one")
    other = payment_case_request_id("acc_one", "pay_two")

    assert first == second
    assert first != other


def test_source_event_time_is_used_as_decision_time() -> None:
    created = datetime(2026, 9, 2, 9, tzinfo=UTC)
    observed = created + timedelta(seconds=3)
    updated = observed + timedelta(seconds=2)

    chosen = payment_decision_time(
        {
            "provider_created_at": created,
            "source_event_created_at": observed,
            "updated_at": updated,
        }
    )
    assert chosen == observed
