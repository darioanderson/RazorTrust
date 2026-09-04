from __future__ import annotations

from datetime import UTC, datetime

from razortrust.ground_truth import _classify_outcome, _extract_ids


def test_payment_failed_is_not_fraud() -> None:
    payment = {
        "payment_id": "pay_TX7UVlTJR0LtbI",
        "status": "failed",
        "amount_refunded": 0,
    }
    outcome, dispute_confirmed, fraud_confirmed = _classify_outcome(
        payment,
        [],
        [],
    )
    assert outcome == "PAYMENT_FAILED"
    assert dispute_confirmed is False
    assert fraud_confirmed is False


def test_partial_refund_is_refund_observed_not_fraud() -> None:
    payment = {
        "payment_id": "pay_TWshivwwfRNt1L",
        "status": "captured",
        "amount_refunded": 900,
    }
    outcome, dispute_confirmed, fraud_confirmed = _classify_outcome(
        payment,
        [],
        [],
    )
    assert outcome == "REFUND_OBSERVED"
    assert dispute_confirmed is False
    assert fraud_confirmed is False


def test_nonfraud_dispute_is_confirmed_dispute_only() -> None:
    dispute = {
        "dispute_id": "disp_Example123456",
        "phase": "chargeback",
        "status": "open",
    }
    outcome, dispute_confirmed, fraud_confirmed = _classify_outcome(
        None,
        [],
        [dispute],
    )
    assert outcome == "DISPUTE_CONFIRMED"
    assert dispute_confirmed is True
    assert fraud_confirmed is False


def test_fraud_phase_is_required_for_fraud_confirmation() -> None:
    dispute = {
        "dispute_id": "disp_Example123456",
        "phase": "fraud",
        "status": "open",
    }
    outcome, dispute_confirmed, fraud_confirmed = _classify_outcome(
        None,
        [],
        [dispute],
    )
    assert outcome == "FRAUD_DISPUTE_CONFIRMED"
    assert dispute_confirmed is True
    assert fraud_confirmed is True


def test_summary_id_extraction_is_structural() -> None:
    payments, orders = _extract_ids(
        {
            "event": "payment.failed",
            "payment_id": "pay_TX7UVlTJR0LtbI",
            "nested": {"order_id": "order_TX7UIOxcABm0jt"},
        }
    )
    assert payments == {"pay_TX7UVlTJR0LtbI"}
    assert orders == {"order_TX7UIOxcABm0jt"}


def test_unrelated_strings_do_not_become_ids() -> None:
    payments, orders = _extract_ids(
        {
            "message": "pay_ is a prefix, not a payment id",
            "when": datetime(2026, 9, 4, tzinfo=UTC),
        }
    )
    assert payments == set()
    assert orders == set()
