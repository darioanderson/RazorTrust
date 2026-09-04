from __future__ import annotations

from datetime import UTC, datetime

from razortrust.processor_timeline import (
    PaymentObservation,
    RefundObservation,
    build_point_in_time_snapshot,
)


def payment_observation(
    *,
    observed_at: datetime,
    amount_refunded: int = 0,
    refund_status: str | None = None,
    currency: str = "INR",
) -> PaymentObservation:
    return PaymentObservation(
        payment_id="pay_001",
        status="captured",
        amount=1000,
        currency=currency,
        method="wallet",
        captured=True,
        amount_refunded=amount_refunded,
        refund_status=refund_status,
        international=False,
        provider_created_at=datetime(2026, 9, 1, 19, 4, tzinfo=UTC),
        source_event_id=f"evt_{int(observed_at.timestamp())}",
        observed_at=observed_at,
    )


def test_future_refund_state_does_not_leak_into_earlier_cutoff() -> None:
    first = payment_observation(
        observed_at=datetime(2026, 9, 1, 19, 4, 12, tzinfo=UTC),
        amount_refunded=0,
    )
    after_refund = payment_observation(
        observed_at=datetime(2026, 9, 1, 21, 27, 13, tzinfo=UTC),
        amount_refunded=500,
        refund_status="partial",
    )

    snapshot = build_point_in_time_snapshot(
        account_id="acc_001",
        as_of=datetime(2026, 9, 1, 20, 0, tzinfo=UTC),
        lookback_hours=24,
        payment_observations=[first, after_refund],
        refund_observations=[],
        dispute_observations=[],
        settlement_observations=[],
        unresolved_pipeline_events=0,
    )

    assert len(snapshot.payments) == 1
    assert snapshot.payments[0].amount_refunded == 0
    assert snapshot.metrics.amount_refunded == 0


def test_later_cutoff_sees_authoritative_refund_state() -> None:
    first = payment_observation(
        observed_at=datetime(2026, 9, 1, 19, 4, 12, tzinfo=UTC),
        amount_refunded=0,
    )
    after_refund = payment_observation(
        observed_at=datetime(2026, 9, 1, 21, 27, 13, tzinfo=UTC),
        amount_refunded=500,
        refund_status="partial",
    )

    snapshot = build_point_in_time_snapshot(
        account_id="acc_001",
        as_of=datetime(2026, 9, 1, 21, 40, tzinfo=UTC),
        lookback_hours=24,
        payment_observations=[first, after_refund],
        refund_observations=[],
        dispute_observations=[],
        settlement_observations=[],
        unresolved_pipeline_events=0,
    )

    assert snapshot.payments[0].amount_refunded == 500
    assert snapshot.metrics.amount_refunded == 500


def test_data_quality_firewall_blocks_impossible_refund_and_unresolved_pipeline() -> None:
    payment = payment_observation(
        observed_at=datetime(2026, 9, 1, 19, 5, tzinfo=UTC),
        amount_refunded=1200,
        refund_status="full",
    )

    snapshot = build_point_in_time_snapshot(
        account_id="acc_001",
        as_of=datetime(2026, 9, 1, 20, 0, tzinfo=UTC),
        lookback_hours=24,
        payment_observations=[payment],
        refund_observations=[],
        dispute_observations=[],
        settlement_observations=[],
        unresolved_pipeline_events=1,
    )

    assert snapshot.data_quality.status == "BLOCKED"
    codes = {issue.code for issue in snapshot.data_quality.issues}
    assert "REFUND_EXCEEDS_PAYMENT" in codes
    assert "UNRESOLVED_PIPELINE_EVENTS" in codes
    assert snapshot.feature_readiness.model_input_ready is False
    assert "DATA_QUALITY_FIREWALL_BLOCKED" in snapshot.feature_readiness.blockers


def test_mixed_currency_window_is_blocked_before_aggregation() -> None:
    inr = payment_observation(observed_at=datetime(2026, 9, 1, 19, 5, tzinfo=UTC))
    usd = payment_observation(
        observed_at=datetime(2026, 9, 1, 19, 6, tzinfo=UTC), currency="USD"
    ).model_copy(
        update={
            "payment_id": "pay_002",
            "provider_created_at": datetime(2026, 9, 1, 19, 5, tzinfo=UTC),
            "source_event_id": "evt_usd",
        }
    )

    snapshot = build_point_in_time_snapshot(
        account_id="acc_001",
        as_of=datetime(2026, 9, 1, 20, 0, tzinfo=UTC),
        lookback_hours=24,
        payment_observations=[inr, usd],
        refund_observations=[],
        dispute_observations=[],
        settlement_observations=[],
        unresolved_pipeline_events=0,
    )

    assert snapshot.data_quality.status == "BLOCKED"
    assert any(issue.code == "MIXED_CURRENCIES" for issue in snapshot.data_quality.issues)


def test_locked_model_is_blocked_until_missing_feature_sources_are_connected() -> None:
    payment = payment_observation(observed_at=datetime(2026, 9, 1, 19, 5, tzinfo=UTC))
    refund = RefundObservation(
        refund_id="rfnd_001",
        payment_id="pay_001",
        amount=400,
        currency="INR",
        status="processed",
        provider_created_at=datetime(2026, 9, 1, 21, 31, tzinfo=UTC),
        source_event_id="evt_refund",
        observed_at=datetime(2026, 9, 1, 21, 31, 20, tzinfo=UTC),
    )

    snapshot = build_point_in_time_snapshot(
        account_id="acc_001",
        as_of=datetime(2026, 9, 1, 21, 40, tzinfo=UTC),
        lookback_hours=24,
        payment_observations=[payment],
        refund_observations=[refund],
        dispute_observations=[],
        settlement_observations=[],
        unresolved_pipeline_events=0,
    )

    assert snapshot.data_quality.status == "PASS"
    assert snapshot.feature_readiness.model_input_ready is False
    missing = set(snapshot.feature_readiness.missing_source_dimensions)
    assert "device_fingerprint" in missing
    assert "customer_geo" in missing
    assert "auth_status with APPROVED/FAILED semantics" in missing
    assert snapshot.metrics.refund_count == 1


def test_knowledge_time_not_provider_time_controls_visibility() -> None:
    provider_time = datetime(2026, 9, 1, 19, 4, tzinfo=UTC)
    late_observation = payment_observation(
        observed_at=datetime(2026, 9, 1, 20, 30, tzinfo=UTC)
    ).model_copy(update={"provider_created_at": provider_time})

    snapshot = build_point_in_time_snapshot(
        account_id="acc_001",
        as_of=datetime(2026, 9, 1, 20, 0, tzinfo=UTC),
        lookback_hours=24,
        payment_observations=[late_observation],
        refund_observations=[],
        dispute_observations=[],
        settlement_observations=[],
        unresolved_pipeline_events=0,
    )

    assert snapshot.payments == []
    assert snapshot.metrics.payment_count == 0
    assert snapshot.knowledge_time_policy == "OBSERVED_AT"
