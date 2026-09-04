from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from razortrust.domain import HoldCase, HoldEvaluationInput, MerchantBaseline, TransactionEvent
from razortrust.features import (
    FEATURE_COLUMNS,
    _histogram_probabilities,
    _kl_divergence,
    build_point_in_time_features,
)

AS_OF = datetime(2026, 8, 28, 12, 0, tzinfo=UTC)


def hold_case() -> HoldCase:
    return HoldCase(
        hold_id=uuid4(),
        request_id=uuid4(),
        merchant_id="merchant_001",
        source_event_id="settlement_001",
        triggered_at=AS_OF,
        reason_code="CAMPAIGN",
    )


def baseline() -> MerchantBaseline:
    return MerchantBaseline(
        volume_mean=4,
        volume_std=2,
        gmv_mean=400,
        gmv_std=100,
        ticket_size_mean=100,
        ticket_size_std=20,
        refund_rate_mean=0.10,
        refund_rate_std=0.05,
        chargeback_rate_mean=0.02,
        chargeback_rate_std=0.01,
        known_devices={"device_1", "device_2"},
        known_geos={"IN"},
        amount_bin_edges=[0, 100, 200],
        amount_bin_probabilities=[0.5, 0.5],
    )


def transaction(
    transaction_id: str,
    hours_before: int,
    amount: float,
    device: str,
    geo: str,
    auth_status: str = "APPROVED",
    refund_timestamp=None,
    chargeback_timestamp=None,
) -> TransactionEvent:
    return TransactionEvent(
        transaction_id=transaction_id,
        merchant_id="merchant_001",
        timestamp=AS_OF - timedelta(hours=hours_before),
        amount=amount,
        device_fingerprint=device,
        customer_geo=geo,
        auth_status=auth_status,
        refund_timestamp=refund_timestamp,
        chargeback_timestamp=chargeback_timestamp,
    )


def evaluation_input() -> HoldEvaluationInput:
    return HoldEvaluationInput(
        baseline=baseline(),
        transactions=[
            transaction(
                "txn_1", 10, 50, "device_1", "IN", refund_timestamp=AS_OF + timedelta(hours=1)
            ),
            transaction("txn_2", 8, 80, "device_1", "IN", auth_status="FAILED"),
            transaction(
                "txn_3", 4, 150, "device_3", "US", refund_timestamp=AS_OF - timedelta(hours=1)
            ),
            transaction(
                "txn_4", 1, 120, "device_2", "IN", chargeback_timestamp=AS_OF + timedelta(days=1)
            ),
        ],
    )


def test_feature_schema_is_exact_and_ordered() -> None:
    features = build_point_in_time_features(hold_case(), evaluation_input())
    assert tuple(features.model_dump()) == FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == 13
    assert features.new_device_ratio == 0.25
    assert features.new_geo_ratio == 0.25
    assert features.failed_auth_ratio == 0.25


def test_future_transactions_and_outcomes_do_not_change_historical_features() -> None:
    hold = hold_case()
    source = evaluation_input()
    before = build_point_in_time_features(hold, source)
    source.transactions.append(
        TransactionEvent(
            transaction_id="txn_future",
            merchant_id=hold.merchant_id,
            timestamp=AS_OF + timedelta(minutes=1),
            amount=5000,
            device_fingerprint="future_device",
            customer_geo="GB",
            auth_status="FAILED",
            refund_timestamp=AS_OF + timedelta(minutes=2),
            chargeback_timestamp=AS_OF + timedelta(minutes=3),
        )
    )
    after = build_point_in_time_features(hold, source)
    assert after == before
    assert before.refund_rate_delta_z == pytest.approx(3.0)
    assert before.chargeback_rate_delta_z == pytest.approx(-2.0)


def test_out_of_range_amounts_are_assigned_to_edge_bins() -> None:
    probabilities = _histogram_probabilities([1000.0], [0.0, 10.0, 20.0])

    assert probabilities[0] < 0.000002
    assert probabilities[1] > 0.999998
    assert _kl_divergence(probabilities, [0.5, 0.5]) > 0.6
