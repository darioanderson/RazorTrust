from __future__ import annotations

from datetime import timedelta

import pytest

from razortrust.domain import HoldCase, HoldDecision, HoldEvaluationInput
from razortrust.features import FEATURE_COLUMNS, build_point_in_time_features
from razortrust.synthetic import (
    LEGITIMATE_FAMILIES,
    RISK_FAMILIES,
    TrueRiskState,
    generate_dataset,
    write_dataset,
)


def test_generator_is_deterministic_and_covers_every_family() -> None:
    first = generate_dataset(seed=7, merchants_per_family=2, transactions_per_merchant=12)
    second = generate_dataset(seed=7, merchants_per_family=2, transactions_per_merchant=12)
    merchants, transactions, holds = first
    assert first == second
    assert len(merchants) == 2 * len(LEGITIMATE_FAMILIES + RISK_FAMILIES)
    assert len(holds) == len(merchants)
    assert {merchant.scenario_family for merchant in merchants} == set(
        LEGITIMATE_FAMILIES + RISK_FAMILIES
    )
    assert any(event.ring_id is not None for event in transactions)
    assert {hold.true_risk_state for hold in holds} == {
        TrueRiskState.LEGITIMATE,
        TrueRiskState.RISKY,
    }
    assert all(
        hold.attack_family is None
        for hold in holds
        if hold.true_risk_state == TrueRiskState.LEGITIMATE
    )
    assert all(
        hold.operational_target == HoldDecision.EVIDENCE_NEEDED
        for hold in holds
        if hold.evidence_resolvable
    )


def test_dataset_manifest_is_reproducible_and_files_are_immutable(tmp_path) -> None:
    first = write_dataset(
        tmp_path / "first", seed=11, merchants_per_family=1, transactions_per_merchant=12
    )
    retry = write_dataset(
        tmp_path / "first", seed=11, merchants_per_family=1, transactions_per_merchant=12
    )
    second = write_dataset(
        tmp_path / "second", seed=11, merchants_per_family=1, transactions_per_merchant=12
    )
    assert first == retry == second
    assert first.dataset_version == "synthetic@3"
    assert set(first.file_sha256) == {
        "merchants.jsonl",
        "transactions.jsonl",
        "holds.jsonl",
    }

    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_dataset(
            tmp_path / "first", seed=12, merchants_per_family=1, transactions_per_merchant=12
        )


def test_synthetic_future_events_cannot_change_historical_features() -> None:
    merchants, transactions, holds = generate_dataset(
        seed=19, merchants_per_family=1, transactions_per_merchant=12
    )
    merchant = merchants[0]
    synthetic_hold = holds[0]
    hold = synthetic_hold.hold
    case = hold.model_dump()
    case["hold_id"] = hold.request_id
    hold_case = HoldCase.model_validate(case)
    merchant_transactions = [
        event for event in transactions if event.merchant_id == merchant.merchant_id
    ]
    before = build_point_in_time_features(
        hold_case,
        HoldEvaluationInput(baseline=merchant.baseline, transactions=merchant_transactions),
    )
    future = merchant_transactions[-1].model_copy(
        update={
            "transaction_id": "future_mutation",
            "timestamp": hold.triggered_at + timedelta(seconds=1),
            "amount": 1_000_000.0,
            "device_fingerprint": "future-device",
            "customer_geo": "future-geo",
            "auth_status": "FAILED",
            "refund_timestamp": hold.triggered_at + timedelta(seconds=2),
            "chargeback_timestamp": hold.triggered_at + timedelta(seconds=3),
        }
    )
    after = build_point_in_time_features(
        hold_case,
        HoldEvaluationInput(
            baseline=merchant.baseline,
            transactions=[*merchant_transactions, future],
        ),
    )
    assert before == after
    assert tuple(before.model_dump()) == FEATURE_COLUMNS
