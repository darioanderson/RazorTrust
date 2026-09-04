from __future__ import annotations

from datetime import timedelta

import numpy as np

from razortrust.ml.sequence_models import (
    FALSE_HOLD_COST,
    FALSE_RELEASE_COST,
    SEQUENCE_FEATURES,
    SequenceModelReport,
    build_sequence_example,
    sequence_gate_passed,
)
from razortrust.synthetic import TrueRiskState, generate_dataset


def test_neural_sequence_example_is_point_in_time() -> None:
    merchants, transactions, holds = generate_dataset(
        seed=121,
        merchants_per_family=1,
        transactions_per_merchant=12,
    )
    del merchants
    hold = holds[0]
    before = build_sequence_example(
        merchant_id=hold.hold.merchant_id,
        transactions=transactions,
        cutoff=hold.hold.triggered_at,
        label=int(hold.true_risk_state == TrueRiskState.RISKY),
        attack_family=hold.attack_family,
        max_events=32,
    )
    future = transactions[0].model_copy(
        update={"timestamp": hold.hold.triggered_at + timedelta(seconds=1)}
    )
    after = build_sequence_example(
        merchant_id=hold.hold.merchant_id,
        transactions=[*transactions, future],
        cutoff=hold.hold.triggered_at,
        label=int(hold.true_risk_state == TrueRiskState.RISKY),
        attack_family=hold.attack_family,
        max_events=32,
    )
    assert np.array_equal(before.values, after.values)
    assert before.values.shape == (32, len(SEQUENCE_FEATURES))
    assert 1 <= before.length <= 32


def test_sequence_gate_requires_metric_cost_and_recall_improvement() -> None:
    passing = SequenceModelReport(
        model_type="lstm",
        evaluation_mode="test",
        folds=3,
        pr_auc=0.82,
        expected_cost=8.0,
        risk_recall=0.90,
        hold_rate=0.40,
        case_count=100,
    )
    assert sequence_gate_passed(
        temporal_pr_auc=0.80,
        temporal_expected_cost=9.0,
        temporal_risk_recall=0.90,
        challenger=passing,
    )
    failing = SequenceModelReport(
        model_type="lstm",
        evaluation_mode="test",
        folds=3,
        pr_auc=0.82,
        expected_cost=8.0,
        risk_recall=0.89,
        hold_rate=0.40,
        case_count=100,
    )
    assert not sequence_gate_passed(
        temporal_pr_auc=0.80,
        temporal_expected_cost=9.0,
        temporal_risk_recall=0.90,
        challenger=failing,
    )
    assert FALSE_RELEASE_COST == 100
    assert FALSE_HOLD_COST == 25
