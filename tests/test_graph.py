from __future__ import annotations

from datetime import timedelta

from razortrust.ml.graph import point_in_time_graph_features
from razortrust.ml.graph_evaluation import (
    build_graph_training_frame,
    evaluate_graph_statistics_gate,
)
from razortrust.synthetic import generate_dataset


def test_graph_features_are_point_in_time() -> None:
    merchants, transactions, holds = generate_dataset(
        seed=7, merchants_per_family=1, transactions_per_merchant=12
    )
    merchant = merchants[0]
    cutoff = holds[0].hold.triggered_at
    target_event = next(
        event for event in transactions if event.merchant_id == merchant.merchant_id
    )
    transactions.append(
        target_event.model_copy(
            update={
                "transaction_id": "cross-merchant-shared-device",
                "merchant_id": "merchant_other",
                "customer_id": "shared_customer",
                "ring_id": "evaluation-only-ground-truth",
            }
        )
    )
    before = point_in_time_graph_features(
        merchant.merchant_id,
        transactions,
        cutoff,
        identifier_hmac_key=b"test-graph-secret-key",
    )
    transactions.append(
        transactions[0].model_copy(
            update={"timestamp": cutoff + timedelta(hours=1), "ring_id": "future-ring"}
        )
    )
    after = point_in_time_graph_features(
        merchant.merchant_id,
        transactions,
        cutoff,
        identifier_hmac_key=b"test-graph-secret-key",
    )
    assert before == after
    assert before["graph_device_count"] > 0
    assert before["device_merchant_degree_max"] >= 2
    assert before["two_hop_merchant_count"] >= 1
    assert "graph_ring_event_ratio" not in before


def test_graph_statistics_have_an_explicit_grouped_ablation_gate() -> None:
    merchants, transactions, holds = generate_dataset(
        seed=17, merchants_per_family=4, transactions_per_merchant=12
    )
    frame = build_graph_training_frame(merchants, transactions, holds)
    report = evaluate_graph_statistics_gate(frame, seed=17, folds=3)
    assert report.evaluation_mode == "grouped_development_cold_subgraph"
    assert 0 <= report.tabular_pr_auc <= 1
    assert 0 <= report.graph_stats_pr_auc <= 1
    assert isinstance(report.gate_passed, bool)
