from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

from razortrust.ml.fdh_graph import (
    FDH_GRAPH_FEATURES,
    FDH_TABULAR_FEATURES,
    build_history_index,
    build_pilot_frame,
    load_fdh_transactions,
    select_chronological_case_control_pilot,
)
from razortrust.ml.fdh_hetero_graph import (
    apply_hetero_normalizer,
    build_point_in_time_hetero_snapshot,
    fit_hetero_normalizer,
)
from razortrust.ml.hetero_graphsage import (
    fit_hetero_graphsage,
    hetero_graphsage_embedding,
    hetero_graphsage_probability,
)


def _fit_xgb(frame: pd.DataFrame, columns: list[str], seed: int):
    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
    )
    model.fit(frame[columns], frame["label"].to_numpy(dtype=int))
    return model


def _select_threshold(probabilities: np.ndarray, labels: np.ndarray) -> float:
    best: tuple[float, float] | None = None
    best_threshold = 0.5
    for threshold in np.linspace(0.01, 0.99, 99):
        predicted = probabilities >= threshold
        f1 = float(f1_score(labels, predicted, zero_division=0))
        candidate = (-f1, -float(threshold))
        if best is None or candidate < best:
            best = candidate
            best_threshold = float(threshold)
    return best_threshold


def _metrics(
    labels: np.ndarray,
    scenarios: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    predicted = probabilities >= threshold
    negatives = labels == 0
    result: dict[str, object] = {
        "pr_auc": round(float(average_precision_score(labels, probabilities)), 8),
        "precision": round(float(precision_score(labels, predicted, zero_division=0)), 8),
        "recall": round(float(recall_score(labels, predicted, zero_division=0)), 8),
        "f1": round(float(f1_score(labels, predicted, zero_division=0)), 8),
        "sample_fpr": round(float(np.mean(predicted[negatives])) if negatives.any() else 0.0, 8),
        "predicted_positive_rate": round(float(np.mean(predicted)), 8),
        "threshold": round(float(threshold), 6),
    }
    for scenario in (1, 2, 3):
        mask = scenarios == scenario
        result[f"scenario_{scenario}_recall"] = round(
            float(recall_score(labels[mask], predicted[mask], zero_division=0))
            if np.any(mask)
            else 0.0,
            8,
        )
        result[f"scenario_{scenario}_cases"] = int(np.sum(mask))
    return result


def _append_embeddings(
    frame: pd.DataFrame,
    embeddings: list[np.ndarray],
) -> tuple[pd.DataFrame, list[str]]:
    if len(frame) != len(embeddings):
        raise ValueError("embedding count must match frame rows")
    matrix = np.vstack(embeddings)
    columns = [f"hetero_gnn_embedding_{index:02d}" for index in range(matrix.shape[1])]
    result = frame.copy()
    for index, column in enumerate(columns):
        result[column] = matrix[:, index]
    return result, columns


def _snapshot_batch(history, indices, args):
    return [
        build_point_in_time_hetero_snapshot(
            history,
            int(index),
            lookback_days=args.graph_lookback_days,
            label_delay_days=args.label_delay_days,
            max_history_events=args.max_history_events,
        )
        for index in indices
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3B V2 heterogeneous GraphSAGE fusion research pilot"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--train-cases", type=int, default=480)
    parser.add_argument("--policy-cases", type=int, default=160)
    parser.add_argument("--test-cases", type=int, default=160)
    parser.add_argument("--representation-fraction", type=float, default=0.60)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--label-delay-days", type=int, default=7)
    parser.add_argument("--graph-lookback-days", type=int, default=30)
    parser.add_argument("--max-history-events", type=int, default=500)
    args = parser.parse_args()

    if not 0.40 <= args.representation_fraction <= 0.75:
        raise SystemExit("--representation-fraction must be between 0.40 and 0.75")
    if args.hidden_dim < 8 or args.hidden_dim > 128:
        raise SystemExit("--hidden-dim must be between 8 and 128")

    raw = load_fdh_transactions(args.data_dir)
    history = build_history_index(raw)
    split = select_chronological_case_control_pilot(
        raw,
        seed=args.seed,
        max_train=args.train_cases,
        max_policy=args.policy_cases,
        max_test=args.test_cases,
    )

    train_indices = np.asarray(split.train, dtype=np.int64)
    representation_count = int(len(train_indices) * args.representation_fraction)
    representation_indices = train_indices[:representation_count]
    classifier_indices = train_indices[representation_count:]
    if len(representation_indices) < 40 or len(classifier_indices) < 40:
        raise SystemExit("Phase 3B V2 requires at least 40 representation and classifier cases")
    if not (
        raw.iloc[representation_indices]["TX_DATETIME"].max()
        < raw.iloc[classifier_indices]["TX_DATETIME"].min()
    ):
        raise RuntimeError("representation/classifier chronology overlap")
    for name, indices in (
        ("representation_train", representation_indices),
        ("classifier_train", classifier_indices),
        ("policy", np.asarray(split.policy, dtype=np.int64)),
        ("test", np.asarray(split.test, dtype=np.int64)),
    ):
        labels_present = set(raw.iloc[indices]["TX_FRAUD"].astype(int).unique().tolist())
        if labels_present != {0, 1}:
            raise SystemExit(f"{name} must contain both legitimate and fraud cases")

    representation_frame = build_pilot_frame(
        history, representation_indices, label_delay_days=args.label_delay_days
    )
    classifier_train = build_pilot_frame(
        history, classifier_indices, label_delay_days=args.label_delay_days
    )
    policy = build_pilot_frame(history, split.policy, label_delay_days=args.label_delay_days)
    test = build_pilot_frame(history, split.test, label_delay_days=args.label_delay_days)

    representation_snapshots_raw = _snapshot_batch(history, representation_indices, args)
    normalizer = fit_hetero_normalizer(representation_snapshots_raw)
    representation_snapshots = [
        apply_hetero_normalizer(snapshot, normalizer)
        for snapshot in representation_snapshots_raw
    ]
    gnn = fit_hetero_graphsage(
        representation_snapshots,
        representation_frame["label"].astype(int).tolist(),
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )

    classifier_snapshots = [
        apply_hetero_normalizer(snapshot, normalizer)
        for snapshot in _snapshot_batch(history, classifier_indices, args)
    ]
    policy_snapshots = [
        apply_hetero_normalizer(snapshot, normalizer)
        for snapshot in _snapshot_batch(history, split.policy, args)
    ]
    test_snapshots = [
        apply_hetero_normalizer(snapshot, normalizer)
        for snapshot in _snapshot_batch(history, split.test, args)
    ]

    classifier_embeddings = [hetero_graphsage_embedding(gnn, s) for s in classifier_snapshots]
    policy_embeddings = [hetero_graphsage_embedding(gnn, s) for s in policy_snapshots]
    test_embeddings = [hetero_graphsage_embedding(gnn, s) for s in test_snapshots]
    gnn_policy_probabilities = np.asarray(
        [hetero_graphsage_probability(gnn, s) for s in policy_snapshots], dtype=float
    )
    gnn_test_probabilities = np.asarray(
        [hetero_graphsage_probability(gnn, s) for s in test_snapshots], dtype=float
    )

    classifier_fusion, embedding_columns = _append_embeddings(
        classifier_train, classifier_embeddings
    )
    policy_fusion, _ = _append_embeddings(policy, policy_embeddings)
    test_fusion, _ = _append_embeddings(test, test_embeddings)

    tabular_columns = list(FDH_TABULAR_FEATURES)
    graph_columns = [*FDH_TABULAR_FEATURES, *FDH_GRAPH_FEATURES]
    fusion_columns = [*graph_columns, *embedding_columns]

    tabular_model = _fit_xgb(classifier_train, tabular_columns, args.seed)
    graph_stats_model = _fit_xgb(classifier_train, graph_columns, args.seed)
    fusion_model = _fit_xgb(classifier_fusion, fusion_columns, args.seed)

    tab_policy = tabular_model.predict_proba(policy[tabular_columns])[:, 1]
    stats_policy = graph_stats_model.predict_proba(policy[graph_columns])[:, 1]
    fusion_policy = fusion_model.predict_proba(policy_fusion[fusion_columns])[:, 1]
    policy_labels = policy["label"].to_numpy(dtype=int)

    tab_threshold = _select_threshold(tab_policy, policy_labels)
    stats_threshold = _select_threshold(stats_policy, policy_labels)
    fusion_threshold = _select_threshold(fusion_policy, policy_labels)
    gnn_threshold = _select_threshold(gnn_policy_probabilities, policy_labels)

    labels = test["label"].to_numpy(dtype=int)
    scenarios = test["scenario"].to_numpy(dtype=int)
    tabular = _metrics(
        labels,
        scenarios,
        tabular_model.predict_proba(test[tabular_columns])[:, 1],
        tab_threshold,
    )
    graph_stats = _metrics(
        labels,
        scenarios,
        graph_stats_model.predict_proba(test[graph_columns])[:, 1],
        stats_threshold,
    )
    fusion = _metrics(
        labels,
        scenarios,
        fusion_model.predict_proba(test_fusion[fusion_columns])[:, 1],
        fusion_threshold,
    )
    gnn_standalone = _metrics(
        labels, scenarios, gnn_test_probabilities, gnn_threshold
    )

    graph_stats_value_gate = bool(
        float(graph_stats["pr_auc"]) >= float(tabular["pr_auc"]) + 0.01
        and float(graph_stats["f1"]) >= float(tabular["f1"])
        and float(graph_stats["recall"]) >= float(tabular["recall"]) - 0.05
        and float(graph_stats["sample_fpr"]) <= float(tabular["sample_fpr"]) + 0.02
    )
    hetero_incremental_gate = bool(
        float(fusion["pr_auc"]) >= float(graph_stats["pr_auc"]) + 0.01
        and float(fusion["f1"]) >= float(graph_stats["f1"])
        and float(fusion["recall"]) >= float(graph_stats["recall"]) - 0.03
        and float(fusion["sample_fpr"]) <= float(graph_stats["sample_fpr"]) + 0.02
        and max(
            float(fusion["scenario_2_recall"]) - float(graph_stats["scenario_2_recall"]),
            float(fusion["scenario_3_recall"]) - float(graph_stats["scenario_3_recall"]),
        )
        > 0.0
    )

    if not graph_stats_value_gate:
        decision = "PHASE3B_V2_GRAPH_STATS_BASELINE_NOT_CONFIRMED"
    elif hetero_incremental_gate:
        decision = "PHASE3B_V2_HETERO_GRAPHSAGE_RETAINS_RESEARCH_VALUE"
    else:
        decision = "PHASE3B_V2_REJECT_HETERO_GRAPHSAGE_COMPLEXITY"

    def time_range(indices: np.ndarray) -> list[str]:
        part = raw.iloc[indices]
        return [str(part["TX_DATETIME"].min()), str(part["TX_DATETIME"].max())]

    report = {
        "schema_version": "2.0",
        "phase": "3B-V2",
        "benchmark_label": "PUBLIC SYNTHETIC STRUCTURED HETEROGENEOUS GRAPH PILOT",
        "domain": (
            "Fraud Detection Handbook simulated transaction fraud; "
            "not Razorpay settlement-hold ground truth"
        ),
        "decision": decision,
        "dataset_rows": int(len(raw)),
        "dataset_fraud": int(raw["TX_FRAUD"].sum()),
        "architecture": {
            "node_types": ["customer", "transaction", "terminal"],
            "relations": [
                "customer-makes-transaction",
                "transaction-made_by-customer",
                "transaction-at-terminal",
                "terminal-hosts-transaction",
            ],
            "historical_transactions_preserved_as_nodes": True,
            "fusion": (
                "tabular + graph statistics + frozen heterogeneous "
                "GraphSAGE embedding -> XGBoost"
            ),
            "embedding_dim": args.hidden_dim,
            "representation_learning_separate_from_classifier_fit": True,
        },
        "pilot_sampling": (
            "chronological case-control pilot; metrics are comparative "
            "research metrics, not population operating rates"
        ),
        "partition_rows": {
            "representation_train": int(len(representation_indices)),
            "classifier_train": int(len(classifier_indices)),
            "policy": int(len(split.policy)),
            "test": int(len(split.test)),
        },
        "partition_time_ranges": {
            "representation_train": time_range(representation_indices),
            "classifier_train": time_range(classifier_indices),
            "policy": time_range(split.policy),
            "test": time_range(split.test),
        },
        "label_delay_days": args.label_delay_days,
        "graph_lookback_days": args.graph_lookback_days,
        "max_history_events_per_snapshot": args.max_history_events,
        "normalization_fit_partition": "representation_train only",
        "tabular_baseline": tabular,
        "graph_statistics": graph_stats,
        "hetero_graphsage_standalone_diagnostic": gnn_standalone,
        "hetero_graphsage_fusion": fusion,
        "graph_statistics_value_gate": graph_stats_value_gate,
        "hetero_graphsage_incremental_gate": hetero_incremental_gate,
        "pre_registered_gate": {
            "graph_stats_vs_tabular": {
                "pr_auc_delta_min": 0.01,
                "f1_not_worse": True,
                "recall_tolerance": -0.05,
                "sample_fpr_tolerance": 0.02,
            },
            "hetero_fusion_vs_graph_stats": {
                "pr_auc_delta_min": 0.01,
                "f1_not_worse": True,
                "recall_tolerance": -0.03,
                "sample_fpr_tolerance": 0.02,
                "scenario_2_or_3_recall_must_improve": True,
            },
        },
        "safety": {
            "history_topology_rule": (
                "historical transaction timestamp < current transaction timestamp"
            ),
            "current_transaction_label_used_as_feature": False,
            "future_labels_used": False,
            "fraud_derived_history_available_only_after_delay_days": args.label_delay_days,
            "train_only_normalization": True,
            "original_phase3_decision_overwritten": False,
            "phase3b_v1_decision_overwritten": False,
            "production_13_feature_contract_changed": False,
            "sealed_test_used": False,
            "stress_set_used": False,
            "production_action_eligible": False,
            "serving_change_authorized": False,
            "automatic_promotion": False,
            "automatic_release_enabled": False,
            "champion_remains": "xgb-if-settlement@2",
            "active_enforcement_runtime": "human-only@1",
        },
    }

    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "phase3b_v2_hetero_graph_report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report: {destination}")


if __name__ == "__main__":
    main()
