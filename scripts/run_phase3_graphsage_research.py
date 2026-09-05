from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

from razortrust.features import FEATURE_COLUMNS
from razortrust.ml.graph_evaluation import (
    RING_ATTACK_FAMILIES,
    build_graph_training_frame,
    evaluate_graph_statistics_gate,
)
from razortrust.ml.graphsage import (
    build_point_in_time_graph_snapshot,
    fit_graphsage_snapshots,
    predict_graphsage_probability,
)
from razortrust.synthetic import TrueRiskState, generate_dataset


def _binary_cost(labels: np.ndarray, hold_actions: np.ndarray) -> float:
    false_release = (~hold_actions) & (labels == 1)
    false_hold = hold_actions & (labels == 0)
    return float(np.mean(false_release * 100 + false_hold * 25))


def _safe_recall(labels: np.ndarray, actions: np.ndarray) -> float:
    if len(labels) == 0 or not np.any(labels == 1):
        return 0.0
    return float(recall_score(labels, actions, zero_division=0))


def _graph_columns(frame) -> list[str]:
    explicit = {
        "shared_device_count_1h",
        "shared_device_count_24h",
        "shared_customer_count",
        "new_neighbor_ratio",
        "device_merchant_degree_max",
        "device_reuse_velocity",
        "component_growth_24h",
        "two_hop_merchant_count",
        "two_hop_known_risk_density",
        "geo_overlap",
        "edge_creation_rate",
    }
    return [column for column in frame if column.startswith("graph_") or column in explicit]


def _grouped_xgb_probabilities(
    frame, labels: np.ndarray, groups, columns: list[str], *, seed: int, folds: int
) -> np.ndarray:
    probabilities = np.zeros(len(frame), dtype=float)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_index, validation_index in splitter.split(frame, labels, groups):
        model = XGBClassifier(
            objective="binary:logistic",
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
        )
        model.fit(frame.iloc[train_index][columns], labels[train_index])
        probabilities[validation_index] = model.predict_proba(
            frame.iloc[validation_index][columns]
        )[:, 1]
    return probabilities


def _metrics(
    labels: np.ndarray, probabilities: np.ndarray, ring_mask: np.ndarray
) -> dict[str, float]:
    actions = probabilities >= 0.5
    return {
        "pr_auc": round(float(average_precision_score(labels, probabilities)), 8),
        "expected_cost": round(_binary_cost(labels, actions), 8),
        "ring_recall": round(_safe_recall(labels[ring_mask], actions[ring_mask]), 8),
        "risk_recall": round(_safe_recall(labels, actions), 8),
        "hold_rate": round(float(np.mean(actions)), 8),
    }


def _graphsage_grouped_probabilities(
    snapshots, labels: np.ndarray, groups, *, seed: int, folds: int, epochs: int
) -> np.ndarray:
    probabilities = np.zeros(len(labels), dtype=float)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    dummy = np.zeros((len(labels), 1), dtype=float)
    for fold_index, (train_index, validation_index) in enumerate(
        splitter.split(dummy, labels, groups), start=1
    ):
        model = fit_graphsage_snapshots(
            [snapshots[index] for index in train_index],
            [int(labels[index]) for index in train_index],
            epochs=epochs,
            seed=seed + fold_index,
        )
        for index in validation_index:
            probabilities[index] = predict_graphsage_probability(model, snapshots[index])
    return probabilities


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leakage-safe Phase-3 graph statistics vs GraphSAGE research gate"
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/research/phase3-graphsage"))
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--merchants-per-family", type=int, default=4)
    parser.add_argument("--transactions-per-merchant", type=int, default=32)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    if args.merchants_per_family < 2:
        raise SystemExit("--merchants-per-family must be >= 2")
    if args.transactions_per_merchant < 12:
        raise SystemExit("--transactions-per-merchant must be >= 12")
    if args.folds < 2:
        raise SystemExit("--folds must be >= 2")
    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")

    merchants, transactions, holds = generate_dataset(
        seed=args.seed,
        merchants_per_family=args.merchants_per_family,
        transactions_per_merchant=args.transactions_per_merchant,
    )
    frame = build_graph_training_frame(merchants, transactions, holds)
    labels = (frame["true_risk_state"].astype(str) == TrueRiskState.RISKY).astype(int).to_numpy()
    groups = frame["merchant_id"].astype(str).to_numpy()
    ring_mask = frame["attack_family"].astype(str).isin(RING_ATTACK_FAMILIES).to_numpy()

    if len(np.unique(labels)) != 2:
        raise SystemExit("Phase-3 gate requires both legitimate and risky examples")
    if len(np.unique(groups)) < args.folds:
        raise SystemExit("Not enough merchant groups for requested folds")

    graph_stats_report = evaluate_graph_statistics_gate(frame, seed=args.seed, folds=args.folds)
    graph_columns = _graph_columns(frame)
    tabular_probabilities = _grouped_xgb_probabilities(
        frame, labels, groups, list(FEATURE_COLUMNS), seed=args.seed, folds=args.folds
    )
    graph_stats_probabilities = _grouped_xgb_probabilities(
        frame,
        labels,
        groups,
        [*FEATURE_COLUMNS, *graph_columns],
        seed=args.seed,
        folds=args.folds,
    )

    snapshots = [
        build_point_in_time_graph_snapshot(
            synthetic_hold.hold.merchant_id,
            transactions,
            synthetic_hold.hold.triggered_at,
        )
        for synthetic_hold in holds
    ]
    graphsage_probabilities = _graphsage_grouped_probabilities(
        snapshots,
        labels,
        groups,
        seed=args.seed,
        folds=args.folds,
        epochs=args.epochs,
    )

    tabular = _metrics(labels, tabular_probabilities, ring_mask)
    graph_stats = _metrics(labels, graph_stats_probabilities, ring_mask)
    graphsage = _metrics(labels, graphsage_probabilities, ring_mask)

    graph_stats_gate = bool(graph_stats_report.gate_passed)
    graphsage_incremental_gate = bool(
        graphsage["pr_auc"] >= graph_stats["pr_auc"] + 0.01
        and graphsage["expected_cost"] < graph_stats["expected_cost"]
        and graphsage["ring_recall"] >= graph_stats["ring_recall"]
    )
    if not graph_stats_gate:
        decision = "BLOCKED_BY_GRAPH_STATS_ABLATION"
    elif graphsage_incremental_gate:
        decision = "GRAPH_SAGE_RETAINS_RESEARCH_VALUE"
    else:
        decision = "REJECT_GRAPHSAGE_COMPLEXITY"

    report = {
        "schema_version": "1.0",
        "phase": 3,
        "decision": decision,
        "seed": args.seed,
        "merchants_per_family": args.merchants_per_family,
        "transactions_per_merchant": args.transactions_per_merchant,
        "folds": args.folds,
        "epochs": args.epochs,
        "case_count": len(labels),
        "transaction_count": len(transactions),
        "evaluation_mode": "GROUPED_DEVELOPMENT_POINT_IN_TIME_NO_SEALED_TEST",
        "fixed_threshold": 0.5,
        "costs": {"false_release": 100, "false_hold": 25},
        "tabular_baseline": tabular,
        "graph_statistics": graph_stats,
        "graphsage": graphsage,
        "graph_statistics_gate": graph_stats_gate,
        "graphsage_incremental_gate": graphsage_incremental_gate,
        "existing_graph_ablation_report": graph_stats_report.model_dump(mode="json"),
        "graph_feature_count": len(graph_columns),
        "graph_features": graph_columns,
        "safety": {
            "timestamp_rule": "transaction.timestamp < hold.triggered_at",
            "ring_id_used_as_feature": False,
            "sealed_test_used": False,
            "stress_set_used": False,
            "production_action_eligible": False,
            "serving_change_authorized": False,
            "automatic_promotion": False,
            "champion_remains": "xgb-if-settlement@2",
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "phase3_graphsage_gate_report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report: {destination}")


if __name__ == "__main__":
    main()
