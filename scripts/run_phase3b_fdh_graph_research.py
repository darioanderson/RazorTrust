from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

from razortrust.ml.fdh_graph import (
    FDH_GRAPH_FEATURES,
    FDH_TABULAR_FEATURES,
    build_history_index,
    build_pilot_frame,
    build_point_in_time_graph_snapshot,
    load_fdh_transactions,
    select_chronological_case_control_pilot,
)
from razortrust.ml.graphsage import fit_graphsage_snapshots, predict_graphsage_probability


def _fit_xgb(frame, columns: list[str], seed: int):
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
            (
                float(recall_score(labels[mask], predicted[mask], zero_division=0))
                if np.any(mask)
                else 0.0
            ),
            8,
        )
        result[f"scenario_{scenario}_cases"] = int(np.sum(mask))
    return result


def _probabilities(model, frame, columns: list[str]) -> np.ndarray:
    return model.predict_proba(frame[columns])[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3B public structured FDH graph pilot")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--train-cases", type=int, default=240)
    parser.add_argument("--policy-cases", type=int, default=120)
    parser.add_argument("--test-cases", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--label-delay-days", type=int, default=7)
    parser.add_argument("--graph-lookback-days", type=int, default=30)
    parser.add_argument("--max-history-events", type=int, default=400)
    args = parser.parse_args()

    raw = load_fdh_transactions(args.data_dir)
    history = build_history_index(raw)
    split = select_chronological_case_control_pilot(
        raw,
        seed=args.seed,
        max_train=args.train_cases,
        max_policy=args.policy_cases,
        max_test=args.test_cases,
    )
    train = build_pilot_frame(history, split.train, label_delay_days=args.label_delay_days)
    policy = build_pilot_frame(history, split.policy, label_delay_days=args.label_delay_days)
    test = build_pilot_frame(history, split.test, label_delay_days=args.label_delay_days)

    tabular_columns = list(FDH_TABULAR_FEATURES)
    graph_columns = [*FDH_TABULAR_FEATURES, *FDH_GRAPH_FEATURES]
    tabular_model = _fit_xgb(train, tabular_columns, args.seed)
    graph_stats_model = _fit_xgb(train, graph_columns, args.seed)

    tab_policy_probs = _probabilities(tabular_model, policy, tabular_columns)
    stats_policy_probs = _probabilities(graph_stats_model, policy, graph_columns)
    tab_threshold = _select_threshold(tab_policy_probs, policy["label"].to_numpy(dtype=int))
    stats_threshold = _select_threshold(stats_policy_probs, policy["label"].to_numpy(dtype=int))

    train_snapshots = [
        build_point_in_time_graph_snapshot(
            history,
            int(index),
            lookback_days=args.graph_lookback_days,
            label_delay_days=args.label_delay_days,
            max_history_events=args.max_history_events,
        )
        for index in split.train
    ]
    graphsage_model = fit_graphsage_snapshots(
        train_snapshots,
        train["label"].astype(int).tolist(),
        epochs=args.epochs,
        seed=args.seed,
    )
    policy_snapshots = [
        build_point_in_time_graph_snapshot(
            history,
            int(index),
            lookback_days=args.graph_lookback_days,
            label_delay_days=args.label_delay_days,
            max_history_events=args.max_history_events,
        )
        for index in split.policy
    ]
    graph_policy_probs = np.asarray(
        [predict_graphsage_probability(graphsage_model, snapshot) for snapshot in policy_snapshots],
        dtype=float,
    )
    graph_threshold = _select_threshold(graph_policy_probs, policy["label"].to_numpy(dtype=int))

    test_snapshots = [
        build_point_in_time_graph_snapshot(
            history,
            int(index),
            lookback_days=args.graph_lookback_days,
            label_delay_days=args.label_delay_days,
            max_history_events=args.max_history_events,
        )
        for index in split.test
    ]
    graph_test_probs = np.asarray(
        [predict_graphsage_probability(graphsage_model, snapshot) for snapshot in test_snapshots],
        dtype=float,
    )

    labels = test["label"].to_numpy(dtype=int)
    scenarios = test["scenario"].to_numpy(dtype=int)
    tabular = _metrics(
        labels,
        scenarios,
        _probabilities(tabular_model, test, tabular_columns),
        tab_threshold,
    )
    graph_stats = _metrics(
        labels,
        scenarios,
        _probabilities(graph_stats_model, test, graph_columns),
        stats_threshold,
    )
    graphsage = _metrics(labels, scenarios, graph_test_probs, graph_threshold)

    graph_stats_gate = bool(
        float(graph_stats["pr_auc"]) >= float(tabular["pr_auc"]) + 0.01
        and float(graph_stats["recall"]) >= float(tabular["recall"])
        and float(graph_stats["sample_fpr"]) <= float(tabular["sample_fpr"]) + 0.02
        and max(
            float(graph_stats["scenario_2_recall"]) - float(tabular["scenario_2_recall"]),
            float(graph_stats["scenario_3_recall"]) - float(tabular["scenario_3_recall"]),
        )
        > 0.0
    )
    graphsage_gate = bool(
        graph_stats_gate
        and float(graphsage["pr_auc"]) >= float(graph_stats["pr_auc"]) + 0.01
        and float(graphsage["recall"]) >= float(graph_stats["recall"])
        and float(graphsage["sample_fpr"]) <= float(graph_stats["sample_fpr"]) + 0.02
        and max(
            float(graphsage["scenario_2_recall"])
            - float(graph_stats["scenario_2_recall"]),
            float(graphsage["scenario_3_recall"])
            - float(graph_stats["scenario_3_recall"]),
        )
        > 0.0
    )
    if not graph_stats_gate:
        decision = "PHASE3B_BLOCKED_BY_GRAPH_STATS_ABLATION"
    elif graphsage_gate:
        decision = "PHASE3B_GRAPHSAGE_RETAINS_PUBLIC_RESEARCH_VALUE"
    else:
        decision = "PHASE3B_REJECT_GRAPHSAGE_COMPLEXITY"

    report = {
        "schema_version": "1.0",
        "phase": "3B",
        "benchmark_label": "PUBLIC SYNTHETIC STRUCTURED GRAPH PILOT",
        "domain": (
            "Fraud Detection Handbook simulated transaction fraud; "
            "not Razorpay settlement-hold ground truth"
        ),
        "decision": decision,
        "dataset_rows": len(raw),
        "dataset_fraud": int(raw["TX_FRAUD"].sum()),
        "pilot_sampling": (
            "chronological case-control pilot; metrics are comparative research metrics, "
            "not population operating rates"
        ),
        "partition_rows": {"train": len(train), "policy": len(policy), "test": len(test)},
        "partition_time_ranges": {
            "train": [str(train["timestamp"].min()), str(train["timestamp"].max())],
            "policy": [str(policy["timestamp"].min()), str(policy["timestamp"].max())],
            "test": [str(test["timestamp"].min()), str(test["timestamp"].max())],
        },
        "label_delay_days": args.label_delay_days,
        "graph_lookback_days": args.graph_lookback_days,
        "max_history_events_per_snapshot": args.max_history_events,
        "tabular_baseline": tabular,
        "graph_statistics": graph_stats,
        "graphsage": graphsage,
        "graph_statistics_gate": graph_stats_gate,
        "graphsage_incremental_gate": graphsage_gate,
        "safety": {
            "history_topology_rule": (
                "historical transaction timestamp < current transaction timestamp"
            ),
            "current_transaction_label_used_as_feature": False,
            "future_labels_used": False,
            "fraud_derived_history_available_only_after_delay_days": args.label_delay_days,
            "original_phase3_decision_overwritten": False,
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
    destination = args.output / "phase3b_fdh_graph_pilot_report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report: {destination}")


if __name__ == "__main__":
    main()
