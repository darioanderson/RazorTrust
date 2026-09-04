from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
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
from razortrust.ml.phase3b_v21_diagnostics import (
    embedding_drift_report,
    fit_sigmoid_score_calibrator,
    operating_metrics,
    probability_quality,
    quantile_calibration_bins,
    score_summary,
    select_threshold_strategies,
    threshold_sweep,
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


def _append_embeddings(
    frame: pd.DataFrame,
    embeddings: list[np.ndarray],
) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    if len(frame) != len(embeddings):
        raise ValueError("embedding count must match frame rows")
    matrix = np.vstack(embeddings)
    columns = [f"hetero_gnn_embedding_{index:02d}" for index in range(matrix.shape[1])]
    result = frame.copy()
    for index, column in enumerate(columns):
        result[column] = matrix[:, index]
    return result, columns, matrix


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


def _partition_report(
    labels: np.ndarray,
    raw: np.ndarray,
    calibrated: np.ndarray,
) -> dict[str, object]:
    return {
        "raw_probability_quality": probability_quality(labels, raw),
        "calibrated_probability_quality": probability_quality(labels, calibrated),
        "raw_score_summary": score_summary(labels, raw),
        "calibrated_score_summary": score_summary(labels, calibrated),
        "raw_calibration_bins": quantile_calibration_bins(labels, raw),
        "calibrated_calibration_bins": quantile_calibration_bins(labels, calibrated),
    }


def _evaluate_thresholds(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: dict[str, float | None],
    *,
    scenarios: np.ndarray | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, threshold in thresholds.items():
        result[name] = (
            operating_metrics(
                labels, probabilities, threshold, scenarios=scenarios
            )
            if threshold is not None
            else {"available": False}
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3B V2.1 calibration, policy and embedding-drift diagnostic"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--train-cases", type=int, default=720)
    parser.add_argument("--policy-cases", type=int, default=200)
    parser.add_argument("--test-cases", type=int, default=200)
    parser.add_argument("--representation-fraction", type=float, default=0.50)
    parser.add_argument("--calibration-fraction", type=float, default=0.35)
    parser.add_argument("--recall-floor", type=float, default=0.70)
    parser.add_argument("--sample-fpr-cap", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--label-delay-days", type=int, default=7)
    parser.add_argument("--graph-lookback-days", type=int, default=30)
    parser.add_argument("--max-history-events", type=int, default=500)
    args = parser.parse_args()

    if not 0.40 <= args.representation_fraction <= 0.65:
        raise SystemExit("--representation-fraction must be between 0.40 and 0.65")
    if not 0.20 <= args.calibration_fraction <= 0.50:
        raise SystemExit("--calibration-fraction must be between 0.20 and 0.50")
    if not 0.0 < args.recall_floor <= 1.0:
        raise SystemExit("--recall-floor must be in (0, 1]")
    if not 0.0 <= args.sample_fpr_cap < 1.0:
        raise SystemExit("--sample-fpr-cap must be in [0, 1)")
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
    later_train = train_indices[representation_count:]
    calibration_count = int(len(later_train) * args.calibration_fraction)
    if calibration_count < 40 or len(later_train) - calibration_count < 40:
        raise SystemExit("V2.1 requires at least 40 classifier and calibration cases")
    classifier_indices = later_train[:-calibration_count]
    calibration_indices = later_train[-calibration_count:]

    named_indices = (
        ("representation_train", representation_indices),
        ("classifier_train", classifier_indices),
        ("calibration", calibration_indices),
        ("policy", np.asarray(split.policy, dtype=np.int64)),
        ("test", np.asarray(split.test, dtype=np.int64)),
    )
    previous_max = None
    for name, indices in named_indices:
        labels_present = set(raw.iloc[indices]["TX_FRAUD"].astype(int).unique().tolist())
        if labels_present != {0, 1}:
            raise SystemExit(f"{name} must contain both legitimate and fraud cases")
        part_min = raw.iloc[indices]["TX_DATETIME"].min()
        part_max = raw.iloc[indices]["TX_DATETIME"].max()
        if previous_max is not None and not previous_max < part_min:
            raise RuntimeError(f"chronology overlap before {name}")
        previous_max = part_max

    representation_frame = build_pilot_frame(
        history,
        representation_indices,
        label_delay_days=args.label_delay_days,
    )
    classifier_train = build_pilot_frame(
        history,
        classifier_indices,
        label_delay_days=args.label_delay_days,
    )
    calibration = build_pilot_frame(
        history,
        calibration_indices,
        label_delay_days=args.label_delay_days,
    )
    policy = build_pilot_frame(history, split.policy, label_delay_days=args.label_delay_days)
    test = build_pilot_frame(history, split.test, label_delay_days=args.label_delay_days)

    representation_raw = _snapshot_batch(history, representation_indices, args)
    normalizer = fit_hetero_normalizer(representation_raw)
    representation_snapshots = [
        apply_hetero_normalizer(snapshot, normalizer) for snapshot in representation_raw
    ]
    gnn = fit_hetero_graphsage(
        representation_snapshots,
        representation_frame["label"].astype(int).tolist(),
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )

    snapshot_groups = {}
    for name, indices in (
        ("classifier_train", classifier_indices),
        ("calibration", calibration_indices),
        ("policy", np.asarray(split.policy, dtype=np.int64)),
        ("test", np.asarray(split.test, dtype=np.int64)),
    ):
        snapshot_groups[name] = [
            apply_hetero_normalizer(snapshot, normalizer)
            for snapshot in _snapshot_batch(history, indices, args)
        ]

    embedding_lists = {
        name: [hetero_graphsage_embedding(gnn, snapshot) for snapshot in snapshots]
        for name, snapshots in snapshot_groups.items()
    }
    gnn_probabilities = {
        name: np.asarray(
            [hetero_graphsage_probability(gnn, snapshot) for snapshot in snapshots],
            dtype=float,
        )
        for name, snapshots in snapshot_groups.items()
    }

    classifier_fusion, embedding_columns, classifier_matrix = _append_embeddings(
        classifier_train,
        embedding_lists["classifier_train"],
    )
    calibration_fusion, _, calibration_matrix = _append_embeddings(
        calibration,
        embedding_lists["calibration"],
    )
    policy_fusion, _, policy_matrix = _append_embeddings(policy, embedding_lists["policy"])
    test_fusion, _, test_matrix = _append_embeddings(test, embedding_lists["test"])

    tabular_columns = list(FDH_TABULAR_FEATURES)
    graph_columns = [*FDH_TABULAR_FEATURES, *FDH_GRAPH_FEATURES]
    fusion_columns = [*graph_columns, *embedding_columns]

    models = {
        "tabular": (
            _fit_xgb(classifier_train, tabular_columns, args.seed),
            tabular_columns,
            classifier_train,
            calibration,
            policy,
            test,
        ),
        "graph_statistics": (
            _fit_xgb(classifier_train, graph_columns, args.seed),
            graph_columns,
            classifier_train,
            calibration,
            policy,
            test,
        ),
        "hetero_fusion": (
            _fit_xgb(classifier_fusion, fusion_columns, args.seed),
            fusion_columns,
            classifier_fusion,
            calibration_fusion,
            policy_fusion,
            test_fusion,
        ),
    }

    model_reports: dict[str, object] = {}
    posthoc_sweeps: dict[str, object] = {
        "schema_version": "2.1",
        "selection_eligible": False,
        "warning": (
            "TEST threshold sweeps are post-hoc diagnostics only and must not be used "
            "for model, calibrator, threshold, or serving selection."
        ),
        "models": {},
    }

    for name, (model, columns, _, calib_frame, policy_frame, test_frame) in models.items():
        raw_calib = model.predict_proba(calib_frame[columns])[:, 1]
        raw_policy = model.predict_proba(policy_frame[columns])[:, 1]
        raw_test = model.predict_proba(test_frame[columns])[:, 1]
        calibration_labels = calib_frame["label"].to_numpy(dtype=int)
        policy_labels = policy_frame["label"].to_numpy(dtype=int)
        test_labels = test_frame["label"].to_numpy(dtype=int)
        policy_scenarios = policy_frame["scenario"].to_numpy(dtype=int)
        test_scenarios = test_frame["scenario"].to_numpy(dtype=int)

        calibrator = fit_sigmoid_score_calibrator(raw_calib, calibration_labels)
        calibrated_calib = calibrator.transform(raw_calib)
        calibrated_policy = calibrator.transform(raw_policy)
        calibrated_test = calibrator.transform(raw_test)

        raw_reference = select_threshold_strategies(
            raw_policy,
            policy_labels,
            recall_floor=args.recall_floor,
            sample_fpr_cap=args.sample_fpr_cap,
        )
        calibrated_thresholds = select_threshold_strategies(
            calibrated_policy,
            policy_labels,
            recall_floor=args.recall_floor,
            sample_fpr_cap=args.sample_fpr_cap,
        )
        raw_v2_threshold = {"max_f1_high_tie": raw_reference["max_f1_high_tie"]}

        model_reports[name] = {
            "calibrator": calibrator.as_dict(),
            "calibration": _partition_report(
                calibration_labels,
                raw_calib,
                calibrated_calib,
            ),
            "policy": {
                **_partition_report(policy_labels, raw_policy, calibrated_policy),
                "raw_v2_reference_operating_point": _evaluate_thresholds(
                    policy_labels,
                    raw_policy,
                    raw_v2_threshold,
                    scenarios=policy_scenarios,
                ),
                "calibrated_threshold_strategies": calibrated_thresholds,
                "calibrated_operating_points": _evaluate_thresholds(
                    policy_labels,
                    calibrated_policy,
                    calibrated_thresholds,
                    scenarios=policy_scenarios,
                ),
            },
            "test": {
                **_partition_report(test_labels, raw_test, calibrated_test),
                "raw_v2_reference_operating_point": _evaluate_thresholds(
                    test_labels,
                    raw_test,
                    raw_v2_threshold,
                    scenarios=test_scenarios,
                ),
                "calibrated_operating_points_frozen_from_policy": _evaluate_thresholds(
                    test_labels,
                    calibrated_test,
                    calibrated_thresholds,
                    scenarios=test_scenarios,
                ),
            },
        }
        posthoc_sweeps["models"][name] = {
            "raw_test": threshold_sweep(test_labels, raw_test),
            "calibrated_test": threshold_sweep(test_labels, calibrated_test),
        }

    for name in ("gnn_standalone",):
        raw_calib = gnn_probabilities["calibration"]
        raw_policy = gnn_probabilities["policy"]
        raw_test = gnn_probabilities["test"]
        calibration_labels = calibration["label"].to_numpy(dtype=int)
        policy_labels = policy["label"].to_numpy(dtype=int)
        test_labels = test["label"].to_numpy(dtype=int)
        policy_scenarios = policy["scenario"].to_numpy(dtype=int)
        test_scenarios = test["scenario"].to_numpy(dtype=int)
        calibrator = fit_sigmoid_score_calibrator(raw_calib, calibration_labels)
        calibrated_calib = calibrator.transform(raw_calib)
        calibrated_policy = calibrator.transform(raw_policy)
        calibrated_test = calibrator.transform(raw_test)
        calibrated_thresholds = select_threshold_strategies(
            calibrated_policy,
            policy_labels,
            recall_floor=args.recall_floor,
            sample_fpr_cap=args.sample_fpr_cap,
        )
        model_reports[name] = {
            "diagnostic_only": True,
            "calibrator": calibrator.as_dict(),
            "calibration": _partition_report(
                calibration_labels,
                raw_calib,
                calibrated_calib,
            ),
            "policy": {
                **_partition_report(policy_labels, raw_policy, calibrated_policy),
                "calibrated_threshold_strategies": calibrated_thresholds,
                "calibrated_operating_points": _evaluate_thresholds(
                    policy_labels,
                    calibrated_policy,
                    calibrated_thresholds,
                    scenarios=policy_scenarios,
                ),
            },
            "test": {
                **_partition_report(test_labels, raw_test, calibrated_test),
                "calibrated_operating_points_frozen_from_policy": _evaluate_thresholds(
                    test_labels,
                    calibrated_test,
                    calibrated_thresholds,
                    scenarios=test_scenarios,
                ),
            },
        }
        posthoc_sweeps["models"][name] = {
            "raw_test": threshold_sweep(test_labels, raw_test),
            "calibrated_test": threshold_sweep(test_labels, calibrated_test),
        }

    embedding_drift = embedding_drift_report(
        {
            "classifier_train": classifier_matrix,
            "calibration": calibration_matrix,
            "policy": policy_matrix,
            "test": test_matrix,
        }
    )

    def time_range(indices: np.ndarray) -> list[str]:
        part = raw.iloc[indices]
        return [str(part["TX_DATETIME"].min()), str(part["TX_DATETIME"].max())]

    report = {
        "schema_version": "2.1",
        "phase": "3B-V2.1",
        "benchmark_label": "PUBLIC SYNTHETIC GRAPH CALIBRATION / POLICY DIAGNOSTIC",
        "decision": "DIAGNOSTIC_ONLY_NO_MODEL_SELECTION",
        "diagnostic_question": (
            "Does V2 heterogeneous GraphSAGE ranking value survive a separate sample "
            "calibration partition and alternative policy operating points, and do "
            "its embeddings drift across future partitions?"
        ),
        "domain": (
            "Fraud Detection Handbook simulated transaction fraud; "
            "not Razorpay settlement-hold ground truth"
        ),
        "dataset_rows": int(len(raw)),
        "dataset_fraud": int(raw["TX_FRAUD"].sum()),
        "sampling_warning": (
            "Chronological case-control research sample. Calibrated values are sample "
            "calibration diagnostics and are not population fraud probabilities."
        ),
        "partition_rows": {
            "representation_train": int(len(representation_indices)),
            "classifier_train": int(len(classifier_indices)),
            "calibration": int(len(calibration_indices)),
            "policy": int(len(split.policy)),
            "test": int(len(split.test)),
        },
        "partition_time_ranges": {
            "representation_train": time_range(representation_indices),
            "classifier_train": time_range(classifier_indices),
            "calibration": time_range(calibration_indices),
            "policy": time_range(np.asarray(split.policy, dtype=np.int64)),
            "test": time_range(np.asarray(split.test, dtype=np.int64)),
        },
        "methodology": {
            "representation_fraction": args.representation_fraction,
            "calibration_fraction_of_post_representation_train": args.calibration_fraction,
            "calibration_method": "sigmoid on logit(raw score), calibration partition only",
            "isotonic_used": False,
            "isotonic_reason": "small diagnostic calibration partition; avoid overfit",
            "policy_recall_floor": args.recall_floor,
            "policy_sample_fpr_cap": args.sample_fpr_cap,
            "threshold_grid": "0.01..0.99 step 0.01 on POLICY only",
            "test_threshold_sweep": "post-hoc diagnostic only; selection-ineligible",
            "normalization_fit_partition": "representation_train only",
            "gnn_fit_partition": "representation_train only",
            "xgb_fit_partition": "classifier_train only",
            "calibrator_fit_partition": "calibration only",
            "threshold_selection_partition": "policy only",
            "test_model_selection_eligible": False,
        },
        "models": model_reports,
        "embedding_drift": embedding_drift,
        "safety": {
            "existing_phase3_decisions_overwritten": False,
            "phase3b_v2_decision_overwritten": False,
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
    report_path = args.output / "phase3b_v21_calibration_diagnostic_report.json"
    sweep_path = args.output / "phase3b_v21_posthoc_test_threshold_sweeps.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    sweep_path.write_text(
        json.dumps(posthoc_sweeps, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print(f"Report: {report_path}")
    print(f"Post-hoc threshold sweeps: {sweep_path}")


if __name__ == "__main__":
    main()
