from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from razortrust.ml.drift_validation import (
    binary_monitoring_metrics,
    chronological_drift_partitions,
    inject_standard_deviation_shift,
    online_change_summary,
    performance_delta,
    severity_from_signals,
)
from razortrust.ml.monitoring import batch_drift_report, write_evidently_drift_report
from razortrust.ml.public_benchmark import (
    calibrate_probabilities,
    fit_calibrator,
    select_operating_point,
    serialize,
)


def _parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def _parse_str_list(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one feature")
    return values


def _content_sha256(frame: pd.DataFrame) -> str:
    row_hashes = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64)
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def _class_check(labels: np.ndarray, name: str) -> None:
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise SystemExit(f"{name} partition must contain both classes")


def _score_batch_report(reference: np.ndarray, current: np.ndarray) -> dict[str, object]:
    report = batch_drift_report(
        pd.DataFrame({"calibrated_risk_score": np.asarray(reference, dtype=float)}),
        pd.DataFrame({"calibrated_risk_score": np.asarray(current, dtype=float)}),
    )
    return report.model_dump(mode="json")


def _binary_errors(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> np.ndarray:
    predicted = np.asarray(probabilities, dtype=float) >= threshold
    return (predicted != np.asarray(labels, dtype=int)).astype(float)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 5 public chronological drift validation: natural drift, controlled "
            "feature injection, score drift, and delayed-label performance drift"
        )
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="Class")
    parser.add_argument("--time-column", default="Time")
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument(
        "--injection-features",
        type=_parse_str_list,
        default=["V1", "V2", "V3"],
    )
    parser.add_argument(
        "--injection-strengths",
        type=_parse_float_list,
        default=[0.25, 0.50, 1.00],
    )
    parser.add_argument("--adwin-delta", type=float, default=0.002)
    args = parser.parse_args()

    raw = pd.read_csv(args.csv, low_memory=False)
    if args.label not in raw:
        raise SystemExit(f"label column not found: {args.label}")
    if args.time_column not in raw:
        raise SystemExit(
            "Phase 5 public drift validation requires a chronological Time column"
        )

    source_sha256 = _content_sha256(raw)
    raw[args.time_column] = pd.to_numeric(raw[args.time_column], errors="coerce")
    if raw[args.time_column].isna().any():
        raise SystemExit("time column contains non-numeric or missing values")
    raw = raw.sort_values(args.time_column, kind="mergesort").reset_index(drop=True)

    labels = pd.to_numeric(raw.pop(args.label), errors="raise").astype(int).to_numpy()
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise SystemExit("label must be binary and encoded as 0/1")

    times = raw[args.time_column].astype(float).to_numpy()
    raw = raw.drop(columns=[args.time_column])
    frame = (
        raw.select_dtypes(include=["number"])
        .replace([np.inf, -np.inf], np.nan)
        .astype(float)
    )
    if frame.shape[1] == 0:
        raise SystemExit("no numeric features found")

    parts = chronological_drift_partitions(len(frame))
    train_medians = frame.iloc[parts.train].median(numeric_only=True).fillna(0.0)
    missing_before = int(frame.isna().sum().sum())
    frame = frame.fillna(train_medians).fillna(0.0)
    if frame.isna().any().any():
        raise RuntimeError("missing values remain after train-only median imputation")

    for name, index in (
        ("train", parts.train),
        ("calibration_fit", parts.calibration_fit),
        ("calibration_select", parts.calibration_select),
        ("policy", parts.policy),
        ("monitor", parts.monitor),
    ):
        _class_check(labels[index], name)

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=args.n_estimators,
        max_depth=6,
        learning_rate=0.03,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
        tree_method="hist",
    )
    model.fit(
        frame.iloc[parts.train],
        labels[parts.train],
        sample_weight=compute_sample_weight(
            class_weight="balanced",
            y=labels[parts.train],
        ),
        verbose=False,
    )

    calibration_fit_raw = model.predict_proba(frame.iloc[parts.calibration_fit])[:, 1]
    calibration_select_raw = model.predict_proba(
        frame.iloc[parts.calibration_select]
    )[:, 1]
    calibration_choice, calibrator = fit_calibrator(
        calibration_fit_raw,
        labels[parts.calibration_fit],
        calibration_select_raw,
        labels[parts.calibration_select],
    )

    policy_probabilities = calibrate_probabilities(
        model.predict_proba(frame.iloc[parts.policy])[:, 1],
        calibration_choice,
        calibrator,
    )
    balanced = select_operating_point(
        policy_probabilities,
        labels[parts.policy],
        name="BALANCED",
        max_fpr=0.01,
        objective="f1",
    )
    threshold = float(balanced.threshold)

    policy_metrics = binary_monitoring_metrics(
        labels[parts.policy],
        policy_probabilities,
        threshold,
    )
    policy_features = frame.iloc[parts.policy].reset_index(drop=True)

    natural_windows: list[dict[str, object]] = []

    for window_number, index in enumerate(parts.monitor_windows, start=1):
        current_features = frame.iloc[index].reset_index(drop=True)
        current_probabilities = calibrate_probabilities(
            model.predict_proba(frame.iloc[index])[:, 1],
            calibration_choice,
            calibrator,
        )
        current_labels = labels[index]

        feature_drift = batch_drift_report(policy_features, current_features)
        score_batch = _score_batch_report(policy_probabilities, current_probabilities)
        score_online = online_change_summary(
            policy_probabilities,
            current_probabilities,
            adwin_delta=args.adwin_delta,
            seed=42 + window_number,
        )

        reference_errors = _binary_errors(
            labels[parts.policy],
            policy_probabilities,
            threshold,
        )
        current_errors = _binary_errors(
            current_labels,
            current_probabilities,
            threshold,
        )
        delayed_error_online = online_change_summary(
            reference_errors,
            current_errors,
            adwin_delta=args.adwin_delta,
            seed=142 + window_number,
        )

        current_metrics = binary_monitoring_metrics(
            current_labels,
            current_probabilities,
            threshold,
        )
        delta = performance_delta(policy_metrics, current_metrics)
        score_drifted = bool(
            score_batch["features"][0]["drifted"] or score_online["any_detected"]
        )
        severity, severity_checks = severity_from_signals(
            drifted_feature_share=float(feature_drift.drifted_feature_share),
            score_drifted=score_drifted,
            performance_delta_values=delta,
        )

        natural_windows.append(
            {
                "window": window_number,
                "rows": len(index),
                "time_min": float(times[index[0]]),
                "time_max": float(times[index[-1]]),
                "feature_drift": feature_drift.model_dump(mode="json"),
                "score_batch_drift": score_batch,
                "score_online_change": score_online,
                "delayed_error_online_change": delayed_error_online,
                "performance": current_metrics,
                "performance_delta_vs_policy": delta,
                "severity": severity,
                "severity_checks": severity_checks,
            }
        )

    injected_results: list[dict[str, object]] = []
    monitor_features = frame.iloc[parts.monitor].reset_index(drop=True)
    monitor_labels = labels[parts.monitor]

    for strength in args.injection_strengths:
        injected = inject_standard_deviation_shift(
            monitor_features,
            reference=policy_features,
            features=args.injection_features,
            strength=float(strength),
        )
        injected_probabilities = calibrate_probabilities(
            model.predict_proba(injected)[:, 1],
            calibration_choice,
            calibrator,
        )
        feature_drift = batch_drift_report(policy_features, injected)
        score_batch = _score_batch_report(policy_probabilities, injected_probabilities)
        score_online = online_change_summary(
            policy_probabilities,
            injected_probabilities,
            adwin_delta=args.adwin_delta,
            seed=1000 + int(round(float(strength) * 100)),
        )
        injected_metrics = binary_monitoring_metrics(
            monitor_labels,
            injected_probabilities,
            threshold,
        )
        delta = performance_delta(policy_metrics, injected_metrics)
        score_drifted = bool(
            score_batch["features"][0]["drifted"] or score_online["any_detected"]
        )
        severity, severity_checks = severity_from_signals(
            drifted_feature_share=float(feature_drift.drifted_feature_share),
            score_drifted=score_drifted,
            performance_delta_values=delta,
        )

        injected_results.append(
            {
                "strength_reference_std": float(strength),
                "injected_features": list(args.injection_features),
                "feature_drift": feature_drift.model_dump(mode="json"),
                "score_batch_drift": score_batch,
                "score_online_change": score_online,
                "performance": injected_metrics,
                "performance_delta_vs_policy": delta,
                "severity": severity,
                "severity_checks": severity_checks,
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    write_evidently_drift_report(
        policy_features,
        monitor_features,
        html_path=str(args.output / "evidently-natural-policy-vs-monitor.html"),
        json_path=str(args.output / "evidently-natural-policy-vs-monitor.json"),
    )
    strongest_injected = inject_standard_deviation_shift(
        monitor_features,
        reference=policy_features,
        features=args.injection_features,
        strength=max(args.injection_strengths),
    )
    write_evidently_drift_report(
        policy_features,
        strongest_injected,
        html_path=str(args.output / "evidently-controlled-strongest.html"),
        json_path=str(args.output / "evidently-controlled-strongest.json"),
    )

    natural_false_alarm_windows = sum(
        row["severity"] in {"AMBER", "RED"} for row in natural_windows
    )
    controlled_detection_count = sum(
        float(row["feature_drift"]["drifted_feature_share"]) > 0.0
        for row in injected_results
    )

    report = {
        "schema_version": "1.0",
        "phase": 5,
        "benchmark_label": "PUBLIC HELD-OUT DRIFT DIAGNOSTIC",
        "domain": "ULB/OpenML transaction fraud; not Razorpay settlement-hold ground truth",
        "source": str(args.csv),
        "source_content_sha256": source_sha256,
        "rows": len(frame),
        "features": int(frame.shape[1]),
        "chronological_claim": True,
        "evaluation": (
            "55% train / 15% calibration / 15% policy-reference / "
            "15% future monitor split into 3 chronological windows"
        ),
        "preprocessing": {
            "train_only_median_imputation": True,
            "missing_values_before_imputation": missing_before,
        },
        "model": "XGBoost binary hist; public research only",
        "calibration": serialize(calibration_choice),
        "operating_point": serialize(balanced),
        "reference_semantics": {
            "feature_and_score_reference": "policy partition",
            "performance_reference": (
                "policy partition; note threshold was selected on this partition, "
                "so performance deltas are diagnostic rather than unbiased estimates"
            ),
        },
        "policy_reference_metrics": policy_metrics,
        "natural_chronological_windows": natural_windows,
        "controlled_injection": {
            "features": list(args.injection_features),
            "strengths_reference_std": list(args.injection_strengths),
            "results": injected_results,
        },
        "online_stream_semantics": {
            "score_stream": "calibrated fraud probability; label-free",
            "delayed_error_stream": (
                "0/1 threshold classification error; requires labels and is delayed-label only"
            ),
            "adwin_delta": args.adwin_delta,
            "kswin_alpha": 0.005,
        },
        "severity_policy": {
            "GREEN": "no material distribution or performance signal",
            "AMBER": "distribution signal or material performance degradation",
            "RED": "distribution signal and material performance degradation",
            "distribution_signal": (
                "drifted_feature_share >= 0.10 OR score batch/online drift"
            ),
            "performance_degraded": (
                "PR-AUC delta <= -0.05 OR recall delta <= -0.10 "
                "OR sample-FPR delta >= +0.01"
            ),
            "research_only": True,
        },
        "diagnostic_summary": {
            "natural_amber_or_red_window_count": natural_false_alarm_windows,
            "natural_window_count": len(natural_windows),
            "controlled_feature_drift_detection_count": controlled_detection_count,
            "controlled_scenario_count": len(injected_results),
        },
        "limitations": [
            "The ULB dataset covers a short historical period and anonymized PCA features.",
            "Natural-window alerts are not production false-positive-rate estimates.",
            "Policy reference performance is selection-used and therefore diagnostic.",
            "Controlled shifts are detector-power probes, not claims about real fraud mechanisms.",
            "No automatic retraining, model promotion, or serving change is authorized.",
        ],
        "safety": {
            "production_action_eligible": False,
            "serving_change_authorized": False,
            "automatic_retraining": False,
            "automatic_promotion": False,
            "automatic_release_enabled": False,
            "sealed_test_used": False,
            "stress_set_used": False,
            "benchmark_does_not_change_serving_champion": True,
            "champion_remains": "xgb-if-settlement@2",
            "active_enforcement_runtime": "human-only@1",
        },
    }

    output_path = args.output / "phase5_public_drift_validation_report.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
