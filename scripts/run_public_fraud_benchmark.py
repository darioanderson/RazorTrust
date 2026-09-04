from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import shutil
import time
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, get_data_home
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from razortrust.ml.public_benchmark import (
    calibrate_probabilities,
    evaluate_held_out,
    fit_calibrator,
    select_operating_point,
    serialize,
)

ULB_OPENML_DATA_ID = 1597


def _clear_openml_cache() -> Path:
    cache_dir = Path(get_data_home()) / "openml"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
    return cache_dir


def _load_openml_frame(
    data_id: int,
    *,
    attempts: int,
    retry_delay_seconds: float,
) -> tuple[pd.DataFrame, str]:
    if attempts < 1:
        raise ValueError("openml attempts must be >= 1")
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            cache_dir = _clear_openml_cache()
            print(
                f"OpenML retry {attempt}/{attempts}: cleared cache {cache_dir}",
                flush=True,
            )
            if retry_delay_seconds > 0:
                time.sleep(retry_delay_seconds)
        try:
            dataset = fetch_openml(
                data_id=data_id,
                as_frame=True,
                parser="auto",
                n_retries=3,
                delay=max(1.0, retry_delay_seconds),
            )
            if dataset.frame is None:
                raise RuntimeError("OpenML did not return a dataframe")
            return dataset.frame.copy(), f"openml:{data_id}"
        except (
            ConnectionError,
            TimeoutError,
            OSError,
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
            urllib.error.URLError,
        ) as exc:
            last_error = exc
            print(
                f"OpenML attempt {attempt}/{attempts} failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
    assert last_error is not None
    raise RuntimeError(
        "OpenML download failed after "
        f"{attempts} attempts. This is a data-transfer failure, not an ML result. "
        "Use --csv <path-to-creditcard.csv> to run the identical benchmark from a "
        "local copy, or retry later. No report was produced."
    ) from last_error


def _load_frame(
    csv: Path | None,
    openml_data_id: int | None,
    *,
    openml_attempts: int,
    openml_retry_delay_seconds: float,
) -> tuple[pd.DataFrame, str]:
    if csv is not None:
        if not csv.is_file():
            raise SystemExit(f"CSV file not found: {csv}")
        return pd.read_csv(csv), f"csv:{csv.resolve()}"
    data_id = openml_data_id or ULB_OPENML_DATA_ID
    return _load_openml_frame(
        data_id,
        attempts=openml_attempts,
        retry_delay_seconds=openml_retry_delay_seconds,
    )


def _stratified_indices(labels: np.ndarray) -> dict[str, np.ndarray]:
    index = np.arange(len(labels))
    train, remainder = train_test_split(
        index,
        train_size=0.55,
        random_state=42,
        stratify=labels,
    )
    calibration, policy_test = train_test_split(
        remainder,
        train_size=1.0 / 3.0,
        random_state=43,
        stratify=labels[remainder],
    )
    policy, test = train_test_split(
        policy_test,
        test_size=0.5,
        random_state=44,
        stratify=labels[policy_test],
    )
    calibration_fit, calibration_select = train_test_split(
        calibration,
        test_size=0.5,
        random_state=45,
        stratify=labels[calibration],
    )
    return {
        "train": np.sort(train),
        "calibration": np.sort(calibration),
        "calibration_fit": np.sort(calibration_fit),
        "calibration_select": np.sort(calibration_select),
        "policy": np.sort(policy),
        "test": np.sort(test),
    }


def _chronological_indices(labels: np.ndarray) -> dict[str, np.ndarray]:
    n = len(labels)
    train_end = int(n * 0.55)
    calibration_end = int(n * 0.70)
    policy_end = int(n * 0.85)
    calibration_mid = train_end + ((calibration_end - train_end) // 2)
    return {
        "train": np.arange(0, train_end),
        "calibration": np.arange(train_end, calibration_end),
        "calibration_fit": np.arange(train_end, calibration_mid),
        "calibration_select": np.arange(calibration_mid, calibration_end),
        "policy": np.arange(calibration_end, policy_end),
        "test": np.arange(policy_end, n),
    }


def _validate_partitions(labels: np.ndarray, partitions: dict[str, np.ndarray]) -> None:
    required = ("train", "calibration_fit", "calibration_select", "policy", "test")
    for name in required:
        if len(np.unique(labels[partitions[name]])) != 2:
            raise SystemExit(f"{name} partition must contain both classes")
    disjoint = ("train", "calibration", "policy", "test")
    sets = {name: set(partitions[name].tolist()) for name in disjoint}
    for index, left in enumerate(disjoint):
        for right in disjoint[index + 1 :]:
            if sets[left].intersection(sets[right]):
                raise RuntimeError(f"partition overlap detected: {left} / {right}")


def _content_sha256(frame: pd.DataFrame) -> str:
    row_hashes = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64)
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leakage-safe public fraud benchmark with untouched held-out test metrics"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--csv", type=Path)
    source.add_argument("--openml-data-id", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="Class")
    parser.add_argument("--time-column", default="Time")
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--openml-attempts", type=int, default=5)
    parser.add_argument("--openml-retry-delay-seconds", type=float, default=5.0)
    args = parser.parse_args()

    raw, source_name = _load_frame(
        args.csv,
        args.openml_data_id,
        openml_attempts=args.openml_attempts,
        openml_retry_delay_seconds=args.openml_retry_delay_seconds,
    )
    if args.label not in raw:
        raise SystemExit(f"label column not found: {args.label}")
    raw_content_sha256 = _content_sha256(raw)

    time_column_present = args.time_column in raw
    if time_column_present:
        raw[args.time_column] = pd.to_numeric(raw[args.time_column], errors="coerce")
        if raw[args.time_column].isna().any():
            raise SystemExit("time column contains non-numeric or missing values")
        raw = raw.sort_values(args.time_column, kind="mergesort").reset_index(drop=True)

    labels = pd.to_numeric(raw.pop(args.label), errors="raise").astype(int).to_numpy()
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise SystemExit("label must be binary and encoded as 0/1")
    if time_column_present:
        raw = raw.drop(columns=[args.time_column])

    frame = raw.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan).astype(float)
    if frame.shape[1] == 0:
        raise SystemExit("no numeric benchmark features found")

    partitions = (
        _chronological_indices(labels) if time_column_present else _stratified_indices(labels)
    )
    _validate_partitions(labels, partitions)

    train_index = partitions["train"]
    train_medians = frame.iloc[train_index].median(numeric_only=True).fillna(0.0)
    missing_before = int(frame.isna().sum().sum())
    frame = frame.fillna(train_medians).fillna(0.0)
    if frame.isna().any().any():
        raise RuntimeError("missing values remain after train-only median imputation")

    def x(name: str) -> pd.DataFrame:
        return frame.iloc[partitions[name]]

    def y(name: str) -> np.ndarray:
        return labels[partitions[name]]

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
        x("train"),
        y("train"),
        sample_weight=compute_sample_weight(class_weight="balanced", y=y("train")),
        verbose=False,
    )

    calibration_fit_raw = model.predict_proba(x("calibration_fit"))[:, 1]
    calibration_select_raw = model.predict_proba(x("calibration_select"))[:, 1]
    calibration_choice, calibrator = fit_calibrator(
        calibration_fit_raw,
        y("calibration_fit"),
        calibration_select_raw,
        y("calibration_select"),
    )

    policy_probabilities = calibrate_probabilities(
        model.predict_proba(x("policy"))[:, 1], calibration_choice, calibrator
    )
    high_precision = select_operating_point(
        policy_probabilities,
        y("policy"),
        name="HIGH_PRECISION",
        max_fpr=0.0015,
        objective="recall",
    )
    balanced = select_operating_point(
        policy_probabilities,
        y("policy"),
        name="BALANCED",
        max_fpr=0.01,
        objective="f1",
    )

    test_probabilities = calibrate_probabilities(
        model.predict_proba(x("test"))[:, 1], calibration_choice, calibrator
    )
    profiles = {
        high_precision.name: {
            "operating_point": serialize(high_precision),
            "held_out_test": serialize(
                evaluate_held_out(
                    test_probabilities,
                    y("test"),
                    high_precision,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=42,
                )
            ),
        },
        balanced.name: {
            "operating_point": serialize(balanced),
            "held_out_test": serialize(
                evaluate_held_out(
                    test_probabilities,
                    y("test"),
                    balanced,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=43,
                )
            ),
        },
    }

    if time_column_present:
        evaluation = "chronological_55_train_15_calibration_15_policy_15_untouched_test"
        calibration_selection = "chronological_first_half_fit_second_half_selection"
    else:
        evaluation = "stratified_random_55_train_15_calibration_15_policy_15_untouched_test"
        calibration_selection = "stratified_disjoint_half_fit_half_selection"

    report = {
        "schema_version": "3.0",
        "benchmark_label": "PUBLIC HELD-OUT BENCHMARK",
        "benchmark": "public_fraud_external_validity",
        "source": source_name,
        "source_content_sha256": raw_content_sha256,
        "source_transport": {
            "openml_attempts_configured": args.openml_attempts if args.csv is None else None,
            "openml_retry_delay_seconds": (
                args.openml_retry_delay_seconds if args.csv is None else None
            ),
            "local_csv_fallback_supported": True,
        },
        "domain": "transaction fraud; not Razorpay settlement-hold ground truth",
        "evaluation": evaluation,
        "time_column_requested": args.time_column,
        "time_column_present": time_column_present,
        "chronological_claim": time_column_present,
        "rows": len(frame),
        "features": int(frame.shape[1]),
        "positive_rate": round(float(np.mean(labels)), 8),
        "partition_rows": {name: len(partitions[name]) for name in partitions},
        "partition_positive_rates": {
            name: round(float(np.mean(y(name))), 8) for name in partitions
        },
        "preprocessing": {
            "numeric_features_only": True,
            "train_only_median_imputation": True,
            "missing_values_before_imputation": missing_before,
            "imputation_fit_rows": len(train_index),
        },
        "model": "XGBoost binary hist",
        "calibration": {
            **serialize(calibration_choice),
            "selection_split": calibration_selection,
            "refit_after_selection": "selected method refit on complete calibration partition only",
        },
        "policy": {
            "partition": "policy only",
            "test_used_for_threshold_selection": False,
        },
        "profiles": profiles,
        "confidence_intervals": {
            "method": "class-stratified iid bootstrap",
            "samples": args.bootstrap_samples,
            "limitation": (
                "does not model temporal/account clustering because the public benchmark "
                "lacks such grouping metadata by default"
            ),
        },
        "false_positive_cost": {
            "monetary_cost_claim": False,
            "measure": "false positives and false positives per 10,000 legitimate transactions",
            "reason": "no defensible real monetary unit cost is supplied by this public dataset",
        },
        "safety": {
            "train_only_preprocessing_fit": True,
            "calibration_fit_and_selection_disjoint": True,
            "policy_separate_from_calibration": True,
            "test_used_for_training": False,
            "test_used_for_preprocessing_fit": False,
            "test_used_for_calibration_selection": False,
            "test_used_for_threshold_selection": False,
            "synthetic_metric_claim": False,
            "razorpay_production_metric_claim": False,
            "benchmark_does_not_change_serving_champion": True,
            "champion_remains": "xgb-if-settlement@2",
            "serving_change_authorized": False,
            "automatic_release_enabled": False,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "public_fraud_benchmark_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
