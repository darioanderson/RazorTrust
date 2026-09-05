from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

DATASET_ID = 1597
DATASET_NAME = "ULB_WORLDLINE_CREDITCARD_2013"
SOURCE_KIND = "PUBLIC_REAL_WORLD_ANONYMIZED"
SOURCE_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
REFERENCE_URL = "https://www.openml.org/d/1597"
BENCHMARK_VERSION = "ulb-creditcard-xgb-if@1.3"
FEATURE_NAMES = tuple([f"V{i}" for i in range(1, 29)] + ["Time", "Amount"])
SOURCE_COLUMNS = tuple(
    ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
)
EXPECTED_ROWS = 284_807
EXPECTED_FRAUD_ROWS = 492
ALPHA = 0.10
FN_COST = 20.0
FP_COST = 1.0


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BenchmarkSplit(StrictModel):
    train_rows: int
    calibration_rows: int
    test_rows: int
    split_mode: str


class BenchmarkStatus(StrictModel):
    status: str
    benchmark_version: str = BENCHMARK_VERSION
    dataset_name: str = DATASET_NAME
    source_kind: str = SOURCE_KIND
    source_url: str = SOURCE_URL
    ready: bool
    research_only: bool = True
    production_action_eligible: bool = False
    automatic_release_enabled: bool = False
    human_authorization_required: bool = True
    prepared_at: datetime | None = None
    row_count: int | None = None
    fraud_count: int | None = None
    raw_dataset_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    source_bytes: int | None = None
    source_schema_verified: bool = False
    source_row_count_verified: bool = False
    normalized_dataset_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    test_average_precision: float | None = None
    test_roc_auc: float | None = None
    test_precision: float | None = None
    test_recall: float | None = None
    test_f1: float | None = None
    test_brier_score: float | None = None
    test_true_positives: int | None = None
    test_false_positives: int | None = None
    test_true_negatives: int | None = None
    test_false_negatives: int | None = None
    test_false_positive_rate: float | None = None
    test_false_negative_rate: float | None = None
    test_cost_units: float | None = None
    test_cost_units_per_1000: float | None = None
    threshold: float | None = None

    split: BenchmarkSplit | None = None
    source_reference_url: str | None = None
    source_columns: list[str] | None = None
    expected_rows: int | None = None
    expected_fraud_rows: int | None = None
    calibration_cost_units: float | None = None
    false_negative_cost: float | None = None
    false_positive_cost: float | None = None
    if_reference_split: str | None = None
    if_reference_rows: int | None = None
    held_out_labels_used_for_recommendation: bool | None = None
    feature_names: list[str] | None = None


class BenchmarkCase(StrictModel):
    row_id: int
    elapsed_seconds: float
    amount: float
    source_label: int | None = None
    source_label_name: str | None = None
    label_revealed: bool = False


class BenchmarkStage(StrictModel):
    layer: str
    status: str
    output: dict[str, Any] = {}
    blockers: list[str] = []


class BenchmarkExecution(StrictModel):
    trace_id: str
    benchmark_version: str = BENCHMARK_VERSION
    source_mode: str = SOURCE_KIND
    row_id: int
    stages: list[BenchmarkStage]
    benchmark_recommendation: str
    human_authorization_required: bool = True
    automatic_release_enabled: bool = False
    production_action_eligible: bool = False


@dataclass
class RuntimeBundle:
    model: Any
    calibrator: Any
    scaler: Any
    isolation_forest: Any
    conformal_scores: Any
    anomaly_reference_scores: Any
    test_x: Any
    test_y: Any
    test_row_ids: Any
    threshold: float
    summary: BenchmarkStatus


def clipped_logit(probability: float) -> float:
    value = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def conformal_prediction_set(
    probability_fraud: float,
    calibration_scores: Any,
    *,
    alpha: float = ALPHA,
) -> tuple[list[str], dict[str, float]]:
    import numpy as np

    scores = np.asarray(calibration_scores, dtype=float)
    if scores.ndim != 1 or not scores.size:
        raise ValueError("calibration scores must be a non-empty vector")

    candidate_scores = {
        "LEGITIMATE": float(probability_fraud),
        "FRAUD": float(1.0 - probability_fraud),
    }
    p_values = {
        label: float((1 + np.sum(scores >= score)) / (scores.size + 1))
        for label, score in candidate_scores.items()
    }
    prediction_set = [
        label for label, p_value in p_values.items() if p_value > alpha
    ]
    return prediction_set, p_values


def choose_cost_threshold(
    labels: Any,
    probabilities: Any,
    *,
    false_negative_cost: float = FN_COST,
    false_positive_cost: float = FP_COST,
) -> tuple[float, float]:
    import numpy as np

    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if y.shape != p.shape or not y.size:
        raise ValueError("labels and probabilities must have equal non-zero shape")

    candidates = np.unique(
        np.concatenate(
            (
                np.array([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90]),
                np.quantile(p, np.linspace(0.01, 0.99, 99)),
            )
        )
    )
    best_threshold = 0.50
    best_cost = float("inf")
    for threshold in candidates:
        predicted = (p >= threshold).astype(int)
        false_negatives = int(np.sum((y == 1) & (predicted == 0)))
        false_positives = int(np.sum((y == 0) & (predicted == 1)))
        cost = (
            false_negative_cost * false_negatives
            + false_positive_cost * false_positives
        )
        if cost < best_cost:
            best_cost = float(cost)
            best_threshold = float(threshold)
    return best_threshold, best_cost


class PublicBenchmarkService:
    def __init__(self, root: str | Path | None = None) -> None:
        configured = (
            root
            if root is not None
            else os.getenv(
                "RAZORTRUST_PUBLIC_BENCHMARK_PATH",
                "/app/var/public-benchmark",
            )
        )
        self.root = Path(configured) / "ulb-creditcard-v1"
        self._runtime: RuntimeBundle | None = None

    @property
    def summary_path(self) -> Path:
        return self.root / "benchmark-summary.json"

    def status(self) -> BenchmarkStatus:
        if self._runtime is not None:
            return self._runtime.summary
        if not self.summary_path.is_file():
            return BenchmarkStatus(status="NOT_PREPARED", ready=False)
        try:
            summary = BenchmarkStatus.model_validate_json(
                self.summary_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return BenchmarkStatus(status="ARTIFACT_INVALID", ready=False)

        reference_path = self.root / "if-calibration-legitimate-scores.npy"
        if (
            summary.benchmark_version != BENCHMARK_VERSION
            or not reference_path.is_file()
        ):
            return BenchmarkStatus(
                status="REPREPARE_REQUIRED",
                ready=False,
                raw_dataset_sha256=summary.raw_dataset_sha256,
                source_bytes=summary.source_bytes,
                source_schema_verified=summary.source_schema_verified,
                source_row_count_verified=summary.source_row_count_verified,
                normalized_dataset_sha256=summary.normalized_dataset_sha256,
            )
        return summary

    def prepare(self, force: bool = False) -> BenchmarkStatus:
        if self._runtime is not None and not force:
            return self._runtime.summary

        self.root.mkdir(parents=True, exist_ok=True)
        if not force and self._load_runtime():
            assert self._runtime is not None
            return self._runtime.summary

        import joblib
        import numpy as np
        import pandas as pd
        import xgboost as xgb
        from sklearn.ensemble import IsolationForest
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            average_precision_score,
            brier_score_loss,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.preprocessing import StandardScaler

        source_path, raw_hash, source_bytes = self._source_file(
            force=force,
        )
        source_frame = pd.read_csv(source_path)

        actual_columns = tuple(str(name) for name in source_frame.columns)
        if actual_columns != SOURCE_COLUMNS:
            raise RuntimeError(
                "ULB source schema mismatch; expected exact 31-column "
                f"schema, got {list(actual_columns)}"
            )

        if int(source_frame.shape[0]) != EXPECTED_ROWS:
            raise RuntimeError(
                "ULB source row-count mismatch; "
                f"expected {EXPECTED_ROWS}, got {source_frame.shape[0]}"
            )

        labels = pd.to_numeric(
            source_frame["Class"],
            errors="raise",
        ).astype(int)
        label_values = set(labels.unique().tolist())
        if not label_values.issubset({0, 1}):
            raise RuntimeError(
                f"ULB source has unexpected Class values: {sorted(label_values)}"
            )

        fraud_rows = int(labels.sum())
        if fraud_rows != EXPECTED_FRAUD_ROWS:
            raise RuntimeError(
                "ULB source fraud-count mismatch; "
                f"expected {EXPECTED_FRAUD_ROWS}, got {fraud_rows}"
            )

        frame = source_frame.loc[:, list(FEATURE_NAMES)].copy()
        for name in FEATURE_NAMES:
            frame[name] = pd.to_numeric(frame[name], errors="raise")

        values = frame.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError("ULB source contains non-finite model inputs")

        if float(frame["Time"].min()) < 0:
            raise RuntimeError("ULB source contains negative Time values")

        normalized = frame.copy()
        normalized["Class"] = labels.to_numpy(dtype=int)
        normalized_hash = hashlib.sha256(
            pd.util.hash_pandas_object(
                normalized,
                index=True,
            ).values.tobytes()
        ).hexdigest()

        row_ids = np.arange(frame.shape[0], dtype=np.int64)
        order = np.argsort(frame["Time"].to_numpy(dtype=float), kind="stable")
        x_all = frame.to_numpy(dtype=np.float32)[order]
        y_all = labels.to_numpy(dtype=np.int8)[order]
        row_ids = row_ids[order]

        n_rows = x_all.shape[0]
        train_end = int(n_rows * 0.60)
        calibration_end = int(n_rows * 0.80)

        x_train = x_all[:train_end]
        y_train = y_all[:train_end]
        x_cal = x_all[train_end:calibration_end]
        y_cal = y_all[train_end:calibration_end]
        x_test = x_all[calibration_end:]
        y_test = y_all[calibration_end:]
        test_row_ids = row_ids[calibration_end:]

        positive = max(int(np.sum(y_train == 1)), 1)
        negative = max(int(np.sum(y_train == 0)), 1)
        scale_pos_weight = float(negative / positive)

        model = xgb.XGBClassifier(
            n_estimators=180,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2.0,
            reg_lambda=5.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=2,
            scale_pos_weight=scale_pos_weight,
        )
        model.fit(x_train, y_train)

        raw_cal = model.predict_proba(x_cal)[:, 1]
        calibration_input = np.array(
            [clipped_logit(item) for item in raw_cal],
            dtype=np.float64,
        ).reshape(-1, 1)
        calibrator = LogisticRegression(
            solver="lbfgs",
            random_state=42,
        )
        calibrator.fit(calibration_input, y_cal)

        calibrated_cal = calibrator.predict_proba(calibration_input)[:, 1]
        threshold, calibration_cost = choose_cost_threshold(
            y_cal,
            calibrated_cal,
        )
        conformal_scores = np.where(
            y_cal == 1,
            1.0 - calibrated_cal,
            calibrated_cal,
        ).astype(np.float32)

        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(x_train)
        legitimate_indices = np.flatnonzero(y_train == 0)
        rng = np.random.default_rng(42)
        if legitimate_indices.size > 40_000:
            legitimate_indices = rng.choice(
                legitimate_indices,
                size=40_000,
                replace=False,
            )
        isolation_forest = IsolationForest(
            n_estimators=160,
            contamination="auto",
            random_state=42,
            n_jobs=2,
        )
        isolation_forest.fit(scaled_train[legitimate_indices])

        calibration_legitimate = x_cal[y_cal == 0]
        if not calibration_legitimate.shape[0]:
            raise RuntimeError(
                "calibration split has no legitimate rows for IF reference"
            )
        anomaly_reference_scores = -isolation_forest.score_samples(
            scaler.transform(calibration_legitimate)
        ).astype(np.float32)

        raw_test = model.predict_proba(x_test)[:, 1]
        test_input = np.array(
            [clipped_logit(item) for item in raw_test],
            dtype=np.float64,
        ).reshape(-1, 1)
        calibrated_test = calibrator.predict_proba(test_input)[:, 1]
        predicted_test = (calibrated_test >= threshold).astype(int)

        test_true_positives = int(
            np.sum((y_test == 1) & (predicted_test == 1))
        )
        test_false_positives = int(
            np.sum((y_test == 0) & (predicted_test == 1))
        )
        test_true_negatives = int(
            np.sum((y_test == 0) & (predicted_test == 0))
        )
        test_false_negatives = int(
            np.sum((y_test == 1) & (predicted_test == 0))
        )
        legitimate_test_rows = test_true_negatives + test_false_positives
        fraud_test_rows = test_true_positives + test_false_negatives
        test_false_positive_rate = (
            test_false_positives / legitimate_test_rows
            if legitimate_test_rows
            else 0.0
        )
        test_false_negative_rate = (
            test_false_negatives / fraud_test_rows
            if fraud_test_rows
            else 0.0
        )
        test_cost_units = float(
            FN_COST * test_false_negatives
            + FP_COST * test_false_positives
        )
        test_cost_units_per_1000 = float(
            1000.0 * test_cost_units / y_test.shape[0]
        )

        summary = BenchmarkStatus(
            status="READY",
            ready=True,
            prepared_at=datetime.now(UTC),
            row_count=int(n_rows),
            fraud_count=int(np.sum(y_all == 1)),
            raw_dataset_sha256=raw_hash,
            source_bytes=source_bytes,
            source_schema_verified=True,
            source_row_count_verified=True,
            normalized_dataset_sha256=normalized_hash,
            test_average_precision=float(
                average_precision_score(y_test, calibrated_test)
            ),
            test_roc_auc=float(roc_auc_score(y_test, calibrated_test)),
            test_precision=float(
                precision_score(y_test, predicted_test, zero_division=0)
            ),
            test_recall=float(
                recall_score(y_test, predicted_test, zero_division=0)
            ),
            test_f1=float(f1_score(y_test, predicted_test, zero_division=0)),
            test_brier_score=float(
                brier_score_loss(y_test, calibrated_test)
            ),
            test_true_positives=test_true_positives,
            test_false_positives=test_false_positives,
            test_true_negatives=test_true_negatives,
            test_false_negatives=test_false_negatives,
            test_false_positive_rate=float(test_false_positive_rate),
            test_false_negative_rate=float(test_false_negative_rate),
            test_cost_units=test_cost_units,
            test_cost_units_per_1000=test_cost_units_per_1000,
            threshold=threshold,
            split=BenchmarkSplit(
                train_rows=int(x_train.shape[0]),
                calibration_rows=int(x_cal.shape[0]),
                test_rows=int(x_test.shape[0]),
                split_mode="chronological_60_20_20",
            ),
            source_reference_url=REFERENCE_URL,
            source_columns=list(SOURCE_COLUMNS),
            expected_rows=EXPECTED_ROWS,
            expected_fraud_rows=EXPECTED_FRAUD_ROWS,
            calibration_cost_units=calibration_cost,
            false_negative_cost=FN_COST,
            false_positive_cost=FP_COST,
            if_reference_split="CALIBRATION_LEGITIMATE_ONLY",
            if_reference_rows=int(
                anomaly_reference_scores.shape[0]
            ),
            held_out_labels_used_for_recommendation=False,
            feature_names=list(FEATURE_NAMES),
        )

        model.save_model(self.root / "xgb-model.json")
        joblib.dump(calibrator, self.root / "calibrator.joblib")
        joblib.dump(scaler, self.root / "scaler.joblib")
        joblib.dump(isolation_forest, self.root / "isolation-forest.joblib")
        np.save(self.root / "conformal-scores.npy", conformal_scores)
        np.save(
            self.root / "if-calibration-legitimate-scores.npy",
            anomaly_reference_scores,
        )
        np.savez_compressed(
            self.root / "test-split.npz",
            x=x_test,
            y=y_test,
            row_ids=test_row_ids,
        )
        (self.root / "feature-names.json").write_text(
            json.dumps(list(FEATURE_NAMES), indent=2) + "\n",
            encoding="utf-8",
        )
        self.summary_path.write_text(
            json.dumps(
                {
                    **summary.model_dump(mode="json"),
                    "split": {
                        "train_rows": int(x_train.shape[0]),
                        "calibration_rows": int(x_cal.shape[0]),
                        "test_rows": int(x_test.shape[0]),
                        "split_mode": "chronological_60_20_20",
                    },
                    "source_reference_url": REFERENCE_URL,
                    "source_columns": list(SOURCE_COLUMNS),
                    "expected_rows": EXPECTED_ROWS,
                    "expected_fraud_rows": EXPECTED_FRAUD_ROWS,
                    "calibration_cost_units": calibration_cost,
                    "false_negative_cost": FN_COST,
                    "false_positive_cost": FP_COST,
                    "if_reference_split": "CALIBRATION_LEGITIMATE_ONLY",
                    "if_reference_rows": int(
                        anomaly_reference_scores.shape[0]
                    ),
                    "held_out_labels_used_for_recommendation": False,
                    "feature_names": list(FEATURE_NAMES),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        self._runtime = RuntimeBundle(
            model=model,
            calibrator=calibrator,
            scaler=scaler,
            isolation_forest=isolation_forest,
            conformal_scores=conformal_scores,
            anomaly_reference_scores=anomaly_reference_scores,
            test_x=x_test,
            test_y=y_test,
            test_row_ids=test_row_ids,
            threshold=threshold,
            summary=summary,
        )
        return summary

    def list_cases(
        self,
        *,
        label: str = "blind",
        limit: int = 20,
    ) -> list[BenchmarkCase]:
        runtime = self._require_runtime()
        import numpy as np

        reveal_labels = label != "blind"
        if label == "blind":
            sample_size = min(limit, runtime.test_y.shape[0])
            if sample_size == 1:
                positions = np.array([runtime.test_y.shape[0] // 2])
            else:
                positions = np.linspace(
                    0,
                    runtime.test_y.shape[0] - 1,
                    num=sample_size,
                    dtype=int,
                )
        elif label == "fraud":
            positions = np.flatnonzero(runtime.test_y == 1)[:limit]
        elif label == "legitimate":
            positions = np.flatnonzero(runtime.test_y == 0)[:limit]
        elif label == "all":
            positions = np.arange(runtime.test_y.shape[0])[:limit]
        else:
            raise ValueError(
                "label must be blind, fraud, legitimate, or all"
            )

        items = []
        time_index = FEATURE_NAMES.index("Time")
        amount_index = FEATURE_NAMES.index("Amount")
        for position in positions:
            label_value = int(runtime.test_y[position])
            items.append(
                BenchmarkCase(
                    row_id=int(runtime.test_row_ids[position]),
                    elapsed_seconds=float(
                        runtime.test_x[position, time_index]
                    ),
                    amount=float(runtime.test_x[position, amount_index]),
                    source_label=label_value if reveal_labels else None,
                    source_label_name=(
                        "FRAUD" if label_value == 1 else "LEGITIMATE"
                    )
                    if reveal_labels
                    else None,
                    label_revealed=reveal_labels,
                )
            )
        return items

    def execute(self, row_id: int) -> BenchmarkExecution:
        runtime = self._require_runtime()
        import numpy as np
        import xgboost as xgb

        matches = np.flatnonzero(runtime.test_row_ids == row_id)
        if not matches.size:
            raise LookupError("row_id is not in the held-out benchmark test split")
        position = int(matches[0])
        row = runtime.test_x[position : position + 1]

        raw_probability = float(runtime.model.predict_proba(row)[0, 1])
        calibrated_probability = float(
            runtime.calibrator.predict_proba(
                [[clipped_logit(raw_probability)]]
            )[0, 1]
        )
        legitimate_probability = float(1.0 - calibrated_probability)

        scaled_row = runtime.scaler.transform(row)
        anomaly_raw = float(
            -runtime.isolation_forest.score_samples(scaled_row)[0]
        )

        reference_scores = runtime.anomaly_reference_scores
        anomaly_percentile = float(
            (np.sum(reference_scores <= anomaly_raw) + 1)
            / (reference_scores.size + 1)
        )

        booster = runtime.model.get_booster()
        contributions = booster.predict(
            xgb.DMatrix(row, feature_names=list(FEATURE_NAMES)),
            pred_contribs=True,
        )[0]
        ranked = sorted(
            zip(FEATURE_NAMES, contributions[:-1], strict=True),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )[:7]
        top_features = [
            {
                "feature": name,
                "value": float(row[0, FEATURE_NAMES.index(name)]),
                "shap_log_odds_contribution": float(contribution),
                "direction": (
                    "toward_fraud" if contribution > 0 else "toward_legitimate"
                ),
            }
            for name, contribution in ranked
        ]

        prediction_set, p_values = conformal_prediction_set(
            calibrated_probability,
            runtime.conformal_scores,
        )
        uncertain = len(prediction_set) != 1

        if calibrated_probability >= runtime.threshold:
            recommendation = "ESCALATE"
        elif uncertain or anomaly_percentile >= 0.95:
            recommendation = "EVIDENCE_NEEDED"
        else:
            recommendation = "LOW_RISK_REVIEW"

        # Evaluation ground truth is deliberately inaccessible to the
        # scoring/recommendation path. Read the held-out label only after
        # XGBoost, calibration, Isolation Forest, SHAP, conformal
        # uncertainty, and the benchmark recommendation are complete.
        source_label = int(runtime.test_y[position])

        trace_id = str(uuid4())
        summary = runtime.summary
        stages = [
            BenchmarkStage(
                layer="SOURCE_PROVENANCE",
                status="RESULT",
                output={
                    "dataset_id": DATASET_ID,
                    "dataset_name": DATASET_NAME,
                    "source_kind": SOURCE_KIND,
                    "source_url": SOURCE_URL,
                    "source_reference_url": REFERENCE_URL,
                    "row_id": row_id,
                    "raw_dataset_sha256": summary.raw_dataset_sha256,
                    "source_bytes": summary.source_bytes,
                    "source_schema_verified": summary.source_schema_verified,
                    "source_row_count_verified": (
                        summary.source_row_count_verified
                    ),
                    "normalized_dataset_sha256": (
                        summary.normalized_dataset_sha256
                    ),
                    "real_world_public_benchmark": True,
                    "razorpay_data": False,
                    "settlement_hold_ground_truth": False,
                },
            ),
            BenchmarkStage(
                layer="DATASET_INTEGRITY",
                status="RESULT",
                output={
                    "rows": summary.row_count,
                    "fraud_rows": summary.fraud_count,
                    "locked_benchmark_features": list(FEATURE_NAMES),
                    "label_excluded_from_model_inputs": True,
                },
            ),
            BenchmarkStage(
                layer="CHRONOLOGICAL_SPLIT",
                status="RESULT",
                output={
                    "mode": "chronological_60_20_20",
                    "selected_case_partition": "HELD_OUT_TEST",
                    "future_rows_used_for_training": False,
                },
            ),
            BenchmarkStage(
                layer="BENCHMARK_XGBOOST",
                status="SCORED",
                output={
                    "model_version": BENCHMARK_VERSION,
                    "raw_probability_fraud": raw_probability,
                    "production_model": False,
                    "frozen_settlement_model_modified": False,
                },
            ),
            BenchmarkStage(
                layer="PROBABILITY_CALIBRATION",
                status="RESULT",
                output={
                    "method": (
                        "logistic_sigmoid_on_chronological_calibration_split"
                    ),
                    "probabilities": {
                        "legitimate": legitimate_probability,
                        "fraud": calibrated_probability,
                    },
                    "decision_threshold": runtime.threshold,
                },
            ),
            BenchmarkStage(
                layer="ISOLATION_FOREST",
                status="RESULT",
                output={
                    "anomaly_raw_score": anomaly_raw,
                    "anomaly_percentile_vs_legitimate_reference": (
                        anomaly_percentile
                    ),
                    "reference_mode": "calibration_legitimate_only",
                    "research_only": True,
                },
            ),
            BenchmarkStage(
                layer="SHAP",
                status="RESULT",
                output={
                    "method": "xgboost_native_tree_shap",
                    "top_features": top_features,
                    "label_used_for_explanation": False,
                },
            ),
            BenchmarkStage(
                layer="CONFORMAL_UNCERTAINTY",
                status="RESULT",
                output={
                    "alpha": ALPHA,
                    "prediction_set": prediction_set,
                    "p_values": p_values,
                    "uncertain": uncertain,
                    "calibration_mode": "split_conformal_chronological",
                },
            ),
            BenchmarkStage(
                layer="BENCHMARK_RECOMMENDATION",
                status="RESULT",
                output={
                    "recommendation": recommendation,
                    "fraud_probability": calibrated_probability,
                    "anomaly_percentile": anomaly_percentile,
                    "research_only": True,
                    "production_action_eligible": False,
                },
            ),
            BenchmarkStage(
                layer="OPA",
                status="BLOCKED",
                output={
                    "reason": (
                        "public benchmark recommendation is not a production "
                        "settlement-policy candidate"
                    ),
                    "policy_input_submitted": False,
                },
            ),
            BenchmarkStage(
                layer="SOURCE_GROUND_TRUTH",
                status="RESULT",
                output={
                    "source_label": source_label,
                    "source_label_name": (
                        "FRAUD" if source_label == 1 else "LEGITIMATE"
                    ),
                    "evaluation_only": True,
                    "revealed_after_scoring": True,
                    "truth_access_mode": "POST_RECOMMENDATION_ONLY",
                    "used_for_model_scoring": False,
                    "used_for_calibration_of_this_case": False,
                    "held_out_labels_used_for_recommendation": False,
                    "prediction_matches_source_label": (
                        (calibrated_probability >= runtime.threshold)
                        == bool(source_label)
                    ),
                },
            ),
            BenchmarkStage(
                layer="HUMAN_ONLY",
                status="WAITING_FOR_HUMAN",
                output={
                    "benchmark_recommendation": recommendation,
                    "enforcement_mode": "human_only",
                    "human_authorization_required": True,
                    "automatic_release_enabled": False,
                    "production_action_eligible": False,
                },
            ),
        ]

        return BenchmarkExecution(
            trace_id=trace_id,
            row_id=row_id,
            stages=stages,
            benchmark_recommendation=recommendation,
        )

    def _source_file(self, *, force: bool) -> tuple[Path, str, int]:
        import urllib.request

        source_dir = self.root / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        source_path = source_dir / "creditcard.csv"

        if force or not source_path.is_file():
            partial_path = source_dir / "creditcard.csv.part"
            if partial_path.exists():
                partial_path.unlink()

            request = urllib.request.Request(
                SOURCE_URL,
                headers={"User-Agent": "RazorTrust-Buildathon/1.0"},
            )
            digest = hashlib.sha256()
            total_bytes = 0

            try:
                with urllib.request.urlopen(
                    request,
                    timeout=120,
                ) as response:
                    if int(response.status) != 200:
                        raise RuntimeError(
                            f"ULB source returned HTTP {response.status}"
                        )
                    with partial_path.open("wb") as handle:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            handle.write(chunk)
                            digest.update(chunk)
                            total_bytes += len(chunk)
                partial_path.replace(source_path)
            except Exception:
                if partial_path.exists():
                    partial_path.unlink()
                raise

            return source_path, digest.hexdigest(), total_bytes

        digest = hashlib.sha256()
        total_bytes = 0
        with source_path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                total_bytes += len(chunk)
        return source_path, digest.hexdigest(), total_bytes

    def _load_runtime(self) -> bool:
        required = (
            self.summary_path,
            self.root / "xgb-model.json",
            self.root / "calibrator.joblib",
            self.root / "scaler.joblib",
            self.root / "isolation-forest.joblib",
            self.root / "conformal-scores.npy",
            self.root / "if-calibration-legitimate-scores.npy",
            self.root / "test-split.npz",
        )
        if not all(path.is_file() for path in required):
            return False

        import joblib
        import numpy as np
        import xgboost as xgb

        summary = BenchmarkStatus.model_validate_json(
            self.summary_path.read_text(encoding="utf-8")
        )
        if (
            not summary.ready
            or summary.benchmark_version != BENCHMARK_VERSION
            or summary.threshold is None
        ):
            return False

        model = xgb.XGBClassifier()
        model.load_model(self.root / "xgb-model.json")
        test = np.load(self.root / "test-split.npz")
        self._runtime = RuntimeBundle(
            model=model,
            calibrator=joblib.load(self.root / "calibrator.joblib"),
            scaler=joblib.load(self.root / "scaler.joblib"),
            isolation_forest=joblib.load(
                self.root / "isolation-forest.joblib"
            ),
            conformal_scores=np.load(self.root / "conformal-scores.npy"),
            anomaly_reference_scores=np.load(
                self.root / "if-calibration-legitimate-scores.npy"
            ),
            test_x=test["x"],
            test_y=test["y"],
            test_row_ids=test["row_ids"],
            threshold=float(summary.threshold),
            summary=summary,
        )
        return True

    def _require_runtime(self) -> RuntimeBundle:
        if self._runtime is None and not self._load_runtime():
            raise LookupError(
                "public benchmark is not prepared; run the prepare endpoint first"
            )
        assert self._runtime is not None
        return self._runtime
