from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import Field
from sklearn.metrics import average_precision_score, log_loss, precision_recall_fscore_support

from ..audit import canonical_json
from ..costs import DEFAULT_COST_MATRIX
from ..domain import HoldDecision, StrictModel
from .dataset import model_matrix
from .modeling import (
    LABEL_TO_INDEX,
    ModelBundle,
    _multiclass_brier,
    _normalize_probabilities,
    _target_indices,
    _threshold_actions,
    _with_anomaly,
)
from .splits import SplitManifest


class ClassMetric(StrictModel):
    label: HoldDecision
    precision: float
    recall: float
    f1: float
    pr_auc: float
    support: int


class MetricInterval(StrictModel):
    metric: str
    estimate: float
    lower_95: float
    upper_95: float
    resamples: int


class BaselineMetric(StrictModel):
    system: str
    expected_cost_units: float
    false_release_rate: float
    escalation_recall: float


class CostSensitivityMetric(StrictModel):
    false_release_cost_multiplier: float
    release_threshold: float
    escalate_threshold: float
    evidence_rate: float
    expected_cost_units: float


class EvaluationReport(StrictModel):
    schema_version: str = "1.0"
    model_version: str
    split_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    test_merchants_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: int
    log_loss: float
    multiclass_brier: float
    top_label_ece: float
    expected_cost_units: float
    false_release_rate: float
    classes: list[ClassMetric]
    content_sha256: str
    cost_matrix_version: str
    cost_matrix_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    confidence_intervals: list[MetricInterval]
    baselines: list[BaselineMetric]
    cost_sensitivity: list[CostSensitivityMetric]


def evaluate_sealed_test(
    frame: pd.DataFrame,
    bundle: ModelBundle,
    manifest: SplitManifest,
    *,
    bootstrap_iterations: int = 1_000,
    seed: int = 42,
) -> EvaluationReport:
    test = frame[frame["merchant_id"].isin(manifest.test_merchants)].reset_index(drop=True)
    if test.empty:
        raise ValueError("sealed test partition is empty")
    labels = _target_indices(test)
    probabilities = bundle.predict_proba(model_matrix(test), test["cohort"])
    predictions = _threshold_actions(
        probabilities, bundle.thresholds.release, bundle.thresholds.escalate
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1, 2], zero_division=0
    )
    one_hot = np.eye(3)[labels]
    class_metrics = [
        ClassMetric(
            label=HoldDecision(label),
            precision=round(float(precision[index]), 8),
            recall=round(float(recall[index]), 8),
            f1=round(float(f1[index]), 8),
            pr_auc=round(
                float(average_precision_score(one_hot[:, index], probabilities[:, index])), 8
            ),
            support=int(support[index]),
        )
        for index, label in enumerate(("RELEASE", "EVIDENCE_NEEDED", "ESCALATE"))
    ]
    non_release = labels != LABEL_TO_INDEX[HoldDecision.RELEASE]
    false_release = float(np.mean(predictions[non_release] == LABEL_TO_INDEX[HoldDecision.RELEASE]))
    matrix = np.asarray(DEFAULT_COST_MATRIX.matrix)
    expected_cost = float(np.mean(matrix[predictions, labels]))
    escalation_recall = _class_recall(labels, predictions, LABEL_TO_INDEX[HoldDecision.ESCALATE])
    confidence_intervals = _merchant_bootstrap_intervals(
        test,
        labels,
        predictions,
        expected_cost,
        false_release,
        escalation_recall,
        iterations=bootstrap_iterations,
        seed=seed,
    )
    baselines = _baseline_metrics(test, bundle, labels, probabilities, predictions)
    cost_sensitivity = _cost_sensitivity(probabilities, labels, bundle)
    content = {
        "schema_version": "1.0",
        "model_version": bundle.model_version,
        "split_manifest_sha256": manifest.content_sha256,
        "test_merchants_sha256": manifest.test_merchants_sha256,
        "row_count": len(test),
        "log_loss": round(float(log_loss(labels, probabilities, labels=[0, 1, 2])), 8),
        "multiclass_brier": round(_multiclass_brier(labels, probabilities), 8),
        "top_label_ece": round(_top_label_ece(labels, probabilities), 8),
        "expected_cost_units": round(expected_cost, 8),
        "false_release_rate": round(false_release, 8),
        "classes": [metric.model_dump(mode="json") for metric in class_metrics],
        "cost_matrix_version": DEFAULT_COST_MATRIX.cost_matrix_version,
        "cost_matrix_sha256": DEFAULT_COST_MATRIX.content_sha256,
        "confidence_intervals": [item.model_dump(mode="json") for item in confidence_intervals],
        "baselines": [item.model_dump(mode="json") for item in baselines],
        "cost_sensitivity": [item.model_dump(mode="json") for item in cost_sensitivity],
    }
    return EvaluationReport.model_validate(
        {
            **content,
            "classes": class_metrics,
            "confidence_intervals": confidence_intervals,
            "baselines": baselines,
            "cost_sensitivity": cost_sensitivity,
            "content_sha256": hashlib.sha256(canonical_json(content)).hexdigest(),
        }
    )


def write_evaluation_report(path: str | Path, report: EvaluationReport) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = (report.model_dump_json(indent=2) + "\n").encode()
    if destination.exists() and destination.read_bytes() != content:
        raise FileExistsError(f"refusing to replace sealed evaluation: {destination}")
    if not destination.exists():
        destination.write_bytes(content)


def _top_label_ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    ece = 0.0
    for lower in np.linspace(0, 1, bins, endpoint=False):
        upper = lower + 1 / bins
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean() - confidence[mask].mean()))
    return ece


def _class_recall(labels: np.ndarray, predictions: np.ndarray, target: int) -> float:
    mask = labels == target
    return float(np.mean(predictions[mask] == target)) if mask.any() else 0.0


def _merchant_bootstrap_intervals(
    test: pd.DataFrame,
    labels: np.ndarray,
    predictions: np.ndarray,
    expected_cost: float,
    false_release: float,
    escalation_recall: float,
    *,
    iterations: int,
    seed: int,
) -> list[MetricInterval]:
    merchants = np.asarray(sorted(test["merchant_id"].astype(str).unique()))
    indices_by_merchant = {
        merchant: np.flatnonzero(test["merchant_id"].astype(str).to_numpy() == merchant)
        for merchant in merchants
    }
    rng = np.random.default_rng(seed)
    matrix = np.asarray(DEFAULT_COST_MATRIX.matrix)
    samples: dict[str, list[float]] = {
        "expected_cost_units": [],
        "false_release_rate": [],
        "escalation_recall": [],
    }
    for _ in range(iterations):
        selected = rng.choice(merchants, size=len(merchants), replace=True)
        index = np.concatenate([indices_by_merchant[merchant] for merchant in selected])
        sampled_labels = labels[index]
        sampled_predictions = predictions[index]
        samples["expected_cost_units"].append(
            float(np.mean(matrix[sampled_predictions, sampled_labels]))
        )
        non_release = sampled_labels != LABEL_TO_INDEX[HoldDecision.RELEASE]
        samples["false_release_rate"].append(
            float(np.mean(sampled_predictions[non_release] == 0)) if non_release.any() else 0.0
        )
        samples["escalation_recall"].append(_class_recall(sampled_labels, sampled_predictions, 2))
    estimates = {
        "expected_cost_units": expected_cost,
        "false_release_rate": false_release,
        "escalation_recall": escalation_recall,
    }
    return [
        MetricInterval(
            metric=name,
            estimate=round(estimates[name], 8),
            lower_95=round(float(np.quantile(values, 0.025)), 8),
            upper_95=round(float(np.quantile(values, 0.975)), 8),
            resamples=iterations,
        )
        for name, values in samples.items()
    ]


def _baseline_metrics(
    test: pd.DataFrame,
    bundle: ModelBundle,
    labels: np.ndarray,
    calibrated_probabilities: np.ndarray,
    full_predictions: np.ndarray,
) -> list[BaselineMetric]:
    features = model_matrix(test)
    with_anomaly = _with_anomaly(features, test["cohort"], bundle.anomaly_scorer)
    zero_novelty = with_anomaly.copy()
    zero_novelty.iloc[:, -1] = 0.0
    rules = np.full(len(test), 1)
    rules[(features["chargeback_rate_delta_z"] > 2) | (features["failed_auth_ratio"] > 0.3)] = 2
    rules[
        (features["chargeback_rate_delta_z"] < 0.5)
        & (features["failed_auth_ratio"] < 0.05)
        & (features["volume_delta_z"] < 2)
    ] = 0
    systems = {
        "always_escalate": np.full(len(test), 2),
        "development_rules": rules,
        "xgb_zero_novelty": np.argmax(
            _normalize_probabilities(bundle.classifier.predict_proba(zero_novelty)), axis=1
        ),
        "xgb_if_uncalibrated": np.argmax(
            _normalize_probabilities(bundle.classifier.predict_proba(with_anomaly)), axis=1
        ),
        "xgb_if_calibrated_argmax": np.argmax(calibrated_probabilities, axis=1),
        "full_tier0_threshold_policy": full_predictions,
    }
    matrix = np.asarray(DEFAULT_COST_MATRIX.matrix)
    results: list[BaselineMetric] = []
    for name, actions in systems.items():
        non_release = labels != 0
        results.append(
            BaselineMetric(
                system=name,
                expected_cost_units=round(float(np.mean(matrix[actions, labels])), 8),
                false_release_rate=round(
                    float(np.mean(actions[non_release] == 0)) if non_release.any() else 0.0, 8
                ),
                escalation_recall=round(_class_recall(labels, actions, 2), 8),
            )
        )
    return results


def _cost_sensitivity(
    probabilities: np.ndarray, labels: np.ndarray, bundle: ModelBundle
) -> list[CostSensitivityMetric]:
    results: list[CostSensitivityMetric] = []
    for multiplier in (0.5, 1.0, 2.0):
        matrix = [row.copy() for row in DEFAULT_COST_MATRIX.matrix]
        matrix[0][1] *= multiplier
        matrix[0][2] *= multiplier
        actions = _threshold_actions(
            probabilities, bundle.thresholds.release, bundle.thresholds.escalate
        )
        results.append(
            CostSensitivityMetric(
                false_release_cost_multiplier=multiplier,
                release_threshold=bundle.thresholds.release,
                escalate_threshold=bundle.thresholds.escalate,
                evidence_rate=round(float(np.mean(actions == 1)), 8),
                expected_cost_units=round(float(np.mean(np.asarray(matrix)[actions, labels])), 8),
            )
        )
    return results
