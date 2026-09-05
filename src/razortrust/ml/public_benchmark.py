from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

CalibrationMethod = Literal["uncalibrated", "sigmoid", "isotonic"]


@dataclass(frozen=True)
class CalibrationChoice:
    method: CalibrationMethod
    log_loss: float
    brier: float
    fit_rows: int
    selection_rows: int
    selection_protocol: str = "disjoint_fit_selection"


@dataclass(frozen=True)
class OperatingPoint:
    name: str
    threshold: float
    max_fpr: float
    policy_precision: float
    policy_recall: float
    policy_f1: float
    policy_fpr: float


@dataclass(frozen=True)
class HeldOutMetrics:
    pr_auc: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    brier: float
    log_loss: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    false_positives_per_10000_legitimate: float
    false_negatives_per_100_fraud: float
    predicted_positive_rate: float
    pr_auc_ci95: tuple[float, float]
    precision_ci95: tuple[float, float]
    recall_ci95: tuple[float, float]


def fit_calibrator(
    fit_probabilities: np.ndarray,
    fit_labels: np.ndarray,
    selection_probabilities: np.ndarray,
    selection_labels: np.ndarray,
) -> tuple[CalibrationChoice, object | None]:
    """Select on disjoint data, then refit the chosen method on all calibration rows."""
    fit_probabilities = _clip(fit_probabilities)
    selection_probabilities = _clip(selection_probabilities)
    _require_binary(fit_labels, "calibration-fit")
    _require_binary(selection_labels, "calibration-selection")

    candidates: list[tuple[CalibrationChoice, CalibrationMethod]] = []
    candidates.append(
        (
            CalibrationChoice(
                method="uncalibrated",
                log_loss=round(float(log_loss(selection_labels, selection_probabilities)), 8),
                brier=round(float(brier_score_loss(selection_labels, selection_probabilities)), 8),
                fit_rows=len(fit_labels),
                selection_rows=len(selection_labels),
            ),
            "uncalibrated",
        )
    )

    sigmoid = LogisticRegression(max_iter=1000, random_state=42)
    sigmoid.fit(_logit(fit_probabilities), fit_labels)
    sigmoid_probs = sigmoid.predict_proba(_logit(selection_probabilities))[:, 1]
    candidates.append(
        (
            CalibrationChoice(
                method="sigmoid",
                log_loss=round(float(log_loss(selection_labels, sigmoid_probs)), 8),
                brier=round(float(brier_score_loss(selection_labels, sigmoid_probs)), 8),
                fit_rows=len(fit_labels),
                selection_rows=len(selection_labels),
            ),
            "sigmoid",
        )
    )

    if len(fit_labels) >= 1000 and np.unique(fit_probabilities).size >= 20:
        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(fit_probabilities, fit_labels)
        isotonic_probs = _clip(isotonic.predict(selection_probabilities))
        candidates.append(
            (
                CalibrationChoice(
                    method="isotonic",
                    log_loss=round(float(log_loss(selection_labels, isotonic_probs)), 8),
                    brier=round(float(brier_score_loss(selection_labels, isotonic_probs)), 8),
                    fit_rows=len(fit_labels),
                    selection_rows=len(selection_labels),
                ),
                "isotonic",
            )
        )

    candidates.sort(key=lambda item: (item[0].log_loss, item[0].brier, item[0].method))
    choice, method = candidates[0]

    all_probabilities = np.concatenate([fit_probabilities, selection_probabilities])
    all_labels = np.concatenate([fit_labels, selection_labels])
    calibrator = _fit_method(method, all_probabilities, all_labels)
    return choice, calibrator


def calibrate_probabilities(
    probabilities: np.ndarray,
    choice: CalibrationChoice,
    calibrator: object | None,
) -> np.ndarray:
    probabilities = _clip(probabilities)
    if choice.method == "uncalibrated" or calibrator is None:
        return probabilities
    if choice.method == "sigmoid":
        assert isinstance(calibrator, LogisticRegression)
        return np.asarray(calibrator.predict_proba(_logit(probabilities))[:, 1], dtype=float)
    if choice.method == "isotonic":
        assert isinstance(calibrator, IsotonicRegression)
        return np.asarray(calibrator.predict(probabilities), dtype=float)
    raise ValueError(f"unsupported calibration method: {choice.method}")


def select_operating_point(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    name: str,
    max_fpr: float,
    objective: Literal["recall", "f1"] = "recall",
) -> OperatingPoint:
    _require_binary(labels, "policy")
    negatives = labels == 0
    best: tuple[float, float, float, OperatingPoint] | None = None
    for threshold in np.linspace(0.001, 0.999, 999):
        predicted = probabilities >= threshold
        fpr = float(np.mean(predicted[negatives]))
        if fpr > max_fpr:
            continue
        precision = float(precision_score(labels, predicted, zero_division=0))
        recall = float(recall_score(labels, predicted, zero_division=0))
        f1 = float(f1_score(labels, predicted, zero_division=0))
        primary = recall if objective == "recall" else f1
        candidate = (
            -primary,
            -precision,
            -float(threshold),
            OperatingPoint(
                name=name,
                threshold=round(float(threshold), 6),
                max_fpr=max_fpr,
                policy_precision=round(precision, 8),
                policy_recall=round(recall, 8),
                policy_f1=round(f1, 8),
                policy_fpr=round(fpr, 8),
            ),
        )
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    if best is None:
        return OperatingPoint(
            name=name,
            threshold=1.0,
            max_fpr=max_fpr,
            policy_precision=0.0,
            policy_recall=0.0,
            policy_f1=0.0,
            policy_fpr=0.0,
        )
    return best[3]


def evaluate_held_out(
    probabilities: np.ndarray,
    labels: np.ndarray,
    operating_point: OperatingPoint,
    *,
    bootstrap_samples: int = 200,
    seed: int = 42,
) -> HeldOutMetrics:
    _require_binary(labels, "held-out test")
    predicted = probabilities >= operating_point.threshold
    positives = labels == 1
    negatives = labels == 0
    tp = int(np.sum(predicted & positives))
    fp = int(np.sum(predicted & negatives))
    tn = int(np.sum((~predicted) & negatives))
    fn = int(np.sum((~predicted) & positives))
    pr_auc = float(average_precision_score(labels, probabilities))
    precision = float(precision_score(labels, predicted, zero_division=0))
    recall = float(recall_score(labels, predicted, zero_division=0))

    rng = np.random.default_rng(seed)
    positive_index = np.flatnonzero(positives)
    negative_index = np.flatnonzero(negatives)
    boot_pr: list[float] = []
    boot_precision: list[float] = []
    boot_recall: list[float] = []
    for _ in range(bootstrap_samples):
        sample = np.concatenate(
            [
                rng.choice(positive_index, size=positive_index.size, replace=True),
                rng.choice(negative_index, size=negative_index.size, replace=True),
            ]
        )
        y = labels[sample]
        p = probabilities[sample]
        pred = p >= operating_point.threshold
        boot_pr.append(float(average_precision_score(y, p)))
        boot_precision.append(float(precision_score(y, pred, zero_division=0)))
        boot_recall.append(float(recall_score(y, pred, zero_division=0)))

    legitimate_count = max(int(np.sum(negatives)), 1)
    fraud_count = max(int(np.sum(positives)), 1)
    return HeldOutMetrics(
        pr_auc=round(pr_auc, 8),
        roc_auc=round(float(roc_auc_score(labels, probabilities)), 8),
        precision=round(precision, 8),
        recall=round(recall, 8),
        f1=round(float(f1_score(labels, predicted, zero_division=0)), 8),
        false_positive_rate=round(fp / legitimate_count, 8),
        brier=round(float(brier_score_loss(labels, probabilities)), 8),
        log_loss=round(float(log_loss(labels, _clip(probabilities))), 8),
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        false_positives_per_10000_legitimate=round(10000.0 * fp / legitimate_count, 8),
        false_negatives_per_100_fraud=round(100.0 * fn / fraud_count, 8),
        predicted_positive_rate=round(float(np.mean(predicted)), 8),
        pr_auc_ci95=_ci(boot_pr),
        precision_ci95=_ci(boot_precision),
        recall_ci95=_ci(boot_recall),
    )


def serialize(value: object) -> dict[str, object]:
    return asdict(cast(Any, value))


def _fit_method(
    method: CalibrationMethod,
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> object | None:
    if method == "uncalibrated":
        return None
    if method == "sigmoid":
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(_logit(probabilities), labels)
        return model
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(probabilities, labels)
        return model
    raise ValueError(f"unsupported calibration method: {method}")


def _require_binary(labels: np.ndarray, partition: str) -> None:
    if len(np.unique(labels)) != 2:
        raise ValueError(f"{partition} partition must contain both classes")


def _clip(probabilities: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)


def _logit(probabilities: np.ndarray) -> np.ndarray:
    p = _clip(probabilities)
    return np.log(p / (1.0 - p)).reshape(-1, 1)


def _ci(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    low, high = np.quantile(np.asarray(values, dtype=float), [0.025, 0.975])
    return (round(float(low), 8), round(float(high), 8))
