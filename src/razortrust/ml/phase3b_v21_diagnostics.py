from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


@dataclass(frozen=True)
class SigmoidScoreCalibrator:
    coefficient: float
    intercept: float
    clip_epsilon: float = 1e-6

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        values = _logit(probabilities, self.clip_epsilon)
        logits = self.coefficient * values + self.intercept
        logits = np.clip(logits, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def as_dict(self) -> dict[str, float | str]:
        return {
            "method": "sigmoid_logit_platt_sample_calibration",
            "coefficient": round(self.coefficient, 10),
            "intercept": round(self.intercept, 10),
            "clip_epsilon": self.clip_epsilon,
        }


def fit_sigmoid_score_calibrator(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> SigmoidScoreCalibrator:
    labels = np.asarray(labels, dtype=int)
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("calibration labels must contain both classes")
    values = _logit(probabilities, 1e-6).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(values, labels)
    return SigmoidScoreCalibrator(
        coefficient=float(model.coef_[0, 0]),
        intercept=float(model.intercept_[0]),
    )


def probability_quality(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = _clip(probabilities)
    return {
        "pr_auc": round(float(average_precision_score(labels, probabilities)), 8),
        "brier": round(float(brier_score_loss(labels, probabilities)), 8),
        "log_loss": round(float(log_loss(labels, probabilities, labels=[0, 1])), 8),
    }


def operating_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    *,
    scenarios: np.ndarray | None = None,
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = probabilities >= float(threshold)
    negatives = labels == 0
    result: dict[str, float | int] = {
        "precision": round(float(precision_score(labels, predicted, zero_division=0)), 8),
        "recall": round(float(recall_score(labels, predicted, zero_division=0)), 8),
        "f1": round(float(f1_score(labels, predicted, zero_division=0)), 8),
        "sample_fpr": round(
            float(np.mean(predicted[negatives])) if np.any(negatives) else 0.0,
            8,
        ),
        "predicted_positive_rate": round(float(np.mean(predicted)), 8),
        "threshold": round(float(threshold), 6),
    }
    if scenarios is not None:
        scenarios = np.asarray(scenarios, dtype=int)
        if len(scenarios) != len(labels):
            raise ValueError("scenario count must match labels")
        for scenario in (1, 2, 3):
            mask = scenarios == scenario
            result[f"scenario_{scenario}_cases"] = int(np.sum(mask))
            result[f"scenario_{scenario}_recall"] = round(
                float(recall_score(labels[mask], predicted[mask], zero_division=0))
                if np.any(mask)
                else 0.0,
                8,
            )
    return result


def select_threshold_strategies(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    recall_floor: float,
    sample_fpr_cap: float,
) -> dict[str, float | None]:
    if not 0.0 < recall_floor <= 1.0:
        raise ValueError("recall_floor must be in (0, 1]")
    if not 0.0 <= sample_fpr_cap < 1.0:
        raise ValueError("sample_fpr_cap must be in [0, 1)")
    grid = np.linspace(0.01, 0.99, 99)
    rows = [
        operating_metrics(labels, probabilities, float(threshold))
        for threshold in grid
    ]

    high_tie = min(
        rows,
        key=lambda row: (-float(row["f1"]), -float(row["threshold"])),
    )
    low_tie = min(
        rows,
        key=lambda row: (-float(row["f1"]), float(row["threshold"])),
    )

    recall_candidates = [row for row in rows if float(row["recall"]) >= recall_floor]
    recall_choice = None
    if recall_candidates:
        recall_choice = min(
            recall_candidates,
            key=lambda row: (
                -float(row["precision"]),
                float(row["sample_fpr"]),
                -float(row["f1"]),
                float(row["threshold"]),
            ),
        )

    fpr_candidates = [row for row in rows if float(row["sample_fpr"]) <= sample_fpr_cap]
    fpr_choice = None
    if fpr_candidates:
        fpr_choice = min(
            fpr_candidates,
            key=lambda row: (
                -float(row["recall"]),
                -float(row["f1"]),
                -float(row["precision"]),
                float(row["threshold"]),
            ),
        )

    return {
        "max_f1_high_tie": float(high_tie["threshold"]),
        "max_f1_low_tie": float(low_tie["threshold"]),
        "recall_floor": (
            float(recall_choice["threshold"]) if recall_choice is not None else None
        ),
        "sample_fpr_cap": (
            float(fpr_choice["threshold"]) if fpr_choice is not None else None
        ),
    }


def score_summary(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    return {
        "all": _distribution(probabilities),
        "legitimate": _distribution(probabilities[labels == 0]),
        "fraud": _distribution(probabilities[labels == 1]),
    }


def quantile_calibration_bins(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 5,
) -> list[dict[str, float | int]]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(labels) != len(probabilities) or len(labels) == 0:
        raise ValueError("labels and probabilities must have the same non-zero length")
    order = np.argsort(probabilities, kind="mergesort")
    chunks = np.array_split(order, min(bins, len(order)))
    result: list[dict[str, float | int]] = []
    for index, chunk in enumerate(chunks, start=1):
        if len(chunk) == 0:
            continue
        result.append(
            {
                "bin": index,
                "count": int(len(chunk)),
                "mean_probability": round(float(np.mean(probabilities[chunk])), 8),
                "fraction_positive": round(float(np.mean(labels[chunk])), 8),
                "min_probability": round(float(np.min(probabilities[chunk])), 8),
                "max_probability": round(float(np.max(probabilities[chunk])), 8),
            }
        )
    return result


def embedding_drift_report(
    partitions: dict[str, np.ndarray],
) -> dict[str, object]:
    required = ("classifier_train", "calibration", "policy", "test")
    missing = [name for name in required if name not in partitions]
    if missing:
        raise ValueError(f"missing embedding partitions: {missing}")
    matrices = {name: np.asarray(partitions[name], dtype=float) for name in required}
    for name, matrix in matrices.items():
        if matrix.ndim != 2 or matrix.shape[0] == 0 or not np.isfinite(matrix).all():
            raise ValueError(f"invalid embedding matrix for {name}")

    summaries = {name: _embedding_summary(matrix) for name, matrix in matrices.items()}
    comparisons = {}
    for left, right in (
        ("classifier_train", "calibration"),
        ("calibration", "policy"),
        ("policy", "test"),
    ):
        comparisons[f"{left}_to_{right}"] = _embedding_shift(
            matrices[left], matrices[right]
        )
    return {"partitions": summaries, "transitions": comparisons}


def threshold_sweep(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, float]]:
    return [
        operating_metrics(labels, probabilities, float(threshold))
        for threshold in np.linspace(0.05, 0.95, 19)
    ]


def _embedding_summary(matrix: np.ndarray) -> dict[str, object]:
    norms = np.linalg.norm(matrix, axis=1)
    return {
        "rows": int(matrix.shape[0]),
        "dimensions": int(matrix.shape[1]),
        "mean_vector": [round(float(value), 8) for value in matrix.mean(axis=0)],
        "std_vector": [round(float(value), 8) for value in matrix.std(axis=0)],
        "l2_norm": _distribution(norms),
    }


def _embedding_shift(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    if left.shape[1] != right.shape[1]:
        raise ValueError("embedding dimensions must match")
    left_mean = left.mean(axis=0)
    right_mean = right.mean(axis=0)
    left_std = left.std(axis=0)
    right_std = right.std(axis=0)
    pooled = np.sqrt((left_std**2 + right_std**2) / 2.0 + 1e-12)
    standardized = np.abs(left_mean - right_mean) / pooled
    return {
        "mean_vector_l2_distance": round(float(np.linalg.norm(left_mean - right_mean)), 8),
        "rms_standardized_mean_shift": round(
            float(np.sqrt(np.mean(standardized**2))),
            8,
        ),
        "max_standardized_mean_shift": round(float(np.max(standardized)), 8),
    }


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": round(float(np.mean(values)), 8),
        "std": round(float(np.std(values)), 8),
        "q05": round(float(np.quantile(values, 0.05)), 8),
        "q25": round(float(np.quantile(values, 0.25)), 8),
        "q50": round(float(np.quantile(values, 0.50)), 8),
        "q75": round(float(np.quantile(values, 0.75)), 8),
        "q95": round(float(np.quantile(values, 0.95)), 8),
    }


def _logit(probabilities: np.ndarray, epsilon: float) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=float), epsilon, 1.0 - epsilon)
    return np.log(values / (1.0 - values))


def _clip(probabilities: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0 - 1e-12)
