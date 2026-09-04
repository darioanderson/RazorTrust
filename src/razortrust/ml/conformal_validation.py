from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from ..domain import HoldDecision
from .uncertainty import ApsConformalAbstainer


def wilson_interval(
    successes: int,
    total: int,
    *,
    z_value: float = 1.959963984540054,
) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0

    p_hat = successes / total
    z2 = z_value**2
    denominator = 1.0 + z2 / total
    center = (p_hat + z2 / (2.0 * total)) / denominator
    radius = (
        z_value
        * np.sqrt(
            (p_hat * (1.0 - p_hat) / total)
            + (z2 / (4.0 * total**2))
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def label_indices(frame: pd.DataFrame) -> np.ndarray:
    mapping = {
        "RELEASE": 0,
        "EVIDENCE_NEEDED": 1,
        "ESCALATE": 2,
    }
    labels = frame["operational_target"].astype(str).map(mapping)
    if labels.isna().any():
        unexpected = sorted(frame.loc[labels.isna(), "operational_target"].astype(str).unique())
        raise ValueError(f"unexpected operational_target values: {unexpected}")
    return labels.to_numpy(dtype=int)


def summarize_prediction_sets(
    abstainer: ApsConformalAbstainer,
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> dict[str, object]:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    sets = abstainer.prediction_sets(probabilities)
    if sets.shape != (len(labels), 3):
        raise ValueError(f"unexpected prediction-set shape: {sets.shape}")

    covered = sets[np.arange(len(labels)), labels]
    sizes = sets.sum(axis=1)

    routes = np.asarray(
        [
            str(abstainer.policy_decision(prediction_set, False))
            for prediction_set in sets
        ],
        dtype=object,
    )
    release_value = str(HoldDecision.RELEASE)
    evidence_value = str(HoldDecision.EVIDENCE_NEEDED)
    escalate_value = str(HoldDecision.ESCALATE)

    true_nonrelease = labels != 0
    false_release = (routes == release_value) & true_nonrelease
    release_count = int(np.sum(routes == release_value))
    false_release_count = int(np.sum(false_release))
    nonrelease_count = int(np.sum(true_nonrelease))

    ambiguous = sizes > 1
    empty = sizes == 0
    full = sizes == 3
    ambiguous_release = int(np.sum(ambiguous & (routes == release_value)))
    empty_release = int(np.sum(empty & (routes == release_value)))
    full_release = int(np.sum(full & (routes == release_value)))

    lower, upper = wilson_interval(int(np.sum(covered)), len(labels))

    class_coverage: dict[str, object] = {}
    for class_index, class_name in enumerate(
        ("RELEASE", "EVIDENCE_NEEDED", "ESCALATE")
    ):
        mask = labels == class_index
        support = int(np.sum(mask))
        class_coverage[class_name] = {
            "support": support,
            "coverage": (
                round(float(np.mean(covered[mask])), 8)
                if support
                else None
            ),
        }

    singleton = sizes == 1
    singleton_accuracy = (
        float(np.mean(covered[singleton])) if np.any(singleton) else 0.0
    )

    route_counts = Counter(routes.tolist())

    return {
        "rows": int(len(labels)),
        "empirical_coverage": round(float(np.mean(covered)), 8),
        "coverage_wilson_95": {
            "lower": round(float(lower), 8),
            "upper": round(float(upper), 8),
        },
        "average_set_size": round(float(np.mean(sizes)), 8),
        "singleton_rate": round(float(np.mean(singleton)), 8),
        "singleton_accuracy": round(singleton_accuracy, 8),
        "ambiguous_set_rate": round(float(np.mean(ambiguous)), 8),
        "empty_set_rate": round(float(np.mean(empty)), 8),
        "full_set_rate": round(float(np.mean(full)), 8),
        "class_coverage_diagnostic_only": class_coverage,
        "routing": {
            "release_recommendation_count": release_count,
            "release_recommendation_rate": round(release_count / len(labels), 8),
            "false_release_recommendation_count": false_release_count,
            "false_release_rate_among_nonrelease_truth": round(
                false_release_count / max(1, nonrelease_count),
                8,
            ),
            "nonrelease_truth_protected_rate": round(
                1.0 - false_release_count / max(1, nonrelease_count),
                8,
            ),
            "ambiguous_release_invariant_violations": ambiguous_release,
            "empty_release_invariant_violations": empty_release,
            "full_release_invariant_violations": full_release,
            "route_counts": {
                "RELEASE": int(route_counts.get(release_value, 0)),
                "EVIDENCE_NEEDED": int(route_counts.get(evidence_value, 0)),
                "ESCALATE": int(route_counts.get(escalate_value, 0)),
            },
        },
    }


def shift_model_matrix(
    matrix: pd.DataFrame,
    *,
    reference: pd.DataFrame,
    features: list[str],
    strength_reference_std: float,
) -> pd.DataFrame:
    shifted = matrix.copy()
    for feature in features:
        if feature not in shifted or feature not in reference:
            raise ValueError(f"shift feature not found: {feature}")
        scale = float(reference[feature].std(ddof=0))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"shift feature has unusable scale: {feature}")
        shifted.loc[:, feature] = (
            shifted[feature].astype(float)
            + float(strength_reference_std) * scale
        )
    return shifted
