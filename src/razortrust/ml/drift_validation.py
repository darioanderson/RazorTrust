from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

from .monitoring import OnlineChangeMonitor


@dataclass(frozen=True)
class ChronologicalDriftPartitions:
    train: np.ndarray
    calibration: np.ndarray
    calibration_fit: np.ndarray
    calibration_select: np.ndarray
    policy: np.ndarray
    monitor: np.ndarray
    monitor_windows: tuple[np.ndarray, np.ndarray, np.ndarray]


def chronological_drift_partitions(n_rows: int) -> ChronologicalDriftPartitions:
    if n_rows < 100:
        raise ValueError("Phase 5 public drift validation requires at least 100 rows")

    train_end = int(n_rows * 0.55)
    calibration_end = int(n_rows * 0.70)
    policy_end = int(n_rows * 0.85)
    calibration_mid = train_end + ((calibration_end - train_end) // 2)

    monitor = np.arange(policy_end, n_rows)
    windows = tuple(
        np.asarray(chunk, dtype=int)
        for chunk in np.array_split(monitor, 3)
    )
    if any(len(window) == 0 for window in windows):
        raise ValueError("monitor partition is too small for three chronological windows")

    return ChronologicalDriftPartitions(
        train=np.arange(0, train_end),
        calibration=np.arange(train_end, calibration_end),
        calibration_fit=np.arange(train_end, calibration_mid),
        calibration_select=np.arange(calibration_mid, calibration_end),
        policy=np.arange(calibration_end, policy_end),
        monitor=monitor,
        monitor_windows=(windows[0], windows[1], windows[2]),
    )


def binary_monitoring_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    actions = p >= float(threshold)

    positives = y == 1
    negatives = ~positives
    false_positives = int(np.sum(actions & negatives))
    false_negatives = int(np.sum((~actions) & positives))
    negative_count = int(np.sum(negatives))

    return {
        "rows": int(len(y)),
        "positive_rate": round(float(np.mean(y)), 8),
        "pr_auc": round(float(average_precision_score(y, p)), 8),
        "precision": round(float(precision_score(y, actions, zero_division=0)), 8),
        "recall": round(float(recall_score(y, actions, zero_division=0)), 8),
        "f1": round(float(f1_score(y, actions, zero_division=0)), 8),
        "sample_fpr": round(false_positives / max(1, negative_count), 8),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "brier": round(float(brier_score_loss(y, p)), 8),
        "log_loss": round(float(log_loss(y, np.clip(p, 1e-8, 1 - 1e-8))), 8),
        "score_mean": round(float(np.mean(p)), 8),
        "score_std": round(float(np.std(p)), 8),
    }


def performance_delta(
    reference: dict[str, float],
    current: dict[str, float],
) -> dict[str, float]:
    return {
        "pr_auc": round(float(current["pr_auc"] - reference["pr_auc"]), 8),
        "recall": round(float(current["recall"] - reference["recall"]), 8),
        "sample_fpr": round(float(current["sample_fpr"] - reference["sample_fpr"]), 8),
        "brier": round(float(current["brier"] - reference["brier"]), 8),
        "log_loss": round(float(current["log_loss"] - reference["log_loss"]), 8),
    }


def severity_from_signals(
    *,
    drifted_feature_share: float,
    score_drifted: bool,
    performance_delta_values: dict[str, float],
) -> tuple[str, dict[str, bool]]:
    performance_degraded = (
        performance_delta_values["pr_auc"] <= -0.05
        or performance_delta_values["recall"] <= -0.10
        or performance_delta_values["sample_fpr"] >= 0.01
    )
    distribution_signal = drifted_feature_share >= 0.10 or score_drifted

    if performance_degraded and distribution_signal:
        severity = "RED"
    elif performance_degraded or distribution_signal:
        severity = "AMBER"
    else:
        severity = "GREEN"

    return severity, {
        "performance_degraded": performance_degraded,
        "distribution_signal": distribution_signal,
        "feature_share_trigger": drifted_feature_share >= 0.10,
        "score_drift_trigger": score_drifted,
    }


def online_change_summary(
    reference_values: np.ndarray,
    current_values: np.ndarray,
    *,
    adwin_delta: float = 0.002,
    seed: int = 42,
) -> dict[str, object]:
    monitor = OnlineChangeMonitor(adwin_delta=adwin_delta, seed=seed)
    for value in np.asarray(reference_values, dtype=float):
        monitor.update(float(value))

    first_adwin: int | None = None
    first_kswin: int | None = None
    adwin_alerts = 0
    kswin_alerts = 0

    for index, value in enumerate(np.asarray(current_values, dtype=float)):
        alert = monitor.update(float(value))
        if alert["adwin"]:
            adwin_alerts += 1
            if first_adwin is None:
                first_adwin = index
        if alert["kswin"]:
            kswin_alerts += 1
            if first_kswin is None:
                first_kswin = index

    return {
        "adwin_detected": adwin_alerts > 0,
        "kswin_detected": kswin_alerts > 0,
        "any_detected": adwin_alerts > 0 or kswin_alerts > 0,
        "adwin_alert_count": adwin_alerts,
        "kswin_alert_count": kswin_alerts,
        "first_adwin_current_index": first_adwin,
        "first_kswin_current_index": first_kswin,
    }


def inject_standard_deviation_shift(
    frame: pd.DataFrame,
    *,
    reference: pd.DataFrame,
    features: list[str],
    strength: float,
) -> pd.DataFrame:
    shifted = frame.copy()
    for feature in features:
        if feature not in shifted or feature not in reference:
            raise ValueError(f"injection feature not found: {feature}")
        scale = float(reference[feature].std(ddof=0))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"injection feature has unusable reference scale: {feature}")
        shifted.loc[:, feature] = shifted[feature].astype(float) + strength * scale
    return shifted
