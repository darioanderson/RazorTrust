from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .autoencoder_ood import calibrate_upper_tail

CONSENSUS_RULES = ("AE_ANY", "AE_BOTH", "TWO_OF_THREE")


@dataclass(frozen=True)
class FamilyRobustThreshold:
    target_false_alarm_rate: float
    threshold: float
    family_thresholds: dict[str, float]
    family_sizes: dict[str, int]


def family_robust_upper_tail(
    scores: np.ndarray,
    families: np.ndarray,
    *,
    target_false_alarm_rate: float,
) -> FamilyRobustThreshold:
    """Calibrate to the most conservative legitimate-family upper-tail threshold.

    Each legitimate family gets its own finite-sample threshold. The final threshold
    is the maximum across families so one easy/large family cannot dominate the
    calibration. This is a research guardrail for family shift, not a coverage claim.
    """
    values = np.asarray(scores, dtype=float)
    labels = np.asarray(families, dtype=str)
    if values.ndim != 1 or labels.ndim != 1 or len(values) != len(labels):
        raise ValueError("scores and families must be same-length 1-D arrays")
    if len(values) < 20 or not np.isfinite(values).all():
        raise ValueError("family-robust calibration requires at least 20 finite scores")
    unique = sorted(set(labels.tolist()))
    if len(unique) < 2:
        raise ValueError("family-robust calibration requires at least two legitimate families")

    thresholds: dict[str, float] = {}
    sizes: dict[str, int] = {}
    for family in unique:
        family_values = values[labels == family]
        if len(family_values) < 20:
            # Keep the finite-sample tail calibration contract honest. Small
            # families are omitted rather than pretending a few rows define a
            # robust extreme tail. Phase 2C deliberately supplies larger
            # pseudo-unseen legitimate family calibration groups.
            continue
        calibrated = calibrate_upper_tail(
            family_values,
            target_false_alarm_rate=target_false_alarm_rate,
        )
        thresholds[family] = float(calibrated.threshold)
        sizes[family] = int(len(family_values))

    if len(thresholds) < 2:
        raise ValueError("fewer than two legitimate families have enough calibration rows")
    return FamilyRobustThreshold(
        target_false_alarm_rate=float(target_false_alarm_rate),
        threshold=float(max(thresholds.values())),
        family_thresholds=thresholds,
        family_sizes=sizes,
    )


def consensus_signal(
    rule: str,
    reconstruction_signal: np.ndarray,
    latent_signal: np.ndarray,
    isolation_signal: np.ndarray,
) -> np.ndarray:
    reconstruction_signal = np.asarray(reconstruction_signal, dtype=bool)
    latent_signal = np.asarray(latent_signal, dtype=bool)
    isolation_signal = np.asarray(isolation_signal, dtype=bool)
    if not (reconstruction_signal.shape == latent_signal.shape == isolation_signal.shape):
        raise ValueError("novelty signal arrays must have identical shapes")
    if rule == "AE_ANY":
        return reconstruction_signal | latent_signal
    if rule == "AE_BOTH":
        return reconstruction_signal & latent_signal
    if rule == "TWO_OF_THREE":
        return (
            reconstruction_signal.astype(np.int8)
            + latent_signal.astype(np.int8)
            + isolation_signal.astype(np.int8)
        ) >= 2
    raise ValueError(f"unknown consensus rule: {rule}")
