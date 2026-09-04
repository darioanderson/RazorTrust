from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..domain import HoldDecision

NOVELTY_GUARD_VERSION = "raw-if-novelty-guard@1-research"
GuardRoute = Literal["EVIDENCE_NEEDED", "ESCALATE"]


@dataclass(frozen=True)
class NoveltyGuardCalibration:
    """Held-out calibration for a raw Isolation-Forest novelty guard.

    The guard deliberately consumes the *raw* anomaly score (higher = more
    anomalous in RazorTrust because the runtime stores ``-score_samples``).
    It does not reinterpret that score as a fraud probability.

    The threshold is an upper order statistic from legitimate calibration
    merchants only. The robust scale is diagnostic: it preserves severity
    beyond the threshold even when an empirical percentile is saturated.
    """

    target_false_alarm_rate: float
    raw_threshold: float
    robust_scale: float
    calibration_size: int
    order_statistic_rank: int
    version: str = NOVELTY_GUARD_VERSION
    calibration_source: str = "LEGITIMATE_CALIBRATION_MERCHANTS_ONLY"

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "version": self.version,
            "target_false_alarm_rate": self.target_false_alarm_rate,
            "raw_threshold": self.raw_threshold,
            "robust_scale": self.robust_scale,
            "calibration_size": self.calibration_size,
            "order_statistic_rank": self.order_statistic_rank,
            "calibration_source": self.calibration_source,
        }
        payload["content_sha256"] = _sha256_json(payload)
        return payload


@dataclass(frozen=True)
class NoveltyScoreBatch:
    raw_scores: np.ndarray
    signal: np.ndarray
    tail_excess: np.ndarray
    severity: np.ndarray
    empirical_tail_probability: np.ndarray


class CalibratedRawNoveltyGuard:
    """Research-only raw-score guard calibrated on held-out legitimate rows."""

    def __init__(self, calibration: NoveltyGuardCalibration, reference_scores: np.ndarray) -> None:
        reference = np.asarray(reference_scores, dtype=float)
        if reference.ndim != 1 or len(reference) != calibration.calibration_size:
            raise ValueError("reference scores do not match calibration metadata")
        if not np.isfinite(reference).all():
            raise ValueError("reference scores must be finite")
        self.calibration = calibration
        self.reference_scores = np.sort(reference)

    @classmethod
    def fit(
        cls,
        legitimate_calibration_raw_scores: Iterable[float],
        *,
        target_false_alarm_rate: float,
    ) -> CalibratedRawNoveltyGuard:
        if not 0 < target_false_alarm_rate < 0.5:
            raise ValueError("target_false_alarm_rate must be between 0 and 0.5")
        values = np.asarray(list(legitimate_calibration_raw_scores), dtype=float)
        if values.ndim != 1 or len(values) < 20:
            raise ValueError("novelty calibration requires at least 20 legitimate rows")
        if not np.isfinite(values).all():
            raise ValueError("novelty calibration scores must be finite")
        ordered = np.sort(values)

        # Finite-sample upper order statistic. The +1 convention is deliberately
        # conservative for small calibration sets and avoids pretending that a
        # 99th percentile is more precise than the number of held-out merchants.
        rank_1_based = int(math.ceil((len(ordered) + 1) * (1.0 - target_false_alarm_rate)))
        rank_1_based = min(max(rank_1_based, 1), len(ordered))
        threshold = float(ordered[rank_1_based - 1])

        median = float(np.median(ordered))
        mad = float(np.median(np.abs(ordered - median))) * 1.4826
        q25, q75 = np.quantile(ordered, [0.25, 0.75])
        iqr_scale = float((q75 - q25) / 1.349) if q75 > q25 else 0.0
        std = float(np.std(ordered, ddof=1)) if len(ordered) > 1 else 0.0
        robust_scale = max(mad, iqr_scale, std * 0.25, 1e-6)

        calibration = NoveltyGuardCalibration(
            target_false_alarm_rate=float(target_false_alarm_rate),
            raw_threshold=threshold,
            robust_scale=robust_scale,
            calibration_size=len(ordered),
            order_statistic_rank=rank_1_based,
        )
        return cls(calibration, ordered)

    def score(self, raw_scores: Iterable[float]) -> NoveltyScoreBatch:
        raw = np.asarray(list(raw_scores), dtype=float)
        if raw.ndim != 1:
            raise ValueError("raw scores must be one-dimensional")
        if not np.isfinite(raw).all():
            raise ValueError("raw scores must be finite")
        threshold = self.calibration.raw_threshold
        excess = np.maximum(0.0, raw - threshold)
        severity = excess / self.calibration.robust_scale
        # Smoothed upper-tail empirical probability. This is a diagnostic rank,
        # not a calibrated fraud probability or a production conformal claim.
        tail_counts = np.asarray(
            [np.count_nonzero(self.reference_scores >= value) for value in raw], dtype=float
        )
        tail_probability = (tail_counts + 1.0) / (len(self.reference_scores) + 1.0)
        return NoveltyScoreBatch(
            raw_scores=raw,
            signal=raw >= threshold,
            tail_excess=excess,
            severity=severity,
            empirical_tail_probability=tail_probability,
        )


def apply_guard_route(
    base_actions: Iterable[HoldDecision | str],
    novelty_signal: Iterable[bool],
    *,
    route: GuardRoute,
) -> np.ndarray:
    """Change only otherwise-RELEASE actions when the novelty guard fires."""
    actions = np.asarray([HoldDecision(str(value)) for value in base_actions], dtype=object)
    signal = np.asarray(list(novelty_signal), dtype=bool)
    if len(actions) != len(signal):
        raise ValueError("actions and novelty_signal must have equal length")
    replacement = HoldDecision(route)
    output = actions.copy()
    mask = (actions == HoldDecision.RELEASE) & signal
    output[mask] = replacement
    return output


def _sha256_json(payload: dict[str, object]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
