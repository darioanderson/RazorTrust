from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from river.drift import ADWIN, KSWIN
from scipy.spatial.distance import jensenshannon
from scipy.stats import genpareto, ks_2samp

from ..domain import StrictModel


class FeatureDriftMetric(StrictModel):
    feature: str
    ks_statistic: float
    p_value: float
    p_value_bh: float
    js_distance: float
    drifted: bool


class BatchDriftReport(StrictModel):
    schema_version: str = "1.0"
    features: list[FeatureDriftMetric]
    drifted_feature_share: float


def batch_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    effect_threshold: float = 0.10,
    alpha: float = 0.05,
) -> BatchDriftReport:
    if list(reference.columns) != list(current.columns):
        raise ValueError("drift frames must use identical columns")
    raw: list[tuple[str, float, float, float]] = []
    for feature in reference.columns:
        reference_values = reference[feature].to_numpy(dtype=float)
        current_values = current[feature].to_numpy(dtype=float)
        ks = ks_2samp(reference_values, current_values)
        low = min(float(reference_values.min()), float(current_values.min()))
        high = max(float(reference_values.max()), float(current_values.max()))
        if high == low:
            js_distance = 0.0
        else:
            bins = np.linspace(low, high, 21)
            reference_hist = np.histogram(reference_values, bins=bins)[0] + 1e-12
            current_hist = np.histogram(current_values, bins=bins)[0] + 1e-12
            js_distance = float(jensenshannon(reference_hist, current_hist))
        raw.append((str(feature), float(ks.statistic), float(ks.pvalue), js_distance))
    adjusted = _benjamini_hochberg([item[2] for item in raw])
    metrics = [
        FeatureDriftMetric(
            feature=item[0],
            ks_statistic=round(item[1], 8),
            p_value=round(item[2], 8),
            p_value_bh=round(adjusted[index], 8),
            js_distance=round(item[3], 8),
            drifted=adjusted[index] < alpha and max(item[1], item[3]) >= effect_threshold,
        )
        for index, item in enumerate(raw)
    ]
    return BatchDriftReport(
        features=metrics,
        drifted_feature_share=round(float(np.mean([metric.drifted for metric in metrics])), 8),
    )


def write_evidently_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    html_path: str,
    json_path: str,
) -> None:
    from evidently import Report
    from evidently.presets import DataDriftPreset

    usable_columns = [
        column
        for column in reference.columns
        if reference[column].nunique() > 1 and current[column].nunique() > 1
    ]
    if not usable_columns:
        raise ValueError("Evidently drift report requires at least one non-constant feature")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy.*")
        snapshot = Report([DataDriftPreset()]).run(
            reference_data=reference[usable_columns],
            current_data=current[usable_columns],
        )
    snapshot.save_html(html_path)
    snapshot.save_json(json_path)


class OnlineChangeMonitor:
    def __init__(self, *, adwin_delta: float = 0.002, seed: int = 42) -> None:
        self.adwin = ADWIN(delta=adwin_delta)
        self.kswin = KSWIN(alpha=0.005, seed=seed)

    def update(self, value: float) -> dict[str, bool]:
        self.adwin.update(value)
        self.kswin.update(value)
        return {"adwin": self.adwin.drift_detected, "kswin": self.kswin.drift_detected}


class EvtTailSeverity:
    def __init__(self, *, quantile: float = 0.95, min_exceedances: int = 20) -> None:
        self.quantile = quantile
        self.min_exceedances = min_exceedances
        self.threshold: float | None = None
        self.parameters: tuple[float, float, float] | None = None

    def fit(self, legitimate_scores: np.ndarray) -> EvtTailSeverity:
        self.threshold = float(np.quantile(legitimate_scores, self.quantile))
        exceedances = legitimate_scores[legitimate_scores > self.threshold] - self.threshold
        if len(exceedances) < self.min_exceedances:
            raise ValueError("EVT requires more legitimate tail exceedances")
        fitted = genpareto.fit(exceedances, floc=0)
        self.parameters = (float(fitted[0]), float(fitted[1]), float(fitted[2]))
        return self

    def survival_probability(self, score: float) -> float:
        if self.threshold is None or self.parameters is None:
            raise RuntimeError("EVT tail model is not fitted")
        if score <= self.threshold:
            return 1.0
        shape, location, scale = self.parameters
        return float(genpareto.sf(score - self.threshold, shape, loc=location, scale=scale))


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values)
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(values) / np.arange(1, len(values) + 1))[::-1]
    )[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0, 1)
    return adjusted.tolist()
