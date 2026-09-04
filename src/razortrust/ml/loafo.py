from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ..domain import StrictModel
from ..synthetic import TrueRiskState
from .dataset import model_matrix
from .modeling import IsolationForestScorer


class LoafoFamilyResult(StrictModel):
    held_out_family: str
    case_count: int
    isolation_forest_recall: float
    autoencoder_recall: float | None
    latent_distance_recall: float
    combined_recall: float
    legitimate_novelty_fpr: float
    combined_auroc: float
    combined_auprc: float


class LoafoReport(StrictModel):
    schema_version: str = "1.0"
    novelty_threshold: float
    families: list[LoafoFamilyResult]


def evaluate_leave_one_attack_family_out(
    frame: pd.DataFrame,
    *,
    novelty_threshold: float = 0.95,
    use_autoencoder: bool = False,
    seed: int = 42,
) -> LoafoReport:
    legitimate = frame[frame["true_risk_state"].astype(str) == TrueRiskState.LEGITIMATE]
    risky = frame[frame["true_risk_state"].astype(str) == TrueRiskState.RISKY]
    if legitimate.empty or risky.empty:
        raise ValueError("LOAFO requires legitimate and risky rows")
    features = model_matrix(legitimate)
    isolation_forest = IsolationForestScorer(seed=seed).fit(
        features,
        legitimate["cohort"],
        pd.Series(True, index=legitimate.index),
    )
    legitimate_if = isolation_forest.transform(features, legitimate["cohort"])
    mean = features.to_numpy(dtype=float).mean(axis=0)
    covariance = np.cov(features.to_numpy(dtype=float), rowvar=False)
    inverse_covariance = np.linalg.pinv(covariance + np.eye(covariance.shape[0]) * 1e-6)
    legitimate_distance = _mahalanobis(features.to_numpy(dtype=float), mean, inverse_covariance)
    sorted_distance = np.sort(legitimate_distance)

    autoencoder = None
    legitimate_ae: np.ndarray | None = None
    if use_autoencoder:
        from .novelty import AutoencoderNoveltyScorer

        autoencoder = AutoencoderNoveltyScorer(seed=seed).fit_group_cross_fitted(
            features,
            legitimate["merchant_id"].astype(str).reset_index(drop=True),
        )
        legitimate_ae = autoencoder.transform(features)

    results: list[LoafoFamilyResult] = []
    for family in sorted(risky["attack_family"].dropna().astype(str).unique()):
        held_out = risky[risky["attack_family"].astype(str) == family]
        held_features = model_matrix(held_out)
        held_if = isolation_forest.transform(held_features, held_out["cohort"])
        held_distance_raw = _mahalanobis(
            held_features.to_numpy(dtype=float), mean, inverse_covariance
        )
        held_distance = np.searchsorted(sorted_distance, held_distance_raw, side="right") / len(
            sorted_distance
        )
        legitimate_distance_percentile = np.searchsorted(
            sorted_distance, legitimate_distance, side="right"
        ) / len(sorted_distance)
        held_ae = autoencoder.transform(held_features) if autoencoder is not None else None
        held_signals = [held_if, held_distance]
        legitimate_signals = [legitimate_if, legitimate_distance_percentile]
        if held_ae is not None and legitimate_ae is not None:
            held_signals.append(held_ae)
            legitimate_signals.append(legitimate_ae)
        held_combined = np.mean(held_signals, axis=0)
        legitimate_combined = np.mean(legitimate_signals, axis=0)
        labels = np.concatenate([np.zeros(len(legitimate_combined)), np.ones(len(held_combined))])
        scores = np.concatenate([legitimate_combined, held_combined])
        results.append(
            LoafoFamilyResult(
                held_out_family=family,
                case_count=len(held_out),
                isolation_forest_recall=round(float(np.mean(held_if >= novelty_threshold)), 8),
                autoencoder_recall=(
                    round(float(np.mean(held_ae >= novelty_threshold)), 8)
                    if held_ae is not None
                    else None
                ),
                latent_distance_recall=round(float(np.mean(held_distance >= novelty_threshold)), 8),
                combined_recall=round(float(np.mean(held_combined >= novelty_threshold)), 8),
                legitimate_novelty_fpr=round(
                    float(np.mean(legitimate_combined >= novelty_threshold)), 8
                ),
                combined_auroc=round(float(roc_auc_score(labels, scores)), 8),
                combined_auprc=round(float(average_precision_score(labels, scores)), 8),
            )
        )
    return LoafoReport(novelty_threshold=novelty_threshold, families=results)


def _mahalanobis(
    values: np.ndarray, mean: np.ndarray, inverse_covariance: np.ndarray
) -> np.ndarray:
    centered = values - mean
    return np.sqrt(np.einsum("ij,jk,ik->i", centered, inverse_covariance, centered))
