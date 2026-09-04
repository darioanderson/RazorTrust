from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .v3_research import LABEL_TO_INDEX, V3A_FEATURES, add_v3b_candidates

FINAL_FEATURES = (
    "scale_acceleration_gap",
    "identity_novelty_dispersion",
    "amount_tail_pressure",
    "auth_identity_interaction",
    "harm_acceleration",
    "stability_coherence",
)

FEATURE_JUSTIFICATIONS = {
    "scale_acceleration_gap": (
        "Absolute disagreement among volume, value, and ticket-size standardized changes; "
        "separates coherent legitimate growth from mix-shift anomalies."
    ),
    "identity_novelty_dispersion": (
        "New-device and new-geography shares weighted by their contemporaneous entropy; "
        "captures whether novelty is broadly dispersed rather than concentrated."
    ),
    "amount_tail_pressure": (
        "Amount-distribution divergence combined with positive ticket-size movement; "
        "a point-in-time proxy for harmful upper-tail pressure."
    ),
    "auth_identity_interaction": (
        "Failed authorization share interacting with identity novelty; distinguishes benign "
        "scale from access/payment-friction patterns."
    ),
    "harm_acceleration": (
        "Positive refund and chargeback deviations interacting with positive GMV movement; "
        "represents value-weighted downstream harm."
    ),
    "stability_coherence": (
        "Smoothly rewards agreement of positive volume and GMV growth while discounting "
        "authorization, refund, chargeback, and identity-novelty pressure."
    ),
}


@dataclass(frozen=True)
class ReleaseThreshold:
    threshold: float
    false_release_rate: float
    true_release_recall: float
    precision: float
    selected_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "false_release_rate": self.false_release_rate,
            "true_release_recall": self.true_release_recall,
            "precision": self.precision,
            "selected_count": self.selected_count,
        }


def add_final_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = add_v3b_candidates(frame)
    positive_gmv = result["gmv_delta_z"].clip(lower=0)
    positive_ticket = result["ticket_size_delta_z"].clip(lower=0)
    positive_refund = result["refund_rate_delta_z"].clip(lower=0)
    positive_chargeback = result["chargeback_rate_delta_z"].clip(lower=0)
    result["scale_acceleration_gap"] = (result["gmv_delta_z"] - result["volume_delta_z"]).abs() + (
        result["ticket_size_delta_z"] - result["gmv_delta_z"]
    ).abs()
    result["identity_novelty_dispersion"] = result["new_device_ratio"] * np.log1p(
        result["device_entropy"].clip(lower=0)
    ) + result["new_geo_ratio"] * np.log1p(result["geo_entropy"].clip(lower=0))
    result["amount_tail_pressure"] = np.log1p(result["amount_distribution_kl"].clip(lower=0)) * (
        1.0 + positive_ticket
    )
    result["auth_identity_interaction"] = result["failed_auth_ratio"] * (
        1.0 + result["new_device_ratio"] + result["new_geo_ratio"]
    )
    result["harm_acceleration"] = (positive_refund + positive_chargeback) * (
        1.0 + np.log1p(positive_gmv)
    )
    result["stability_coherence"] = np.sqrt(
        result["volume_delta_z"].clip(lower=0) * positive_gmv
    ) / (1.0 + result["auth_identity_interaction"] + result["harm_acceleration"])
    values = result.loc[:, FINAL_FEATURES].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("final-cycle feature derivation produced non-finite values")
    return result


def temporal_splits(frame: pd.DataFrame, *, bins: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds, stratified by family and isolated by merchant."""
    fold_id = np.full(len(frame), -1, dtype=int)
    for _, group in frame.groupby("scenario_family", sort=True):
        ordered = group.assign(
            _tie=group["merchant_id"]
            .astype(str)
            .map(lambda value: hashlib.sha256(value.encode()).hexdigest())
        ).sort_values(["hold_triggered_at", "_tie"])
        positions = np.arange(len(ordered))
        fold_id[ordered.index.to_numpy()] = np.minimum((positions * bins) // len(ordered), bins - 1)
    splits = []
    for validation_bin in range(2, bins):
        fit = np.flatnonzero(fold_id < validation_bin)
        validation = np.flatnonzero(fold_id == validation_bin)
        if len(set(frame.iloc[fit]["merchant_id"]) & set(frame.iloc[validation]["merchant_id"])):
            raise RuntimeError("merchant leakage in final-cycle temporal split")
        splits.append((fit, validation))
    return splits


def binary_parameters(seed: int, device: str, overrides: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": device,
        "random_state": seed,
        "n_jobs": 1,
        "n_estimators": 260,
        "max_depth": 3,
        "learning_rate": 0.04,
        "min_child_weight": 3.0,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.02,
        "reg_lambda": 2.0,
    }
    values.update(overrides)
    return values


def rolling_oof(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    *,
    seed: int,
    device: str,
    parameters: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    release = (frame["operational_target"].to_numpy() == "RELEASE").astype(int)
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for fold, (fit, validation) in enumerate(temporal_splits(frame), start=1):
        model = XGBClassifier(**binary_parameters(seed + fold, device, parameters))
        model.fit(
            frame.iloc[fit].loc[:, features],
            release[fit],
            sample_weight=compute_sample_weight(class_weight="balanced", y=release[fit]),
            verbose=False,
        )
        fold_scores = np.asarray(model.predict_proba(frame.iloc[validation].loc[:, features]))[:, 1]
        scores.append(fold_scores)
        labels.append(release[validation])
        records.append(
            {
                "fold": fold,
                "fit_rows": len(fit),
                "validation_rows": len(validation),
                "fit_latest_time": str(frame.iloc[fit]["hold_triggered_at"].max()),
                "validation_earliest_time": str(frame.iloc[validation]["hold_triggered_at"].min()),
                "merchant_overlap": 0,
            }
        )
    return np.concatenate(scores), np.concatenate(labels), records


def fit_calibrator(scores: np.ndarray, labels: np.ndarray, seed: int) -> LogisticRegression:
    model = LogisticRegression(random_state=seed, max_iter=1000)
    model.fit(logit(scores), labels)
    return model


def calibrate(calibrator: LogisticRegression, scores: np.ndarray) -> np.ndarray:
    return np.asarray(calibrator.predict_proba(logit(scores)))[:, 1]


def logit(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def select_release_threshold(
    scores: np.ndarray, labels: np.ndarray, *, max_false_release_rate: float = 0.05
) -> ReleaseThreshold:
    candidates = np.r_[np.nextafter(1.0, 2.0), np.unique(scores)]
    best: ReleaseThreshold | None = None
    non_release = labels == 0
    release = labels == 1
    for threshold in candidates:
        selected = scores >= threshold
        fpr = float(np.mean(selected[non_release]))
        if fpr > max_false_release_rate + 1e-12:
            continue
        recall = float(np.mean(selected[release]))
        precision = float(np.mean(labels[selected])) if selected.any() else 1.0
        candidate = ReleaseThreshold(
            threshold=float(threshold),
            false_release_rate=fpr,
            true_release_recall=recall,
            precision=precision,
            selected_count=int(selected.sum()),
        )
        if best is None or (recall, precision, -threshold) > (
            best.true_release_recall,
            best.precision,
            -best.threshold,
        ):
            best = candidate
    if best is None:
        raise RuntimeError("no release threshold satisfies the fixed false-release constraint")
    return best


def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": round(float(roc_auc_score(labels, scores)), 8),
        "partial_roc_auc_at_5pct_fpr": round(float(roc_auc_score(labels, scores, max_fpr=0.05)), 8),
        "average_precision": round(float(average_precision_score(labels, scores)), 8),
        "binary_log_loss": round(float(log_loss(labels, np.c_[1 - scores, scores])), 8),
        "binary_brier": round(float(np.mean((scores - labels) ** 2)), 8),
        "expected_calibration_error": round(binary_ece(labels, scores), 8),
    }


def binary_ece(labels: np.ndarray, scores: np.ndarray) -> float:
    result = 0.0
    for left, right in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:], strict=True):
        mask = (scores >= left) & ((scores <= right) if right == 1 else (scores < right))
        if mask.any():
            result += float(mask.mean()) * abs(float(scores[mask].mean() - labels[mask].mean()))
    return result


def reliability_table(labels: np.ndarray, scores: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for index, (left, right) in enumerate(
        zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:], strict=True)
    ):
        mask = (scores >= left) & ((scores <= right) if right == 1 else (scores < right))
        rows.append(
            {
                "bin": index,
                "left": left,
                "right": right,
                "count": int(mask.sum()),
                "mean_confidence": float(scores[mask].mean()) if mask.any() else None,
                "observed_release_rate": float(labels[mask].mean()) if mask.any() else None,
            }
        )
    return rows


def fit_release_model(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    *,
    seed: int,
    device: str,
    parameters: dict[str, Any],
) -> XGBClassifier:
    labels = (frame["operational_target"].to_numpy() == "RELEASE").astype(int)
    model = XGBClassifier(**binary_parameters(seed, device, parameters))
    model.fit(
        frame.loc[:, features],
        labels,
        sample_weight=compute_sample_weight(class_weight="balanced", y=labels),
        verbose=False,
    )
    return model


def fit_router_model(
    frame: pd.DataFrame, features: tuple[str, ...], *, seed: int, device: str
) -> XGBClassifier:
    subset = frame.loc[frame["operational_target"] != "RELEASE"]
    labels = (subset["operational_target"].to_numpy() == "ESCALATE").astype(int)
    model = XGBClassifier(**binary_parameters(seed, device, {"n_estimators": 220, "max_depth": 3}))
    model.fit(
        subset.loc[:, features],
        labels,
        sample_weight=compute_sample_weight(class_weight="balanced", y=labels),
        verbose=False,
    )
    return model


def predict_actions(
    release_model: XGBClassifier,
    router_model: XGBClassifier,
    calibrator: LogisticRegression,
    frame: pd.DataFrame,
    features: tuple[str, ...],
    release_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = frame.loc[:, features].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("non-finite inference input")
    release_raw = np.asarray(release_model.predict_proba(frame.loc[:, features]))[:, 1]
    release_probability = calibrate(calibrator, release_raw)
    escalate_conditional = np.asarray(router_model.predict_proba(frame.loc[:, features]))[:, 1]
    actions = np.full(len(frame), LABEL_TO_INDEX["EVIDENCE_NEEDED"], dtype=int)
    actions[escalate_conditional >= 0.5] = LABEL_TO_INDEX["ESCALATE"]
    actions[release_probability >= release_threshold] = LABEL_TO_INDEX["RELEASE"]
    probabilities = np.c_[
        release_probability,
        (1 - release_probability) * (1 - escalate_conditional),
        (1 - release_probability) * escalate_conditional,
    ]
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return actions, probabilities, release_probability


def development_error_structure(
    frame: pd.DataFrame, labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, Any]:
    # OOF rows are the last three of five within-family temporal bins.
    oof_indices = np.concatenate([validation for _, validation in temporal_splits(frame)])
    oof_frame = frame.iloc[oof_indices].reset_index(drop=True)
    selected = scores >= threshold
    false_release = (labels == 0) & selected
    missed_release = (labels == 1) & ~selected

    def grouped(mask: np.ndarray) -> list[dict[str, Any]]:
        counts = oof_frame.loc[mask].groupby("scenario_family").size()
        totals = oof_frame.groupby("scenario_family").size()
        return [
            {
                "scenario_family": str(family),
                "count": int(count),
                "within_family_rate": round(float(count / totals[family]), 8),
            }
            for family, count in counts.sort_values(ascending=False).items()
        ]

    contrasts = []
    for feature in (*V3A_FEATURES, *FINAL_FEATURES):
        values = oof_frame[feature].to_numpy(float)
        contrasts.append(
            {
                "feature": feature,
                "false_release_mean": float(values[false_release].mean())
                if false_release.any()
                else None,
                "missed_release_mean": float(values[missed_release].mean())
                if missed_release.any()
                else None,
                "all_rows_mean": float(values.mean()),
            }
        )
    return {
        "scope": "development temporal OOF only",
        "false_release_families": grouped(false_release),
        "missed_release_families": grouped(missed_release),
        "feature_contrasts": contrasts,
        "sealed_accessed": False,
        "unknown_family_accessed": False,
    }
