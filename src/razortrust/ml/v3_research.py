from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, build_info

from ..audit import canonical_json
from ..costs import DEFAULT_COST_MATRIX
from ..features import FEATURE_COLUMNS

LABELS = ("RELEASE", "EVIDENCE_NEEDED", "ESCALATE")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
UNKNOWN_FAMILIES = ("geo_expansion", "seasonal_peak", "slow_low_ring")
V3A_FEATURES = tuple(FEATURE_COLUMNS)
V3B_CANDIDATES = (
    "auth_novelty_pressure",
    "refund_harm_pressure",
    "scale_coherence",
    "legitimate_stability_score",
    "identity_dispersion_gap",
    "amount_velocity_pressure",
)
MAX_FALSE_RELEASE_RATE = 0.05
MIN_TRUE_RELEASE_RECALL = 0.20
DEFAULT_SEED = 20260903


@dataclass(frozen=True)
class ThresholdResult:
    release_threshold: float
    escalate_threshold: float
    false_release_rate: float
    true_release_recall: float
    expected_cost_units: float
    evidence_rate: float
    escalate_rate: float
    actions: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "release_threshold": self.release_threshold,
            "escalate_threshold": self.escalate_threshold,
            "false_release_rate": self.false_release_rate,
            "true_release_recall": self.true_release_recall,
            "expected_cost_units": self.expected_cost_units,
            "evidence_rate": self.evidence_rate,
            "escalate_rate": self.escalate_rate,
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_v31_frame(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        *V3A_FEATURES,
        "merchant_id",
        "scenario_family",
        "operational_target",
        "hold_triggered_at",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"v3.1 dataset is missing columns: {sorted(missing)}")
    if frame["merchant_id"].astype(str).duplicated().any():
        raise ValueError("v3 research requires one hold window per merchant")
    if not set(frame["operational_target"]).issubset(LABEL_TO_INDEX):
        raise ValueError("v3.1 dataset contains an unsupported operational target")
    values = frame.loc[:, V3A_FEATURES].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("v3.1 feature matrix contains non-finite values")
    result = frame.copy()
    result["hold_triggered_at"] = pd.to_datetime(result["hold_triggered_at"], utc=True)
    return result


def create_research_partitions(
    frame: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    unknown_families: Iterable[str] = UNKNOWN_FAMILIES,
) -> tuple[pd.Series, dict[str, Any]]:
    """Lock chronological outer partitions before looking at model performance.

    Three complete families are reserved for the final unknown-family evaluation.
    Within every remaining family, the earliest 70% is development, the next 15%
    is the v3B gate, and the latest 15% remains sealed. Hashes break timestamp ties.
    """
    unknown = set(unknown_families)
    absent = unknown.difference(set(frame["scenario_family"].astype(str)))
    if absent:
        raise ValueError(f"unknown-family holdout is absent from the dataset: {sorted(absent)}")
    partition = pd.Series("", index=frame.index, dtype=str)
    partition.loc[frame["scenario_family"].isin(unknown)] = "unknown_family"
    counts: dict[str, dict[str, int]] = {}
    for family, group in frame.loc[partition == ""].groupby("scenario_family", sort=True):
        ordered = group.assign(
            _tie=group["merchant_id"]
            .astype(str)
            .map(lambda value: hashlib.sha256(f"{seed}|{value}".encode()).hexdigest())
        ).sort_values(["hold_triggered_at", "_tie"])
        size = len(ordered)
        development_end = math.floor(size * 0.70)
        gate_end = math.floor(size * 0.85)
        if development_end < 1 or gate_end <= development_end or gate_end >= size:
            raise ValueError(f"family {family!r} is too small for 70/15/15 partitioning")
        partition.loc[ordered.index[:development_end]] = "development"
        partition.loc[ordered.index[development_end:gate_end]] = "v3b_gate"
        partition.loc[ordered.index[gate_end:]] = "sealed"
        counts[str(family)] = {
            "development": development_end,
            "v3b_gate": gate_end - development_end,
            "sealed": size - gate_end,
        }
    if (partition == "").any():
        raise RuntimeError("partition assignment left unassigned rows")
    merchant_sets = {
        name: set(frame.loc[partition == name, "merchant_id"].astype(str))
        for name in sorted(set(partition))
    }
    names = list(merchant_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            if merchant_sets[left] & merchant_sets[right]:
                raise RuntimeError(f"merchant leakage between {left} and {right}")
    manifest = {
        "schema_version": "1.0",
        "seed": seed,
        "strategy": "family_stratified_chronological_70_15_15_with_unknown_family_holdout",
        "unknown_families": sorted(unknown),
        "family_counts": counts,
        "partition_counts": partition.value_counts().sort_index().to_dict(),
        "merchant_hashes": {
            name: hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()
            for name, values in merchant_sets.items()
        },
    }
    manifest["content_sha256"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
    return partition, manifest


def add_v3b_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    positive_volume = result["volume_delta_z"].clip(lower=0)
    positive_gmv = result["gmv_delta_z"].clip(lower=0)
    positive_refund = result["refund_rate_delta_z"].clip(lower=0)
    positive_chargeback = result["chargeback_rate_delta_z"].clip(lower=0)
    result["auth_novelty_pressure"] = result["failed_auth_ratio"] * (
        1.0 + result["new_device_ratio"] + result["new_geo_ratio"]
    )
    result["refund_harm_pressure"] = positive_refund + positive_chargeback
    result["scale_coherence"] = np.sqrt(positive_volume * positive_gmv)
    result["legitimate_stability_score"] = np.exp(
        -(
            result["failed_auth_ratio"] * 4.0
            + result["new_device_ratio"]
            + result["new_geo_ratio"]
            + positive_refund.clip(upper=5) / 5.0
            + positive_chargeback.clip(upper=5) / 5.0
        )
    )
    result["identity_dispersion_gap"] = result["new_device_ratio"] / (
        1.0 + result["device_entropy"]
    )
    result["amount_velocity_pressure"] = positive_gmv * np.log1p(result["amount_distribution_kl"])
    values = result.loc[:, V3B_CANDIDATES].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("v3B candidate feature derivation produced non-finite values")
    return result


def target_indices(frame: pd.DataFrame) -> np.ndarray:
    return frame["operational_target"].map(LABEL_TO_INDEX).to_numpy(dtype=int)


def xgb_parameters(
    *, seed: int, device: str, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "objective": "multi:softprob",
        "num_class": 3,
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 2.0,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "eval_metric": "mlogloss",
        "random_state": seed,
        "n_jobs": 1,
        "tree_method": "hist",
        "device": device,
    }
    parameters.update(overrides or {})
    return parameters


def require_cuda(device: str) -> None:
    if not device.startswith("cuda"):
        return
    info = build_info()
    if not info.get("USE_CUDA"):
        raise RuntimeError("GPU execution was requested but XGBoost was built without CUDA")


def cross_validated_probabilities(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    *,
    seed: int,
    device: str,
    parameters: dict[str, Any] | None = None,
    folds: int = 5,
) -> np.ndarray:
    require_cuda(device)
    labels = target_indices(frame)
    probabilities = np.empty((len(frame), len(LABELS)), dtype=float)
    assigned = np.zeros(len(frame), dtype=bool)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (fit_index, validation_index) in enumerate(splitter.split(frame, labels), start=1):
        estimator = XGBClassifier(
            **xgb_parameters(
                seed=seed + fold,
                device=device,
                overrides=parameters,
            )
        )
        estimator.fit(
            frame.iloc[fit_index].loc[:, features],
            labels[fit_index],
            sample_weight=compute_sample_weight(class_weight="balanced", y=labels[fit_index]),
            verbose=False,
        )
        probabilities[validation_index] = estimator.predict_proba(
            frame.iloc[validation_index].loc[:, features]
        )
        assigned[validation_index] = True
    if not assigned.all():
        raise RuntimeError("cross-validation failed to score every development row")
    return probabilities


def fit_classifier(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    *,
    seed: int,
    device: str,
    parameters: dict[str, Any] | None = None,
) -> XGBClassifier:
    require_cuda(device)
    labels = target_indices(frame)
    estimator = XGBClassifier(**xgb_parameters(seed=seed, device=device, overrides=parameters))
    estimator.fit(
        frame.loc[:, features],
        labels,
        sample_weight=compute_sample_weight(class_weight="balanced", y=labels),
        verbose=False,
    )
    return estimator


def predict_probabilities(
    estimator: XGBClassifier,
    calibrator: LogisticRegression,
    frame: pd.DataFrame,
    features: tuple[str, ...],
) -> np.ndarray:
    validate_inference_frame(frame, features)
    raw = np.asarray(estimator.predict_proba(frame.loc[:, features]), dtype=float)
    return calibrate_probabilities(calibrator, raw)


def validate_inference_frame(frame: pd.DataFrame, features: tuple[str, ...]) -> None:
    missing = set(features).difference(frame.columns)
    if missing:
        raise ValueError(f"inference input is missing features: {sorted(missing)}")
    values = frame.loc[:, features].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("inference input contains non-finite feature values")


def fail_closed_action(
    estimator: XGBClassifier | None,
    calibrator: LogisticRegression | None,
    frame: pd.DataFrame,
    features: tuple[str, ...],
    thresholds: ThresholdResult,
) -> tuple[str, str | None]:
    """Operational boundary: every inference failure routes to human review."""
    try:
        if estimator is None or calibrator is None:
            raise RuntimeError("model or calibrator unavailable")
        probabilities = predict_probabilities(estimator, calibrator, frame, features)
        action = threshold_actions(
            probabilities,
            thresholds.release_threshold,
            thresholds.escalate_threshold,
        )[0]
        return LABELS[int(action)], None
    except (ArithmeticError, RuntimeError, TypeError, ValueError) as exc:
        return "EVIDENCE_NEEDED", type(exc).__name__


def fit_probability_calibrator(
    probabilities: np.ndarray, labels: np.ndarray, *, seed: int
) -> LogisticRegression:
    calibrator = LogisticRegression(
        max_iter=1000,
        random_state=seed,
    )
    calibrator.fit(_logit_features(probabilities), labels)
    return calibrator


def calibrate_probabilities(
    calibrator: LogisticRegression, probabilities: np.ndarray
) -> np.ndarray:
    return np.asarray(calibrator.predict_proba(_logit_features(probabilities)), dtype=float)


def _logit_features(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped)


def select_thresholds(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    max_false_release_rate: float = MAX_FALSE_RELEASE_RATE,
    min_true_release_recall: float = MIN_TRUE_RELEASE_RECALL,
) -> ThresholdResult:
    best: ThresholdResult | None = None
    matrix = np.asarray(DEFAULT_COST_MATRIX.matrix, dtype=float)
    for release_threshold in np.arange(0.30, 1.00, 0.01):
        for escalate_threshold in np.arange(0.30, 1.00, 0.01):
            actions = threshold_actions(probabilities, release_threshold, escalate_threshold)
            non_release = labels != LABEL_TO_INDEX["RELEASE"]
            true_release = labels == LABEL_TO_INDEX["RELEASE"]
            false_release_rate = float(np.mean(actions[non_release] == 0))
            true_release_recall = float(np.mean(actions[true_release] == 0))
            if (
                false_release_rate > max_false_release_rate
                or true_release_recall < min_true_release_recall
            ):
                continue
            candidate = ThresholdResult(
                release_threshold=round(float(release_threshold), 2),
                escalate_threshold=round(float(escalate_threshold), 2),
                false_release_rate=round(false_release_rate, 8),
                true_release_recall=round(true_release_recall, 8),
                expected_cost_units=round(float(np.mean(matrix[actions, labels])), 8),
                evidence_rate=round(float(np.mean(actions == 1)), 8),
                escalate_rate=round(float(np.mean(actions == 2)), 8),
                actions=actions.copy(),
            )
            score = (
                candidate.expected_cost_units,
                -candidate.true_release_recall,
                candidate.false_release_rate,
            )
            if best is None or score < (
                best.expected_cost_units,
                -best.true_release_recall,
                best.false_release_rate,
            ):
                best = candidate
    if best is None:
        raise ValueError("no threshold pair satisfies the locked false-release and recall target")
    return best


def threshold_actions(
    probabilities: np.ndarray, release_threshold: float, escalate_threshold: float
) -> np.ndarray:
    actions = np.full(len(probabilities), LABEL_TO_INDEX["EVIDENCE_NEEDED"], dtype=int)
    release = probabilities[:, 0] >= release_threshold
    actions[release] = LABEL_TO_INDEX["RELEASE"]
    actions[(~release) & (probabilities[:, 2] >= escalate_threshold)] = LABEL_TO_INDEX["ESCALATE"]
    return actions


def evaluate_actions(
    labels: np.ndarray,
    actions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    non_release = labels != 0
    true_release = labels == 0
    matrix = np.asarray(DEFAULT_COST_MATRIX.matrix, dtype=float)
    confusion = {
        actual: {
            predicted: int(np.sum((labels == actual_index) & (actions == predicted_index)))
            for predicted_index, predicted in enumerate(LABELS)
        }
        for actual_index, actual in enumerate(LABELS)
    }
    return {
        "row_count": len(labels),
        "false_release_count": int(np.sum((actions == 0) & non_release)),
        "false_release_rate": round(float(np.mean(actions[non_release] == 0)), 8),
        "true_release_recall": round(float(np.mean(actions[true_release] == 0)), 8),
        "evidence_rate": round(float(np.mean(actions == 1)), 8),
        "escalate_rate": round(float(np.mean(actions == 2)), 8),
        "expected_cost_units": round(float(np.mean(matrix[actions, labels])), 8),
        "log_loss": round(float(log_loss(labels, probabilities, labels=[0, 1, 2])), 8),
        "multiclass_brier": round(
            float(np.mean(np.sum((probabilities - np.eye(3)[labels]) ** 2, axis=1))), 8
        ),
        "expected_calibration_error": round(_expected_calibration_error(labels, probabilities), 8),
        "confusion": confusion,
    }


def _expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    total = 0.0
    boundaries = np.linspace(0, 1, 11)
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        mask = (confidence >= left) & (
            (confidence <= right) if right == 1.0 else (confidence < right)
        )
        if mask.any():
            total += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return total


def audit_v3a_errors(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    thresholds: ThresholdResult,
) -> dict[str, Any]:
    enriched = add_v3b_candidates(frame).reset_index(drop=True)
    labels = target_indices(enriched)
    actions = thresholds.actions
    false_release = (labels != 0) & (actions == 0)
    missed_release = (labels == 0) & (actions != 0)
    correct_non_release = (labels != 0) & (actions != 0)
    correct_release = (labels == 0) & (actions == 0)
    boundary = np.abs(probabilities[:, 0] - thresholds.release_threshold) <= 0.15
    boundary_labels = labels[boundary] == 0
    rankings: list[dict[str, Any]] = []
    for feature in (*V3A_FEATURES, *V3B_CANDIDATES):
        values = enriched[feature].to_numpy(dtype=float)
        boundary_auc = None
        direction = "higher_supports_release"
        if boundary_labels.any() and (~boundary_labels).any():
            auc = float(roc_auc_score(boundary_labels, values[boundary]))
            if auc < 0.5:
                auc = 1.0 - auc
                direction = "lower_supports_release"
            boundary_auc = round(auc, 8)
        rankings.append(
            {
                "feature": feature,
                "feature_set": "v3B_candidate" if feature in V3B_CANDIDATES else "v3A",
                "boundary_auc": boundary_auc,
                "direction": direction,
                "false_release_mean": _masked_mean(values, false_release),
                "correct_non_release_mean": _masked_mean(values, correct_non_release),
                "missed_release_mean": _masked_mean(values, missed_release),
                "correct_release_mean": _masked_mean(values, correct_release),
            }
        )
    rankings.sort(key=lambda item: item["boundary_auc"] or 0.0, reverse=True)
    recommended = [
        item["feature"]
        for item in rankings
        if item["feature_set"] == "v3B_candidate"
        and item["boundary_auc"] is not None
        and item["boundary_auc"] >= 0.54
    ][:4]
    return {
        "schema_version": "1.0",
        "experiment": "v3A_error_structure_audit",
        "data_scope": "development_oof_only",
        "sealed_test_accessed": False,
        "stress_test_accessed": False,
        "locked_target": {
            "maximum_false_release_rate": MAX_FALSE_RELEASE_RATE,
            "minimum_true_release_recall": MIN_TRUE_RELEASE_RECALL,
        },
        "thresholds": thresholds.as_dict(),
        "metrics": evaluate_actions(labels, actions, probabilities),
        "false_release_families": _family_errors(enriched, false_release),
        "missed_legitimate_families": _family_errors(enriched, missed_release),
        "near_boundary_row_count": int(boundary.sum()),
        "signal_ranking": rankings,
        "recommended_v3b_features": recommended,
        "recommendation_rule": "top four derived candidates with boundary AUC >= 0.54",
    }


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    return round(float(values[mask].mean()), 8) if mask.any() else None


def _family_errors(frame: pd.DataFrame, mask: np.ndarray) -> list[dict[str, Any]]:
    errors = frame.loc[mask].groupby("scenario_family").size()
    totals = frame.groupby("scenario_family").size()
    return [
        {
            "scenario_family": str(family),
            "count": int(count),
            "within_family_rate": round(float(count / totals.loc[family]), 8),
        }
        for family, count in errors.sort_values(ascending=False).items()
    ]


def write_json(path: str | Path, value: Any, *, immutable: bool = True) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n"
    if destination.exists() and immutable and destination.read_text(encoding="utf-8") != payload:
        raise FileExistsError(f"refusing to replace immutable research artifact: {destination}")
    destination.write_text(payload, encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"cannot serialize {type(value)!r}")
