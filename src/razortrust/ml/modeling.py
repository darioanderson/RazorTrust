from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pydantic import Field
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from ..costs import DEFAULT_COST_MATRIX, CostMatrixArtifact
from ..domain import HoldDecision, StrictModel, Thresholds
from ..features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION
from ..policy_config import DEFAULT_DECISION_POLICY
from ..synthetic import TrueRiskState
from .dataset import model_matrix
from .splits import SplitManifest

ANOMALY_FEATURE = "isolation_forest_percentile"
MODEL_COLUMNS = (*FEATURE_COLUMNS, ANOMALY_FEATURE)
LABEL_TO_INDEX = {
    HoldDecision.RELEASE: 0,
    HoldDecision.EVIDENCE_NEEDED: 1,
    HoldDecision.ESCALATE: 2,
}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}


class CalibrationMetric(StrictModel):
    method: str
    log_loss: float
    multiclass_brier: float


class ThresholdSelection(StrictModel):
    thresholds: Thresholds
    expected_cost_units: float
    false_release_rate: float
    true_release_recall: float
    evidence_rate: float
    escalation_rate: float
    policy_version: str


class TrainingReport(StrictModel):
    model_version: str
    feature_schema_version: str
    split_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_weights_applied: bool
    anomaly_model_version: str
    anomaly_reference_mode: str
    calibration_comparison: list[CalibrationMetric]
    selected_calibration_method: str
    threshold_selection: ThresholdSelection
    cost_matrix_version: str
    cost_matrix_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    threshold_probability_source: str = "merchant_group_oof"


class IsolationForestScorer:
    """Continuous pooled/per-cohort novelty signal fitted on legitimate training rows.

    Version 2 keeps the final Isolation Forest fitted on all legitimate training rows,
    but builds the empirical percentile reference from merchant-group out-of-fold
    scores. That prevents a row from defining its own novelty percentile. Legacy
    signed releases without ``reference_mode`` continue to load as version 1.
    """

    version = "isolation-forest@1"

    def __init__(self, *, seed: int = 42, min_cohort_rows: int = 40) -> None:
        self.seed = seed
        self.min_cohort_rows = min_cohort_rows
        self.models: dict[str, IsolationForest] = {}
        self.reference_scores: dict[str, np.ndarray] = {}
        self.version = "isolation-forest@2"
        self.reference_mode = "merchant_group_oof"

    def fit(
        self,
        features: pd.DataFrame,
        cohorts: pd.Series,
        legitimate_mask: pd.Series,
        merchant_groups: pd.Series | None = None,
    ) -> IsolationForestScorer:
        reference = features.loc[legitimate_mask].reset_index(drop=True)
        if len(reference) < 2:
            raise ValueError("Isolation Forest requires at least two legitimate reference rows")
        reference_cohorts = cohorts.loc[legitimate_mask].astype(str).reset_index(drop=True)
        if merchant_groups is None:
            reference_groups = pd.Series(
                [f"row-{index}" for index in range(len(reference))], dtype=str
            )
        else:
            reference_groups = (
                merchant_groups.loc[legitimate_mask].astype(str).reset_index(drop=True)
            )
        self._fit_partition("pooled", reference, reference_groups)
        for cohort in sorted(set(reference_cohorts)):
            mask = reference_cohorts == cohort
            cohort_rows = reference.loc[mask].reset_index(drop=True)
            cohort_groups = reference_groups.loc[mask].reset_index(drop=True)
            if len(cohort_rows) >= self.min_cohort_rows and cohort_groups.nunique() >= 2:
                self._fit_partition(cohort, cohort_rows, cohort_groups)
        return self

    def transform(self, features: pd.DataFrame, cohorts: pd.Series) -> np.ndarray:
        return self.score_details(features, cohorts).percentiles

    def score_details(
        self, features: pd.DataFrame, cohorts: pd.Series
    ) -> IsolationForestScoreDetails:
        percentiles = np.empty(len(features), dtype=float)
        raw_scores = np.empty(len(features), dtype=float)
        reference_max_scores = np.empty(len(features), dtype=float)
        tail_excesses = np.empty(len(features), dtype=float)
        reference_sizes = np.empty(len(features), dtype=int)
        partitions: list[str] = []
        normalized_cohorts = cohorts.astype(str).reset_index(drop=True)
        normalized_features = features.reset_index(drop=True)
        for row_index in range(len(normalized_features)):
            cohort = normalized_cohorts.iloc[row_index]
            partition = cohort if cohort in self.models else "pooled"
            row = normalized_features.iloc[[row_index]]
            raw_score = -float(self.models[partition].score_samples(row)[0])
            reference = self.reference_scores[partition]
            reference_max = float(reference[-1])
            percentile = float(np.searchsorted(reference, raw_score, side="right") / len(reference))
            percentiles[row_index] = percentile
            raw_scores[row_index] = raw_score
            reference_max_scores[row_index] = reference_max
            tail_excesses[row_index] = max(0.0, raw_score - reference_max)
            reference_sizes[row_index] = len(reference)
            partitions.append(partition)
        return IsolationForestScoreDetails(
            percentiles=percentiles,
            raw_scores=raw_scores,
            reference_max_scores=reference_max_scores,
            tail_excesses=tail_excesses,
            reference_sizes=reference_sizes,
            partitions=partitions,
            scorer_version=str(getattr(self, "version", "isolation-forest@1")),
            reference_mode=str(getattr(self, "reference_mode", "legacy_in_sample")),
        )

    def _fit_partition(
        self, partition: str, features: pd.DataFrame, merchant_groups: pd.Series
    ) -> None:
        reference_scores = self._group_oof_reference_scores(
            features, merchant_groups, partition=partition
        )
        model = self._new_model(len(features), seed_offset=self._partition_seed(partition))
        model.fit(features)
        self.models[partition] = model
        self.reference_scores[partition] = np.sort(reference_scores)

    def _group_oof_reference_scores(
        self, features: pd.DataFrame, merchant_groups: pd.Series, *, partition: str
    ) -> np.ndarray:
        groups = merchant_groups.astype(str).reset_index(drop=True)
        unique_groups = int(groups.nunique())
        if unique_groups < 2:
            raise ValueError(
                f"Isolation Forest {partition} reference requires at least two merchant groups"
            )
        n_splits = min(5, unique_groups)
        splitter = GroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.seed + self._partition_seed(partition),
        )
        scores = np.empty(len(features), dtype=float)
        assigned = np.zeros(len(features), dtype=bool)
        for fold, (fit_index, reference_index) in enumerate(
            splitter.split(features, groups=groups), start=1
        ):
            fold_model = self._new_model(
                len(fit_index),
                seed_offset=self._partition_seed(partition) + fold,
            )
            fold_model.fit(features.iloc[fit_index])
            scores[reference_index] = -fold_model.score_samples(features.iloc[reference_index])
            assigned[reference_index] = True
        if not assigned.all():
            raise RuntimeError("Isolation Forest cross-fitting did not score every reference row")
        return scores

    def _new_model(self, rows: int, *, seed_offset: int) -> IsolationForest:
        return IsolationForest(
            n_estimators=300,
            max_samples=min(512, rows),
            contamination="auto",
            max_features=1.0,
            random_state=self.seed + seed_offset,
            n_jobs=1,
        )

    @staticmethod
    def _partition_seed(partition: str) -> int:
        digest = hashlib.sha256(partition.encode("utf-8")).digest()
        return int.from_bytes(digest[:2], "big")


@dataclass(frozen=True)
class IsolationForestScoreDetails:
    percentiles: np.ndarray
    raw_scores: np.ndarray
    reference_max_scores: np.ndarray
    tail_excesses: np.ndarray
    reference_sizes: np.ndarray
    partitions: list[str]
    scorer_version: str
    reference_mode: str


@dataclass
class ModelBundle:
    model_version: str
    anomaly_scorer: IsolationForestScorer
    classifier: XGBClassifier
    calibrator: Any | None
    calibration_method: str
    thresholds: Thresholds
    policy_version: str
    cost_matrix_version: str
    cost_matrix_sha256: str

    def predict_proba(self, features: pd.DataFrame, cohorts: pd.Series) -> np.ndarray:
        model_features = _with_anomaly(features, cohorts, self.anomaly_scorer)
        estimator = self.calibrator or self.classifier
        return _normalize_probabilities(estimator.predict_proba(model_features))


@dataclass
class TrainingResult:
    bundle: ModelBundle
    report: TrainingReport
    policy_probabilities: np.ndarray
    policy_labels: np.ndarray


def train_model_bundle(
    frame: pd.DataFrame,
    split_manifest: SplitManifest,
    *,
    seed: int = 42,
    n_estimators: int = 300,
    xgb_params: dict[str, Any] | None = None,
) -> TrainingResult:
    """Fit the Phase 1 IF -> XGBoost -> calibration -> cost-policy pipeline."""
    train = _partition(frame, split_manifest.train_merchants)
    calibration = _partition(frame, split_manifest.calibration_merchants)
    anomaly_scorer = IsolationForestScorer(seed=seed).fit(
        model_matrix(train),
        train["cohort"],
        train["true_risk_state"].astype(str) == TrueRiskState.LEGITIMATE,
        train["merchant_id"].astype(str),
    )
    train_features = _with_anomaly(model_matrix(train), train["cohort"], anomaly_scorer)
    y_train = _target_indices(train)
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    classifier_parameters: dict[str, Any] = {
        "objective": "multi:softprob",
        "num_class": 3,
        "n_estimators": n_estimators,
        "max_depth": 4,
        "learning_rate": 0.08,
        "min_child_weight": 2,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "eval_metric": "mlogloss",
        "random_state": seed,
        "n_jobs": 1,
        "tree_method": "hist",
    }
    classifier_parameters.update(xgb_params or {})
    classifier = XGBClassifier(**classifier_parameters)
    classifier.fit(train_features, y_train, sample_weight=sample_weights, verbose=False)

    calibration_features = _with_anomaly(
        model_matrix(calibration), calibration["cohort"], anomaly_scorer
    )
    y_calibration = _target_indices(calibration)
    selected_method, calibrator, comparison, policy_probabilities = _select_calibrator(
        classifier,
        calibration_features,
        y_calibration,
        calibration["merchant_id"].astype(str),
        seed=seed + 2,
    )
    threshold_selection = optimize_thresholds(policy_probabilities, y_calibration)
    model_version = "xgb-if-settlement@2"
    bundle = ModelBundle(
        model_version=model_version,
        anomaly_scorer=anomaly_scorer,
        classifier=classifier,
        calibrator=calibrator,
        calibration_method=selected_method,
        thresholds=threshold_selection.thresholds,
        policy_version=threshold_selection.policy_version,
        cost_matrix_version=DEFAULT_COST_MATRIX.cost_matrix_version,
        cost_matrix_sha256=DEFAULT_COST_MATRIX.content_sha256,
    )
    return TrainingResult(
        bundle=bundle,
        policy_probabilities=policy_probabilities,
        policy_labels=y_calibration,
        report=TrainingReport(
            model_version=model_version,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            split_manifest_sha256=split_manifest.content_sha256,
            sample_weights_applied=True,
            anomaly_model_version=anomaly_scorer.version,
            anomaly_reference_mode=anomaly_scorer.reference_mode,
            calibration_comparison=comparison,
            selected_calibration_method=selected_method,
            threshold_selection=threshold_selection,
            cost_matrix_version=DEFAULT_COST_MATRIX.cost_matrix_version,
            cost_matrix_sha256=DEFAULT_COST_MATRIX.content_sha256,
        ),
    )


def optimize_thresholds(
    probabilities: np.ndarray,
    true_labels: np.ndarray,
    *,
    max_false_release_rate: float = DEFAULT_DECISION_POLICY.maximum_false_release_rate,
    min_true_release_recall: float = 0.20,
    cost_matrix: CostMatrixArtifact = DEFAULT_COST_MATRIX,
) -> ThresholdSelection:
    best: ThresholdSelection | None = None
    for release_threshold in np.arange(0.30, 1.00, 0.02):
        for escalate_threshold in np.arange(0.30, 1.00, 0.02):
            actions = _threshold_actions(
                probabilities, float(release_threshold), float(escalate_threshold)
            )
            non_release = true_labels != LABEL_TO_INDEX[HoldDecision.RELEASE]
            false_release_rate = float(
                np.mean(actions[non_release] == LABEL_TO_INDEX[HoldDecision.RELEASE])
            )
            if false_release_rate > max_false_release_rate:
                continue
            true_release = true_labels == LABEL_TO_INDEX[HoldDecision.RELEASE]
            true_release_recall = float(
                np.mean(actions[true_release] == LABEL_TO_INDEX[HoldDecision.RELEASE])
            )
            if true_release_recall < min_true_release_recall:
                continue
            matrix = np.asarray(cost_matrix.matrix, dtype=float)
            expected_cost = float(np.mean(matrix[actions, true_labels]))
            candidate = ThresholdSelection(
                thresholds=Thresholds(
                    release=round(float(release_threshold), 2),
                    escalate=round(float(escalate_threshold), 2),
                ),
                expected_cost_units=round(expected_cost, 6),
                false_release_rate=round(false_release_rate, 6),
                true_release_recall=round(true_release_recall, 6),
                evidence_rate=round(
                    float(np.mean(actions == LABEL_TO_INDEX[HoldDecision.EVIDENCE_NEEDED])), 6
                ),
                escalation_rate=round(
                    float(np.mean(actions == LABEL_TO_INDEX[HoldDecision.ESCALATE])), 6
                ),
                policy_version="cost-policy@2",
            )
            if best is None or candidate.expected_cost_units < best.expected_cost_units:
                best = candidate
    if best is None:
        raise ValueError("no threshold pair satisfies the false-release constraint")
    return best


def save_model_bundle(path: str | Path, result: TrainingResult) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated.*",
            category=DeprecationWarning,
        )
        joblib.dump(result.bundle, destination)
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def load_model_bundle(path: str | Path) -> ModelBundle:
    # NumPy 2.5 warns while reading joblib's established structured dtype metadata.
    # Keep that third-party compatibility warning out of API startup and test output.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                ".*(align should be passed as Python or NumPy boolean|"
                "Setting the shape on a NumPy array has been deprecated).*"
            ),
            category=DeprecationWarning,
        )
        bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle):
        raise ValueError("artifact is not a RazorTrust ModelBundle")
    return bundle


def _partition(frame: pd.DataFrame, merchants: list[str]) -> pd.DataFrame:
    partition = frame[frame["merchant_id"].isin(merchants)].reset_index(drop=True)
    if partition.empty:
        raise ValueError("model partition is empty")
    return partition


def _target_indices(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        [LABEL_TO_INDEX[HoldDecision(value)] for value in frame["operational_target"]],
        dtype=int,
    )


def _with_anomaly(
    features: pd.DataFrame,
    cohorts: pd.Series,
    scorer: IsolationForestScorer,
) -> pd.DataFrame:
    result = features.copy()
    result[ANOMALY_FEATURE] = scorer.transform(features, cohorts)
    return result.loc[:, list(MODEL_COLUMNS)]


def _select_calibrator(
    classifier: XGBClassifier,
    features: pd.DataFrame,
    labels: np.ndarray,
    groups: pd.Series,
    *,
    seed: int,
) -> tuple[str, Any | None, list[CalibrationMetric], np.ndarray]:
    fit_idx, selection_idx = _calibration_split(features, labels, groups, seed=seed)
    comparison: list[CalibrationMetric] = []
    candidates: dict[str, Any | None] = {"uncalibrated": None}
    for method in ("sigmoid", "isotonic", "temperature"):
        calibrator = CalibratedClassifierCV(
            FrozenEstimator(classifier),
            method=method,
            cv=_calibration_cv(labels[fit_idx]),
            n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The least populated class in y has only .*",
                category=UserWarning,
            )
            calibrator.fit(features.iloc[fit_idx], labels[fit_idx])
        candidates[method] = calibrator
    for method, calibrator in candidates.items():
        estimator = calibrator or classifier
        probabilities = _normalize_probabilities(
            estimator.predict_proba(features.iloc[selection_idx])
        )
        comparison.append(
            CalibrationMetric(
                method=method,
                log_loss=round(
                    float(log_loss(labels[selection_idx], probabilities, labels=[0, 1, 2])),
                    8,
                ),
                multiclass_brier=round(_multiclass_brier(labels[selection_idx], probabilities), 8),
            )
        )
    selected = min(comparison, key=lambda metric: (metric.log_loss, metric.multiclass_brier))
    if selected.method == "uncalibrated":
        policy_probabilities = _normalize_probabilities(classifier.predict_proba(features))
        return selected.method, None, comparison, policy_probabilities
    policy_probabilities = _cross_fitted_calibration_probabilities(
        classifier, selected.method, features, labels, groups, seed=seed + 1
    )
    final_calibrator = CalibratedClassifierCV(
        FrozenEstimator(classifier),
        method=selected.method,
        cv=_calibration_cv(labels),
        n_jobs=1,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The least populated class in y has only .*",
            category=UserWarning,
        )
        final_calibrator.fit(features, labels)
    return selected.method, final_calibrator, comparison, policy_probabilities


def _cross_fitted_calibration_probabilities(
    classifier: XGBClassifier,
    method: str,
    features: pd.DataFrame,
    labels: np.ndarray,
    groups: pd.Series,
    *,
    seed: int,
) -> np.ndarray:
    class_counts = np.bincount(labels, minlength=3)
    n_splits = min(4, int(class_counts.min()), groups.nunique())
    if n_splits < 2:
        raise ValueError("cross-fitted calibration requires at least two merchants per class")
    probabilities = np.empty((len(features), 3), dtype=float)
    assigned = np.zeros(len(features), dtype=bool)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fit_idx, policy_idx in splitter.split(features, labels, groups):
        calibrator = CalibratedClassifierCV(
            FrozenEstimator(classifier),
            method=method,
            cv=_calibration_cv(labels[fit_idx]),
            n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The least populated class in y has only .*",
                category=UserWarning,
            )
            calibrator.fit(features.iloc[fit_idx], labels[fit_idx])
        probabilities[policy_idx] = _normalize_probabilities(
            calibrator.predict_proba(features.iloc[policy_idx])
        )
        assigned[policy_idx] = True
    if not assigned.all():
        raise RuntimeError("cross-fitted calibration did not score every policy row")
    return probabilities


def _calibration_cv(labels: np.ndarray) -> int | list[tuple[np.ndarray, np.ndarray]]:
    """Choose a valid fold count while the frozen base estimator supplies predictions."""
    class_counts = np.bincount(labels, minlength=3)
    if np.any(class_counts == 0):
        raise ValueError("calibration requires at least one row per class")
    minimum_class_count = int(class_counts.min())
    if minimum_class_count >= 2:
        return min(5, minimum_class_count)
    # With a FrozenEstimator, folds only partition prediction rows; the base model is
    # never refitted. A concrete two-fold iterable avoids sklearn's stratified-CV
    # class-count rejection on tiny development datasets while keeping each row OOF.
    indices = np.arange(len(labels))
    return list(KFold(n_splits=2, shuffle=True, random_state=42).split(indices))


def _calibration_split(
    features: pd.DataFrame,
    labels: np.ndarray,
    groups: pd.Series,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    expected = set(labels)
    splitter = GroupShuffleSplit(n_splits=100, test_size=0.5, random_state=seed)
    for fit_idx, selection_idx in splitter.split(features, labels, groups):
        if set(labels[fit_idx]) == expected and set(labels[selection_idx]) == expected:
            return fit_idx, selection_idx
    raise ValueError("calibration merchants cannot be split with every class on both sides")


def _multiclass_brier(labels: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.eye(3)[labels]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def _normalize_probabilities(probabilities: Any) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    row_totals = values.sum(axis=1, keepdims=True)
    if np.any(row_totals <= 0):
        raise ValueError("model returned a probability row with no mass")
    return values / row_totals


def _threshold_actions(
    probabilities: np.ndarray, release_threshold: float, escalate_threshold: float
) -> np.ndarray:
    actions = np.full(len(probabilities), LABEL_TO_INDEX[HoldDecision.EVIDENCE_NEEDED])
    actions[probabilities[:, LABEL_TO_INDEX[HoldDecision.RELEASE]] >= release_threshold] = (
        LABEL_TO_INDEX[HoldDecision.RELEASE]
    )
    release_mask = actions == LABEL_TO_INDEX[HoldDecision.RELEASE]
    actions[
        (~release_mask)
        & (probabilities[:, LABEL_TO_INDEX[HoldDecision.ESCALATE]] >= escalate_threshold)
    ] = LABEL_TO_INDEX[HoldDecision.ESCALATE]
    return actions
