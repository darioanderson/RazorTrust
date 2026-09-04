from __future__ import annotations

import numpy as np
from mapie.classification import SplitConformalClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

from ..domain import HoldDecision, StrictModel


class _ProbabilityEstimator(ClassifierMixin, BaseEstimator):
    """Expose precomputed calibrated probabilities through sklearn's estimator contract."""

    def __init__(self) -> None:
        self.classes_ = np.asarray([0, 1, 2])

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float)

    def predict(self, values: np.ndarray) -> np.ndarray:
        return self.predict_proba(values).argmax(axis=1)

    def __sklearn_is_fitted__(self) -> bool:
        return True


class ConformalSummary(StrictModel):
    confidence_level: float
    empirical_coverage: float
    average_set_size: float
    singleton_rate: float
    ambiguous_set_rate: float


class ApsConformalAbstainer:
    """MAPIE APS prediction sets used only as an abstention/policy signal."""

    def __init__(self, confidence_level: float = 0.90, seed: int = 42) -> None:
        self.confidence_level = confidence_level
        self.model = SplitConformalClassifier(
            estimator=_ProbabilityEstimator(),
            confidence_level=confidence_level,
            conformity_score="aps",
            prefit=True,
            random_state=seed,
        )

    def conformalize(
        self, cross_fitted_probabilities: np.ndarray, labels: np.ndarray
    ) -> ApsConformalAbstainer:
        self.model.conformalize(cross_fitted_probabilities, labels)
        return self

    def prediction_sets(self, probabilities: np.ndarray) -> np.ndarray:
        _, sets = self.model.predict_set(
            probabilities, conformity_score_params={"include_last_label": True}
        )
        return np.asarray(sets[:, :, 0], dtype=bool)

    def evaluate(self, probabilities: np.ndarray, labels: np.ndarray) -> ConformalSummary:
        sets = self.prediction_sets(probabilities)
        coverage = float(np.mean(sets[np.arange(len(labels)), labels]))
        sizes = sets.sum(axis=1)
        return ConformalSummary(
            confidence_level=self.confidence_level,
            empirical_coverage=round(coverage, 8),
            average_set_size=round(float(sizes.mean()), 8),
            singleton_rate=round(float(np.mean(sizes == 1)), 8),
            ambiguous_set_rate=round(float(np.mean(sizes > 1)), 8),
        )

    @staticmethod
    def policy_decision(prediction_set: np.ndarray, novelty_override: bool) -> HoldDecision:
        labels = set(np.flatnonzero(prediction_set))
        if novelty_override or not labels or labels == {0, 1, 2}:
            return HoldDecision.ESCALATE
        if labels == {0}:
            return HoldDecision.RELEASE
        if labels == {2}:
            return HoldDecision.ESCALATE
        if labels == {1} or labels == {0, 1}:
            return HoldDecision.EVIDENCE_NEEDED
        return HoldDecision.ESCALATE
