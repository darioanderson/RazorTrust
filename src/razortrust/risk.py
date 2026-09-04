from __future__ import annotations

from typing import Protocol

from .costs import DEFAULT_COST_MATRIX, CostMatrixArtifact
from .domain import (
    FeatureContribution,
    FeatureVector,
    HoldDecision,
    ModelPrediction,
    ProbabilityVector,
    Thresholds,
)

DEVELOPMENT_MODEL_VERSION = "development-rules@1"
DEFAULT_THRESHOLDS = Thresholds(release=0.80, escalate=0.55)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class RiskModel(Protocol):
    version: str

    def predict(self, features: FeatureVector, cohort: str = "pooled") -> ModelPrediction: ...


class DevelopmentRiskModel:
    """Deterministic adapter used while the trained Tier 0 artifact is unavailable.

    It is intentionally labelled as uncalibrated rule output. Nothing produced by
    this class is represented as XGBoost, SHAP, or production model performance.
    """

    version = DEVELOPMENT_MODEL_VERSION
    thresholds = DEFAULT_THRESHOLDS

    def predict(self, features: FeatureVector, cohort: str = "pooled") -> ModelPrediction:
        contributions = {
            "volume_delta_z": 0.06 * _clamp(features.volume_delta_z / 5),
            "gmv_delta_z": 0.08 * _clamp(features.gmv_delta_z / 5),
            "ticket_size_delta_z": 0.06 * _clamp(features.ticket_size_delta_z / 5),
            "new_device_ratio": 0.12 * features.new_device_ratio,
            "new_geo_ratio": 0.08 * features.new_geo_ratio,
            "refund_rate_delta_z": 0.10 * _clamp(features.refund_rate_delta_z / 5),
            "chargeback_rate_delta_z": 0.14 * _clamp(features.chargeback_rate_delta_z / 5),
            "failed_auth_ratio": 0.14 * _clamp(features.failed_auth_ratio / 0.5),
            "volume_trend_slope": 0.05 * _clamp(features.volume_trend_slope / 3),
            "interarrival_time_cv": 0.04 * _clamp((0.8 - features.interarrival_time_cv) / 0.8),
            "device_entropy": 0.03 * _clamp(features.device_entropy / 4),
            "geo_entropy": 0.02 * _clamp(features.geo_entropy / 3),
            "amount_distribution_kl": 0.08 * _clamp(features.amount_distribution_kl / 2),
        }
        risk = _clamp(sum(contributions.values()))
        _ = cohort

        escalate = _clamp(0.05 + 0.80 * risk)
        release = _clamp(0.90 - 0.95 * risk)
        evidence_needed = max(0.0, 1.0 - release - escalate)
        total = release + evidence_needed + escalate
        probabilities = ProbabilityVector(
            release=release / total,
            evidence_needed=evidence_needed / total,
            escalate=escalate / total,
        )
        observed = features.model_dump()
        top_features = [
            FeatureContribution(
                feature=name,
                observed_value=float(observed[name]),
                contribution_value=round(value, 6),
                direction="toward_ESCALATE",
                reference="documented development rule weight",
                attribution_method="development_rules",
            )
            for name, value in sorted(
                contributions.items(), key=lambda item: item[1], reverse=True
            )[:3]
        ]
        return ModelPrediction(
            probabilities=probabilities,
            calibrated=False,
            calibration_method="none",
            anomaly_score=round(_clamp(risk * 1.1), 6),
            top_features=top_features,
            model_version=self.version,
        )


class HumanOnlyRiskModel:
    """Deterministic no-auto-release adapter for human-gated deployments.

    This is an explicit operating mode, not a fallback model. It never recommends
    RELEASE and never produces a production-eligible automated action.
    """

    version = "human-only@1"
    thresholds = DEFAULT_THRESHOLDS
    human_only = True

    def predict(self, features: FeatureVector, cohort: str = "pooled") -> ModelPrediction:
        del features, cohort
        return ModelPrediction(
            probabilities=ProbabilityVector(release=0.0, evidence_needed=0.0, escalate=1.0),
            calibrated=False,
            calibration_method="human_only",
            anomaly_score=1.0,
            novelty_override=False,
            top_features=[],
            model_version=self.version,
        )


class UnavailableRiskModel:
    """Fail-closed adapter used when a configured production artifact cannot load."""

    version = "unavailable-model@1"
    thresholds = DEFAULT_THRESHOLDS

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def predict(self, features: FeatureVector, cohort: str = "pooled") -> ModelPrediction:
        del features, cohort
        raise RuntimeError(f"configured model is unavailable: {self.reason}")


class CostPolicy:
    """Converts class probabilities into an operational candidate decision."""

    def __init__(self, cost_matrix: CostMatrixArtifact = DEFAULT_COST_MATRIX) -> None:
        self.cost_matrix = cost_matrix

    def choose(
        self,
        prediction: ModelPrediction,
        evidence_round: int,
        thresholds: Thresholds = DEFAULT_THRESHOLDS,
    ) -> tuple[HoldDecision, float]:
        probabilities = prediction.probabilities
        vector = (probabilities.release, probabilities.evidence_needed, probabilities.escalate)
        expected_costs = self.cost_matrix.expected_costs(vector)
        candidate = min(expected_costs.items(), key=lambda item: item[1])[0]

        if probabilities.release >= thresholds.release and not prediction.novelty_override:
            candidate = HoldDecision.RELEASE
        elif probabilities.escalate >= thresholds.escalate:
            candidate = HoldDecision.ESCALATE
        elif evidence_round == 0:
            candidate = HoldDecision.EVIDENCE_NEEDED
        else:
            candidate = HoldDecision.ESCALATE
        return candidate, round(expected_costs[candidate], 6)

    def expected_cost(self, prediction: ModelPrediction, action: HoldDecision) -> float:
        probabilities = prediction.probabilities
        costs = self.cost_matrix.expected_costs(
            (probabilities.release, probabilities.evidence_needed, probabilities.escalate)
        )
        return round(costs[action], 6)
