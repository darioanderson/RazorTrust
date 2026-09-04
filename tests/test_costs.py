from __future__ import annotations

from razortrust.costs import DEFAULT_COST_MATRIX, load_cost_matrix
from razortrust.domain import (
    HoldDecision,
    ModelPrediction,
    ProbabilityVector,
    Thresholds,
)
from razortrust.risk import CostPolicy


def test_training_and_serving_share_one_versioned_cost_artifact() -> None:
    loaded = load_cost_matrix()
    assert loaded == DEFAULT_COST_MATRIX
    assert loaded.cost_matrix_version == "hold-cost@2"
    assert len(loaded.content_sha256) == 64

    prediction = ModelPrediction(
        probabilities=ProbabilityVector(release=0.2, evidence_needed=0.3, escalate=0.5),
        calibrated=True,
        calibration_method="test",
        anomaly_score=0.1,
        top_features=[],
        model_version="test@1",
    )
    policy = CostPolicy(loaded)
    candidate, expected_cost = policy.choose(
        prediction, evidence_round=0, thresholds=Thresholds(release=0.99, escalate=0.99)
    )
    assert candidate == HoldDecision.EVIDENCE_NEEDED
    assert expected_cost == policy.expected_cost(prediction, candidate)
