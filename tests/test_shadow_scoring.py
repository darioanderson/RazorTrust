from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from razortrust.domain import (
    FeatureContribution,
    FeatureVector,
    ModelPrediction,
    ProbabilityVector,
    Thresholds,
)
from razortrust.features import FEATURE_COLUMNS
from razortrust.live_features import FeatureContractPreview
from razortrust.risk import CostPolicy
from razortrust.shadow_scoring import ShadowScoringService, fixture_feature_vector


class MemoryStore:
    def __init__(self) -> None:
        self.rows = []

    async def persist(self, response):
        self.rows.append(response)
        return response


class FakeRiskModel:
    version = "xgb-if-settlement@test"
    thresholds = Thresholds(release=0.80, escalate=0.55)

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, features: FeatureVector, cohort: str = "pooled") -> ModelPrediction:
        self.calls += 1
        assert cohort == "pooled"
        assert tuple(features.model_dump()) == FEATURE_COLUMNS
        return ModelPrediction(
            probabilities=ProbabilityVector(release=0.15, evidence_needed=0.20, escalate=0.65),
            calibrated=True,
            calibration_method="sigmoid",
            anomaly_score=0.91,
            anomaly_raw_score=0.67,
            anomaly_reference_max=0.61,
            anomaly_tail_excess=0.06,
            anomaly_reference_size=240,
            anomaly_model_version="isolation-forest@test",
            anomaly_reference_mode="merchant_group_oof",
            novelty_override=False,
            top_features=[
                FeatureContribution(
                    feature="failed_auth_ratio",
                    observed_value=features.failed_auth_ratio,
                    contribution_value=0.42,
                    direction="toward_ESCALATE",
                    reference="test",
                    attribution_method="tree_shap",
                )
            ],
            model_version=self.version,
        )


def _preview(
    *, blockers: list[str], feature_vector: dict[str, float] | None
) -> FeatureContractPreview:
    as_of = datetime(2026, 9, 2, 12, tzinfo=UTC)
    return FeatureContractPreview(
        account_id="acc_test",
        as_of=as_of,
        current_window_start=as_of - timedelta(hours=24),
        baseline_start=as_of - timedelta(days=31),
        baseline_end=as_of - timedelta(days=1),
        baseline_transactions=35,
        baseline_active_days=10,
        current_transactions=3,
        telemetry_coverage_baseline=1.0,
        telemetry_coverage_current=1.0,
        blockers=blockers,
        feature_vector=feature_vector,
        feature_vector_sha256=("a" * 64 if feature_vector is not None else None),
        shadow_score_eligible=feature_vector is not None and not blockers,
        production_action_eligible=False,
    )


@pytest.mark.asyncio
async def test_live_shadow_score_fails_closed_without_feature_contract() -> None:
    store = MemoryStore()
    model = FakeRiskModel()
    service = ShadowScoringService(store, model, CostPolicy())
    result = await service.score_live_preview(
        _preview(blockers=["INSUFFICIENT_BASELINE_TRANSACTIONS"], feature_vector=None)
    )
    assert result.status == "BLOCKED"
    assert result.shadow_recommendation is None
    assert result.production_action_eligible is False
    assert "INSUFFICIENT_BASELINE_TRANSACTIONS" in result.blockers
    assert model.calls == 0
    assert len(store.rows) == 1


@pytest.mark.asyncio
async def test_complete_live_contract_scores_and_persists_provenance() -> None:
    store = MemoryStore()
    model = FakeRiskModel()
    service = ShadowScoringService(store, model, CostPolicy())
    vector = fixture_feature_vector("normal_baseline")
    result = await service.score_live_preview(
        _preview(blockers=[], feature_vector=vector.model_dump())
    )
    assert result.status == "SCORED"
    assert result.model_version == model.version
    assert result.calibration_method == "sigmoid"
    assert result.probabilities is not None
    assert result.anomaly_score == 0.91
    assert result.anomaly_raw_score == 0.67
    assert result.anomaly_reference_max == 0.61
    assert result.anomaly_tail_excess == 0.06
    assert result.anomaly_reference_size == 240
    assert result.anomaly_model_version == "isolation-forest@test"
    assert result.anomaly_reference_mode == "merchant_group_oof"
    assert result.shadow_recommendation is not None
    assert result.feature_provenance is not None
    assert result.feature_provenance["failed_auth_ratio"]["knowledge_time_policy"] == "OBSERVED_AT"
    assert (
        "razorpay_checkout_payment_attempts"
        in result.feature_provenance["failed_auth_ratio"]["source_tables"]
    )
    assert result.production_action_eligible is False
    assert model.calls == 1


@pytest.mark.asyncio
async def test_fixture_score_is_explicitly_mechanics_only() -> None:
    store = MemoryStore()
    model = FakeRiskModel()
    service = ShadowScoringService(store, model, CostPolicy())
    result = await service.score_fixture(
        fixture_name="novel_risk_burst",
        as_of=datetime(2026, 9, 2, 12, tzinfo=UTC),
    )
    assert result.status == "SCORED"
    assert result.source_mode == "SYNTHETIC_FIXTURE_MECHANICS_ONLY"
    assert result.account_id == "fixture:novel_risk_burst"
    assert tuple(result.feature_vector or {}) == FEATURE_COLUMNS
    assert result.feature_vector_sha256 is not None
    assert result.production_action_eligible is False


def test_fixture_contract_is_exact_locked_feature_order() -> None:
    for fixture in ("normal_baseline", "novel_risk_burst"):
        vector = fixture_feature_vector(fixture)
        assert tuple(vector.model_dump()) == FEATURE_COLUMNS
    with pytest.raises(ValueError):
        fixture_feature_vector("not-a-real-fixture")
