from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from razortrust.audit import AuditLedger
from razortrust.domain import (
    FeatureContribution,
    HoldCase,
    HoldDecision,
    ModelPrediction,
    PolicyResult,
    ProbabilityVector,
    Thresholds,
)
from razortrust.features import FEATURE_COLUMNS
from razortrust.layer_execution import LayerExecutionService
from razortrust.live_features import FeatureContractPreview
from razortrust.policy import PolicyUnavailable
from razortrust.risk import CostPolicy
from razortrust.shadow_runtime import ShadowAnalysisRuntime


class FakeModel:
    version = "xgb-if-settlement@2"
    thresholds = Thresholds(release=0.80, escalate=0.55)

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, features, cohort="pooled") -> ModelPrediction:
        self.calls += 1
        assert cohort == "pooled"
        assert tuple(features.model_dump()) == FEATURE_COLUMNS
        return ModelPrediction(
            probabilities=ProbabilityVector(release=0.10, evidence_needed=0.20, escalate=0.70),
            calibrated=True,
            calibration_method="sigmoid",
            anomaly_score=0.93,
            anomaly_raw_score=0.70,
            anomaly_reference_max=0.60,
            anomaly_tail_excess=0.10,
            anomaly_reference_size=250,
            anomaly_model_version="isolation-forest@2",
            anomaly_reference_mode="merchant_group_oof",
            novelty_override=False,
            top_features=[
                FeatureContribution(
                    feature="failed_auth_ratio",
                    observed_value=0.4,
                    contribution_value=0.5,
                    direction="toward_ESCALATE",
                    reference="test",
                    attribution_method="tree_shap",
                )
            ],
            model_version=self.version,
        )


class FakePolicy:
    async def check(self, request):
        assert request.model_version == "xgb-if-settlement@2"
        return PolicyResult(
            allowed_decision=HoldDecision.ESCALATE,
            guardrail_triggered=False,
            reasons=["TEST"],
            policy_version="hold-policy@test",
        )


class FailingPolicy:
    async def check(self, _request):
        raise PolicyUnavailable("OPA unavailable")


def _runtime(model=None) -> ShadowAnalysisRuntime:
    chosen = model or FakeModel()
    return ShadowAnalysisRuntime(
        model=chosen,
        status="READY",
        expected_model_version="xgb-if-settlement@2",
        model_version=chosen.version,
        release_path="/tmp/release",
        metadata={"model_version": "xgb-if-settlement@2", "policy_version": "hold-cost@2"},
    )


def _hold() -> HoldCase:
    now = datetime(2026, 9, 4, 12, tzinfo=UTC)
    return HoldCase(
        hold_id=uuid4(),
        request_id=uuid4(),
        merchant_id="merchant-real-001",
        source_event_id="rzp-event-001",
        triggered_at=now,
        reason_code="LIVE_CASE",
    )


def _preview(*, blocked: bool = False) -> FeatureContractPreview:
    now = datetime(2026, 9, 4, 12, tzinfo=UTC)
    features = {name: 0.0 for name in FEATURE_COLUMNS}
    features["failed_auth_ratio"] = 0.4
    return FeatureContractPreview(
        account_id="merchant-real-001",
        as_of=now,
        current_window_start=now - timedelta(hours=24),
        baseline_start=now - timedelta(days=31),
        baseline_end=now - timedelta(days=1),
        baseline_transactions=40,
        baseline_active_days=12,
        current_transactions=5,
        telemetry_coverage_baseline=1.0,
        telemetry_coverage_current=1.0,
        blockers=["CURRENT_WINDOW_TELEMETRY_INCOMPLETE"] if blocked else [],
        feature_vector=None if blocked else features,
        feature_vector_sha256=None if blocked else "a" * 64,
        shadow_score_eligible=not blocked,
        production_action_eligible=False,
    )


@pytest.mark.asyncio
async def test_data_quality_blocks_model_and_routes_to_human(tmp_path):
    model = FakeModel()
    service = LayerExecutionService(
        analysis_runtime=_runtime(model),
        cost_policy=CostPolicy(),
        hold_policy=FakePolicy(),
        enforcement_mode="human_only",
        ledger=AuditLedger(tmp_path / "audit.jsonl"),
    )
    result = await service.execute(hold=_hold(), preview=_preview(blocked=True))

    assert model.calls == 0
    assert result.ai_recommendation == HoldDecision.EVIDENCE_NEEDED
    assert result.automatic_release_enabled is False
    assert result.production_action_eligible is False
    by_layer = {stage.layer: stage for stage in result.stages}
    assert by_layer["DATA_QUALITY_FIREWALL"].status == "BLOCKED"
    assert by_layer["CORE_ML"].status == "BLOCKED"
    assert by_layer["HUMAN_ONLY"].status == "WAITING_FOR_HUMAN"


@pytest.mark.asyncio
async def test_real_feature_vector_flows_through_model_policy_and_human_gate(tmp_path):
    model = FakeModel()
    ledger = AuditLedger(tmp_path / "audit.jsonl")
    service = LayerExecutionService(
        analysis_runtime=_runtime(model),
        cost_policy=CostPolicy(),
        hold_policy=FakePolicy(),
        enforcement_mode="human_only",
        ledger=ledger,
    )
    hold = _hold()
    result = await service.execute(hold=hold, preview=_preview())

    assert model.calls == 1
    assert result.ai_recommendation == HoldDecision.ESCALATE
    assert result.enforcement_mode == "human_only"
    assert result.automatic_release_enabled is False
    by_layer = {stage.layer: stage for stage in result.stages}
    assert by_layer["FEATURE_ENGINE_V2"].output["feature_count"] == 13
    assert by_layer["CORE_ML"].output["model_version"] == "xgb-if-settlement@2"
    assert by_layer["NOVELTY_ISOLATION_FOREST"].output["anomaly_score"] == 0.93
    assert by_layer["DENOISING_AE_MAHALANOBIS_OOD"].status == "RESEARCH_ONLY"
    assert by_layer["CONFORMAL_UNCERTAINTY"].status == "RESEARCH_ONLY"
    attribution = by_layer["SHAP_EXPLANATION"].output["top_features"][0]["attribution_method"]
    assert attribution == "tree_shap"
    assert by_layer["OPA_GUARDRAILS"].status == "RESULT"
    assert by_layer["HUMAN_ONLY"].status == "WAITING_FOR_HUMAN"
    assert by_layer["HASH_CHAINED_AUDIT"].status == "RECORDED"
    assert ledger.verify()[0] is True


@pytest.mark.asyncio
async def test_opa_failure_fails_closed_to_escalate(tmp_path):
    service = LayerExecutionService(
        analysis_runtime=_runtime(),
        cost_policy=CostPolicy(),
        hold_policy=FailingPolicy(),
        enforcement_mode="human_only",
        ledger=AuditLedger(tmp_path / "audit.jsonl"),
    )
    result = await service.execute(hold=_hold(), preview=_preview())
    by_layer = {stage.layer: stage for stage in result.stages}

    assert result.ai_recommendation == HoldDecision.ESCALATE
    assert by_layer["OPA_GUARDRAILS"].status == "ERROR"
    assert by_layer["HUMAN_ONLY"].output["automatic_release_enabled"] is False


@pytest.mark.asyncio
async def test_unavailable_signed_runtime_never_scores_or_releases(tmp_path):
    from razortrust.risk import UnavailableRiskModel

    runtime = ShadowAnalysisRuntime(
        model=UnavailableRiskModel("missing"),
        status="ERROR",
        expected_model_version="xgb-if-settlement@2",
        model_version="unavailable-model@1",
        release_path=None,
        metadata={},
        error="missing",
    )
    service = LayerExecutionService(
        analysis_runtime=runtime,
        cost_policy=CostPolicy(),
        hold_policy=FakePolicy(),
        enforcement_mode="human_only",
        ledger=AuditLedger(tmp_path / "audit.jsonl"),
    )
    result = await service.execute(hold=_hold(), preview=_preview())
    by_layer = {stage.layer: stage for stage in result.stages}

    assert result.ai_recommendation == HoldDecision.ESCALATE
    assert by_layer["CORE_ML"].status == "ERROR"
    assert result.automatic_release_enabled is False
