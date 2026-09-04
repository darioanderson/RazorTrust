from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from .audit import AuditLedger
from .domain import FeatureVector, HoldCase, HoldDecision, PolicyCheck, StrictModel
from .ground_truth import (
    ProviderGroundTruthStore,
    ProviderGroundTruthUnavailable,
)
from .live_features import FeatureContractPreview
from .policy import HoldPolicy, PolicyUnavailable
from .risk import DEFAULT_THRESHOLDS, CostPolicy
from .shadow_runtime import ShadowAnalysisRuntime


class LayerStage(StrictModel):
    layer: str
    status: str
    output: dict[str, Any] | None = None
    blockers: list[str] = []


class LayerExecutionResponse(StrictModel):
    trace_id: UUID
    hold_id: UUID
    account_id: str
    as_of: str
    source_mode: str
    stages: list[LayerStage]
    ai_recommendation: HoldDecision | None
    enforcement_mode: str
    human_authorization_required: bool = True
    automatic_release_enabled: bool = False
    production_action_eligible: bool = False


class LayerExecutionContext(StrictModel):
    source_mode: str = "RAZORPAY_STORED_POINT_IN_TIME"
    input_provenance: dict[str, Any] = {}
    secure_ingestion_status: str = "UPSTREAM"
    secure_ingestion_output: dict[str, Any] | None = None
    ground_truth_stage: LayerStage | None = None


class LayerExecutionService:
    """Execute the real point-in-time analysis stack for one existing hold."""

    def __init__(
        self,
        *,
        analysis_runtime: ShadowAnalysisRuntime,
        cost_policy: CostPolicy,
        hold_policy: HoldPolicy,
        enforcement_mode: str,
        ledger: AuditLedger,
        ground_truth_store: ProviderGroundTruthStore | None = None,
    ) -> None:
        self.analysis_runtime = analysis_runtime
        self.cost_policy = cost_policy
        self.hold_policy = hold_policy
        self.enforcement_mode = enforcement_mode
        self.ledger = ledger
        self.ground_truth_store = ground_truth_store

    async def execute(
        self,
        *,
        hold: HoldCase,
        preview: FeatureContractPreview,
        context: LayerExecutionContext | None = None,
    ) -> LayerExecutionResponse:
        trace_id = uuid4()
        execution_context = context or LayerExecutionContext()
        provenance = {
            "hold_id": str(hold.hold_id),
            "merchant_id": hold.merchant_id,
            "source_event_id": hold.source_event_id,
            "triggered_at": hold.triggered_at.isoformat(),
            "source_mode": execution_context.source_mode,
            **execution_context.input_provenance,
        }
        ingestion_output = execution_context.secure_ingestion_output or {
            "scope": (
                "validated webhook / reconciliation pipeline upstream "
                "of this executor"
            ),
            "per_case_signature_proof_exposed_here": False,
            "note": (
                "This executor consumes the authoritative stored timeline; "
                "it does not re-verify an already stored webhook body."
            ),
        }
        stages: list[LayerStage] = [
            LayerStage(
                layer="INPUT_PROVENANCE",
                status="RESULT",
                output=provenance,
            ),
            LayerStage(
                layer="SECURE_INGESTION",
                status=execution_context.secure_ingestion_status,
                output=ingestion_output,
            ),
            LayerStage(
                layer="AUTHORITATIVE_HISTORY",
                status="RESULT",
                output={
                    "baseline_transactions": preview.baseline_transactions,
                    "baseline_active_days": preview.baseline_active_days,
                    "current_transactions": preview.current_transactions,
                    "telemetry_coverage_baseline": preview.telemetry_coverage_baseline,
                    "telemetry_coverage_current": preview.telemetry_coverage_current,
                    "baseline_start": preview.baseline_start.isoformat(),
                    "baseline_end": preview.baseline_end.isoformat(),
                    "current_window_start": preview.current_window_start.isoformat(),
                },
            ),
        ]

        if preview.blockers or preview.feature_vector is None:
            stages.append(
                LayerStage(
                    layer="DATA_QUALITY_FIREWALL",
                    status="BLOCKED",
                    output={"model_score_allowed": False},
                    blockers=list(preview.blockers),
                )
            )
            stages.append(
                LayerStage(
                    layer="FEATURE_ENGINE_V2",
                    status="BLOCKED",
                    blockers=list(preview.blockers),
                )
            )
            stages.extend(self._research_only_stages())
            stages.append(
                LayerStage(
                    layer="CORE_ML",
                    status="BLOCKED",
                    output={"reason": "data quality firewall blocked model scoring"},
                )
            )
            stages.append(
                LayerStage(
                    layer="COST_AWARE_POLICY",
                    status="BLOCKED",
                    output={"reason": "no model probabilities available"},
                )
            )
            stages.append(
                LayerStage(
                    layer="OPA_GUARDRAILS",
                    status="BLOCKED",
                    output={"reason": "no model candidate was produced"},
                )
            )
            recommendation = HoldDecision.EVIDENCE_NEEDED
            stages.append(await self._ground_truth(hold, execution_context))
            stages.append(self._human_gate(recommendation))
            self._append_audit(
                hold=hold,
                trace_id=trace_id,
                status="BLOCKED_DATA_QUALITY",
                recommendation=recommendation,
                feature_hash=preview.feature_vector_sha256,
                model_version=None,
            )
            return self._response(
                trace_id,
                hold,
                preview,
                stages,
                recommendation,
                execution_context,
            )

        stages.append(
            LayerStage(
                layer="DATA_QUALITY_FIREWALL",
                status="PASS",
                output={"model_score_allowed": True, "blockers": []},
            )
        )
        stages.append(
            LayerStage(
                layer="FEATURE_ENGINE_V2",
                status="RESULT",
                output={
                    "feature_count": len(preview.feature_vector),
                    "feature_vector_sha256": preview.feature_vector_sha256,
                    "features": preview.feature_vector,
                    "production_action_eligible": False,
                },
            )
        )

        if not self.analysis_runtime.available:
            stages.append(
                LayerStage(
                    layer="CORE_ML",
                    status="ERROR",
                    output={
                        "expected_model_version": self.analysis_runtime.expected_model_version,
                        "runtime_status": self.analysis_runtime.status,
                        "error": self.analysis_runtime.error,
                    },
                )
            )
            stages.extend(self._research_only_stages())
            stages.append(
                LayerStage(
                    layer="OPA_GUARDRAILS",
                    status="BLOCKED",
                    output={"reason": "signed shadow analysis runtime unavailable"},
                )
            )
            recommendation = HoldDecision.ESCALATE
            stages.append(await self._ground_truth(hold, execution_context))
            stages.append(self._human_gate(recommendation))
            self._append_audit(
                hold=hold,
                trace_id=trace_id,
                status="MODEL_UNAVAILABLE",
                recommendation=recommendation,
                feature_hash=preview.feature_vector_sha256,
                model_version=self.analysis_runtime.model_version,
            )
            return self._response(
                trace_id,
                hold,
                preview,
                stages,
                recommendation,
                execution_context,
            )

        vector = FeatureVector.model_validate(preview.feature_vector)
        try:
            prediction = self.analysis_runtime.model.predict(vector, cohort="pooled")
        except (RuntimeError, ValueError) as exc:
            stages.append(
                LayerStage(
                    layer="CORE_ML",
                    status="ERROR",
                    output={"error_type": type(exc).__name__, "error": str(exc)},
                )
            )
            stages.extend(self._research_only_stages())
            stages.append(
                LayerStage(
                    layer="OPA_GUARDRAILS",
                    status="BLOCKED",
                    output={"reason": "model scoring failed"},
                )
            )
            recommendation = HoldDecision.ESCALATE
            stages.append(await self._ground_truth(hold, execution_context))
            stages.append(self._human_gate(recommendation))
            self._append_audit(
                hold=hold,
                trace_id=trace_id,
                status="MODEL_ERROR",
                recommendation=recommendation,
                feature_hash=preview.feature_vector_sha256,
                model_version=self.analysis_runtime.model_version,
            )
            return self._response(
                trace_id,
                hold,
                preview,
                stages,
                recommendation,
                execution_context,
            )

        stages.append(
            LayerStage(
                layer="CORE_ML",
                status="RESULT",
                output={
                    "model_version": prediction.model_version,
                    "calibrated": prediction.calibrated,
                    "calibration_method": prediction.calibration_method,
                    "probabilities": prediction.probabilities.model_dump(mode="json"),
                    "analysis_only": True,
                    "production_action_eligible": False,
                },
            )
        )
        stages.append(
            LayerStage(
                layer="NOVELTY_ISOLATION_FOREST",
                status="RESULT",
                output={
                    "anomaly_score": prediction.anomaly_score,
                    "anomaly_raw_score": prediction.anomaly_raw_score,
                    "anomaly_reference_max": prediction.anomaly_reference_max,
                    "anomaly_tail_excess": prediction.anomaly_tail_excess,
                    "anomaly_reference_size": prediction.anomaly_reference_size,
                    "anomaly_model_version": prediction.anomaly_model_version,
                    "anomaly_reference_mode": prediction.anomaly_reference_mode,
                    "novelty_override": prediction.novelty_override,
                },
            )
        )
        stages.extend(self._research_only_stages())
        top_features = [item.model_dump(mode="json") for item in prediction.top_features]
        stages.append(
            LayerStage(
                layer="SHAP_EXPLANATION",
                status="RESULT" if top_features else "NOT_AVAILABLE",
                output={
                    "top_features": top_features,
                    "interpretation": "model attribution only; not causal policy reasoning",
                },
            )
        )

        thresholds = getattr(self.analysis_runtime.model, "thresholds", DEFAULT_THRESHOLDS)
        candidate, expected_cost = self.cost_policy.choose(
            prediction,
            hold.evidence_round,
            thresholds,
        )
        stages.append(
            LayerStage(
                layer="COST_AWARE_POLICY",
                status="RESULT",
                output={
                    "candidate_decision": candidate.value,
                    "expected_cost_units": expected_cost,
                    "thresholds": thresholds.model_dump(mode="json"),
                },
            )
        )

        metadata_policy_version = self.analysis_runtime.metadata.get("policy_version")
        if not isinstance(metadata_policy_version, str) or not metadata_policy_version:
            recommendation = HoldDecision.ESCALATE
            stages.append(
                LayerStage(
                    layer="OPA_GUARDRAILS",
                    status="ERROR",
                    output={
                        "reason": "verified signed release did not expose policy_version",
                        "fail_closed_recommendation": HoldDecision.ESCALATE.value,
                    },
                )
            )
        else:
            policy_request = PolicyCheck(
                hold_id=hold.hold_id,
                candidate_decision=candidate,
                probabilities=prediction.probabilities,
                thresholds=thresholds,
                novelty_override=prediction.novelty_override,
                evidence_round=hold.evidence_round,
                model_version=prediction.model_version,
                policy_version=metadata_policy_version,
                system_error=False,
            )
            try:
                policy_result = await self.hold_policy.check(policy_request)
                recommendation = policy_result.allowed_decision
                stages.append(
                    LayerStage(
                        layer="OPA_GUARDRAILS",
                        status="RESULT",
                        output=policy_result.model_dump(mode="json"),
                    )
                )
            except PolicyUnavailable as exc:
                recommendation = HoldDecision.ESCALATE
                stages.append(
                    LayerStage(
                        layer="OPA_GUARDRAILS",
                        status="ERROR",
                        output={
                            "error_type": type(exc).__name__,
                            "fail_closed_recommendation": HoldDecision.ESCALATE.value,
                        },
                    )
                )

        stages.append(await self._ground_truth(hold, execution_context))
        stages.append(self._human_gate(recommendation))
        self._append_audit(
            hold=hold,
            trace_id=trace_id,
            status="COMPLETED",
            recommendation=recommendation,
            feature_hash=preview.feature_vector_sha256,
            model_version=prediction.model_version,
        )
        stages.append(
            LayerStage(
                layer="HASH_CHAINED_AUDIT",
                status="RECORDED",
                output={"trace_id": str(trace_id)},
            )
        )
        return self._response(
                trace_id,
                hold,
                preview,
                stages,
                recommendation,
                execution_context,
            )

    async def _ground_truth(
        self,
        hold: HoldCase,
        context: LayerExecutionContext,
    ) -> LayerStage:
        if context.ground_truth_stage is not None:
            return context.ground_truth_stage
        if self.ground_truth_store is None:
            return LayerStage(
                layer="PROVIDER_GROUND_TRUTH",
                status="NOT_AVAILABLE",
                output={
                    "evaluation_only": True,
                    "used_for_model_scoring": False,
                    "used_for_policy": False,
                    "reason": "provider ground-truth store is not configured",
                },
            )
        try:
            truth = await self.ground_truth_store.for_hold(hold)
        except ProviderGroundTruthUnavailable as exc:
            return LayerStage(
                layer="PROVIDER_GROUND_TRUTH",
                status="ERROR",
                output={
                    "evaluation_only": True,
                    "used_for_model_scoring": False,
                    "used_for_policy": False,
                    "error": str(exc),
                },
            )

        status = "RESULT" if truth.case_link_status == "LINKED" else "NOT_LINKED"
        return LayerStage(
            layer="PROVIDER_GROUND_TRUTH",
            status=status,
            output=truth.model_dump(mode="json"),
        )

    @staticmethod
    def _research_only_stages() -> list[LayerStage]:
        return [
            LayerStage(
                layer="DENOISING_AE_MAHALANOBIS_OOD",
                status="RESEARCH_ONLY",
                output={
                    "active_serving_artifact": False,
                    "result": None,
                    "note": "Phase-2 research capability is not promoted into this serving path.",
                },
            ),
            LayerStage(
                layer="CONFORMAL_UNCERTAINTY",
                status="RESEARCH_ONLY",
                output={
                    "active_serving_artifact": False,
                    "result": None,
                    "research_status": "PHASE6_V11_RETAIN_CONFORMAL_RESEARCH_VALUE",
                },
            ),
        ]

    def _human_gate(self, recommendation: HoldDecision) -> LayerStage:
        return LayerStage(
            layer="HUMAN_ONLY",
            status="WAITING_FOR_HUMAN",
            output={
                "ai_recommendation": recommendation.value,
                "enforcement_mode": self.enforcement_mode,
                "human_authorization_required": True,
                "automatic_release_enabled": False,
                "production_action_eligible": False,
            },
        )

    def _append_audit(
        self,
        *,
        hold: HoldCase,
        trace_id: UUID,
        status: str,
        recommendation: HoldDecision,
        feature_hash: str | None,
        model_version: str | None,
    ) -> None:
        self.ledger.append(
            case_id=hold.hold_id,
            actor_id="layer-execution-core",
            event_type="LAYER_EXECUTION",
            payload={
                "trace_id": str(trace_id),
                "status": status,
                "feature_vector_sha256": feature_hash,
                "analysis_model_version": model_version,
                "ai_recommendation": recommendation.value,
                "enforcement_mode": self.enforcement_mode,
                "automatic_release_enabled": False,
            },
        )

    def _response(
        self,
        trace_id: UUID,
        hold: HoldCase,
        preview: FeatureContractPreview,
        stages: list[LayerStage],
        recommendation: HoldDecision | None,
        context: LayerExecutionContext,
    ) -> LayerExecutionResponse:
        return LayerExecutionResponse(
            trace_id=trace_id,
            hold_id=hold.hold_id,
            account_id=hold.merchant_id,
            as_of=preview.as_of.isoformat(),
            source_mode=context.source_mode,
            stages=stages,
            ai_recommendation=recommendation,
            enforcement_mode=self.enforcement_mode,
        )
