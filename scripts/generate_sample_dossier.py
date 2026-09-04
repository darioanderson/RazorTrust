from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

from razortrust.domain import (
    DecisionReason,
    DecisionResponse,
    FeatureContribution,
    HoldCase,
    HoldDecision,
    HoldState,
    ProbabilityVector,
    Thresholds,
)
from razortrust.dossier import build_evidence_dossier


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a representative Buildathon dossier")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    now = datetime(2026, 9, 3, 9, 30, tzinfo=UTC)
    hold_id = UUID("4a3c1dd4-318c-40f6-9283-a36cb27b5f11")
    hold = HoldCase(
        hold_id=hold_id,
        request_id=UUID("76f0a24a-9903-4e6a-a62c-6cdd13eb1334"),
        merchant_id="merchant_travel_demo_014",
        source_event_id="settlement_setl_demo_9042",
        triggered_at=now - timedelta(hours=2),
        reason_code="AUTH_INCIDENT",
        state=HoldState.HUMAN_REVIEW,
        evidence_round=1,
        created_at=now - timedelta(hours=2),
        updated_at=now,
    )
    decision = DecisionResponse(
        hold_id=hold_id,
        trace_id="af5b7c28a43d41c4b9c1f11a85c289e7",
        decision=HoldDecision.ESCALATE,
        probabilities=ProbabilityVector(
            release=0.047,
            evidence_needed=0.191,
            escalate=0.762,
        ),
        calibrated=True,
        calibration_method="multinomial_logit",
        anomaly_score=0.93,
        expected_cost_units=5.0,
        cost_matrix_version="hold-cost@2",
        cost_matrix_sha256="8d2d3d9a4a82a8a585ba8e2140a3c130b806847e7345756858059b723f430331",
        thresholds=Thresholds(release=0.78, escalate=0.35),
        top_features=[
            FeatureContribution(
                feature="failed_auth_ratio",
                observed_value=0.31,
                contribution_value=1.42,
                direction="toward_ESCALATE",
                reference="merchant baseline",
                attribution_method="tree_shap",
                model_output_explained=HoldDecision.ESCALATE,
            ),
            FeatureContribution(
                feature="new_device_ratio",
                observed_value=0.64,
                contribution_value=0.91,
                direction="toward_ESCALATE",
                reference="merchant baseline",
                attribution_method="tree_shap",
                model_output_explained=HoldDecision.ESCALATE,
            ),
            FeatureContribution(
                feature="volume_delta_z",
                observed_value=0.18,
                contribution_value=-0.36,
                direction="toward_RELEASE",
                reference="merchant baseline",
                attribution_method="tree_shap",
                model_output_explained=HoldDecision.ESCALATE,
            ),
        ],
        model_version="xgb-settlement-v3A-research",
        feature_schema_version="1.0.0",
        policy_version="hold-policy@1",
        evidence_round=1,
        novelty_override=True,
        reason_code=DecisionReason.NOVELTY_DISAGREEMENT,
        audit_head_hash="93d6f4f4053e9d72c0d27b3f38bdff0371e77bd51df52aeb0dcfd5a2a5be9e46",
        created_at=now,
    )
    records = [
        {
            "sequence_no": 1,
            "event_type": "HOLD_CREATED",
            "timestamp": (now - timedelta(hours=2)).isoformat(),
            "record_hash": "0eddb2ce7e6fef7d2811e93b055dc4babcd6b4059b0b1e2d8d0105ee2734017a",
            "actor": {"type": "SYSTEM", "id": "hold-intake"},
        },
        {
            "sequence_no": 2,
            "event_type": "EVIDENCE_SUBMITTED",
            "timestamp": (now - timedelta(minutes=42)).isoformat(),
            "record_hash": "8daf5028910cf241d638424098dcaa2747abb2091814856ad71ddcf01d48a35c",
            "actor": {"type": "AI", "id": "evidence-service"},
            "payload": {
                "content_sha256": "bc21bfc7330c0e2af68a2943978000ca700985c96ea60c6f0f3930476994070a"
            },
        },
        {
            "sequence_no": 3,
            "event_type": "RISK_DECISION",
            "timestamp": now.isoformat(),
            "record_hash": decision.audit_head_hash,
            "actor": {"type": "AI", "id": "risk-decision-service"},
        },
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(build_evidence_dossier(hold, decision, records))
    print(args.output)


if __name__ == "__main__":
    main()
