from __future__ import annotations

from razortrust.conclusion import build_decision_conclusion
from razortrust.layer_execution import LayerStage


def test_conclusion_uses_only_returned_stage_facts() -> None:
    stages = [
        LayerStage(
            layer="DATA_QUALITY_FIREWALL",
            status="PASS",
            output={"model_score_allowed": True},
        ),
        LayerStage(
            layer="CORE_ML",
            status="RESULT",
            output={
                "calibrated": True,
                "probabilities": {
                    "release": 0.1,
                    "evidence_needed": 0.2,
                    "escalate": 0.7,
                },
            },
        ),
        LayerStage(
            layer="PROVIDER_GROUND_TRUTH",
            status="NOT_LINKED",
            output={"used_for_model_scoring": False},
        ),
    ]

    result = build_decision_conclusion(
        stages=stages,
        recommendation="ESCALATE",
        source_mode="RAZORPAY_STORED_POINT_IN_TIME",
    )

    assert "escalate at 70.0%" in result.model_statement
    assert "must not be described as confirmed fraud" in result.dispute_statement
    assert result.journey == [
        "DATA_QUALITY_FIREWALL: PASS",
        "CORE_ML: RESULT",
        "PROVIDER_GROUND_TRUTH: NOT_LINKED",
    ]
    assert result.human_authorization_required is True
    assert result.automatic_release_enabled is False
    assert result.production_action_eligible is False
