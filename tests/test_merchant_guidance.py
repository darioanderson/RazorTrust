from __future__ import annotations

from datetime import UTC, datetime

from razortrust.domain import FeatureContribution, HoldDecision
from razortrust.merchant_guidance import build_merchant_guidance


def test_evidence_decision_has_plain_language_reasons_documents_and_deadline() -> None:
    created_at = datetime(2026, 8, 29, 10, tzinfo=UTC)
    guidance = build_merchant_guidance(
        HoldDecision.EVIDENCE_NEEDED,
        [
            FeatureContribution(
                feature="gmv_delta_z",
                observed_value=5.0,
                contribution_value=0.8,
                direction="toward_EVIDENCE_NEEDED",
                reference="test",
                attribution_method="tree_shap",
            )
        ],
        created_at=created_at,
    )

    assert guidance is not None
    assert "gross transaction value" in guidance.reasons[0]
    assert guidance.required_evidence[0].title == "Campaign or sales-event proof"
    assert (guidance.submit_by - created_at).total_seconds() == 48 * 3600
    assert "human review" in guidance.next_step


def test_non_evidence_decisions_do_not_receive_an_evidence_checklist() -> None:
    assert (
        build_merchant_guidance(
            HoldDecision.ESCALATE,
            [],
            created_at=datetime(2026, 8, 29, tzinfo=UTC),
        )
        is None
    )
