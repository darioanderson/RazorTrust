from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO
from uuid import uuid4

from pypdf import PdfReader

from razortrust.domain import (
    DecisionReason,
    DecisionResponse,
    HoldCase,
    HoldDecision,
    ProbabilityVector,
    Thresholds,
)
from razortrust.dossier import build_evidence_dossier


def test_evidence_dossier_is_a_readable_two_page_pdf() -> None:
    hold_id = uuid4()
    now = datetime.now(UTC)
    hold = HoldCase(
        hold_id=hold_id,
        request_id=uuid4(),
        merchant_id="merchant_demo_001",
        source_event_id="settlement_001",
        triggered_at=now,
        reason_code="AUTH_INCIDENT",
    )
    decision = DecisionResponse(
        hold_id=hold_id,
        trace_id="a" * 32,
        decision=HoldDecision.ESCALATE,
        probabilities=ProbabilityVector(release=0.05, evidence_needed=0.15, escalate=0.8),
        calibrated=True,
        calibration_method="sigmoid",
        anomaly_score=0.9,
        expected_cost_units=5.0,
        cost_matrix_version="test-costs",
        cost_matrix_sha256="b" * 64,
        thresholds=Thresholds(release=0.8, escalate=0.6),
        top_features=[],
        model_version="model-test",
        feature_schema_version="1.0.0",
        policy_version="policy-test",
        evidence_round=0,
        novelty_override=False,
        reason_code=DecisionReason.MODEL_POLICY,
        audit_head_hash="c" * 64,
        created_at=now,
    )
    records = [
        {
            "sequence_no": 1,
            "event_type": "RISK_DECISION",
            "timestamp": now.isoformat(),
            "record_hash": "c" * 64,
            "actor": {"type": "AI", "id": "risk-decision-service"},
        }
    ]
    content = build_evidence_dossier(hold, decision, records)
    assert content.startswith(b"%PDF")
    reader = PdfReader(BytesIO(content))
    assert len(reader.pages) >= 2
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "merchant_demo_001" in text
    assert "Attribution and signed audit timeline" in text
    assert "human reviewer" in text
