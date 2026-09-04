from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

from razortrust.domain import EvidenceRecord, EvidenceSubmission, HoldCase, utc_now
from razortrust.evidence import SignedAttestationEvidenceVerifier
from razortrust.security import generate_release_keypair, sign_manifest


def test_signed_provider_attestation_is_verified_and_tamper_evident() -> None:
    private_key, public_key = generate_release_keypair()
    now = utc_now()
    hold = HoldCase(
        hold_id=uuid4(),
        request_id=uuid4(),
        merchant_id="merchant-42",
        source_event_id="settlement-42",
        triggered_at=now,
        reason_code="CAMPAIGN",
    )
    payload = {
        "schema_version": "1.0",
        "content_sha256": "a" * 64,
        "evidence_type": "CAMPAIGN",
        "subject_merchant_id": "merchant-42",
        "evidence_observed_at": (now - timedelta(hours=1)).isoformat(),
    }
    submission = EvidenceSubmission(
        request_id=uuid4(),
        evidence_type="CAMPAIGN",
        submitted_at=now,
        evidence_observed_at=now - timedelta(hours=1),
        content_sha256="a" * 64,
        metadata={
            "attestation_key_id": "provider-1",
            "attestation_signature": sign_manifest(payload, private_key),
            "subject_merchant_id": "merchant-42",
        },
    )
    record = EvidenceRecord(
        evidence_id=uuid4(),
        hold_id=hold.hold_id,
        submission=submission,
        recency_hours=1,
        type_match=True,
    )
    verifier = SignedAttestationEvidenceVerifier({"provider-1": public_key})
    assessment = verifier.assess(hold, record)
    assert assessment.verdict == "SUPPORTED"
    assert assessment.verification_mode == "SIGNED_PROVIDER_ATTESTATION"

    tampered = record.model_copy(
        update={"submission": submission.model_copy(update={"content_sha256": "b" * 64})}
    )
    assert verifier.assess(hold, tampered).verdict == "INSUFFICIENT"
