from __future__ import annotations

from typing import Protocol

from .domain import EvidenceAssessment, EvidenceRecord, EvidenceVerdict, HoldCase
from .repository import expected_evidence_type
from .security import verify_manifest


class EvidenceVerifier(Protocol):
    @property
    def ready(self) -> bool: ...

    def assess(self, hold: HoldCase, evidence: EvidenceRecord) -> EvidenceAssessment: ...


class LegacyEvidenceVerifier:
    """Development-only compatibility verifier; never configure this in staging."""

    ready = True

    def assess(self, hold: HoldCase, evidence: EvidenceRecord) -> EvidenceAssessment:
        return _legacy_assessment(hold, evidence)


class SignedAttestationEvidenceVerifier:
    """Verify an external provider's Ed25519 attestation over immutable evidence facts."""

    version = "signed-evidence-attestation@1"

    def __init__(self, public_keys: dict[str, str]) -> None:
        self._keys = {
            key.strip(): value.strip() for key, value in public_keys.items() if key and value
        }

    @property
    def ready(self) -> bool:
        return bool(self._keys)

    def assess(self, hold: HoldCase, evidence: EvidenceRecord) -> EvidenceAssessment:
        metadata = evidence.submission.metadata
        key_id = str(metadata.get("attestation_key_id", ""))
        signature = str(metadata.get("attestation_signature", ""))
        subject = str(metadata.get("subject_merchant_id", ""))
        expected_type = expected_evidence_type(hold.reason_code)
        type_match = expected_type is None or evidence.submission.evidence_type == expected_type
        subject_match = subject == hold.merchant_id
        time_match = evidence.recency_hours <= 168
        payload = {
            "schema_version": "1.0",
            "content_sha256": evidence.submission.content_sha256,
            "evidence_type": evidence.submission.evidence_type,
            "subject_merchant_id": subject,
            "evidence_observed_at": evidence.submission.evidence_observed_at.isoformat(),
        }
        public_key = self._keys.get(key_id)
        signature_valid = bool(
            public_key and signature and verify_manifest(payload, signature, public_key)
        )
        if signature_valid and type_match and subject_match and time_match:
            verdict = EvidenceVerdict.SUPPORTED
        elif signature_valid and (not type_match or not subject_match):
            verdict = EvidenceVerdict.CONTRADICTORY
        else:
            verdict = EvidenceVerdict.INSUFFICIENT
        return EvidenceAssessment(
            verdict=verdict,
            expected_type=expected_type,
            submitted_type=evidence.submission.evidence_type,
            subject_match=subject_match,
            time_match=time_match,
            verification_mode="SIGNED_PROVIDER_ATTESTATION",
            content_sha256=evidence.submission.content_sha256,
            verifier_version=self.version,
            attestation_key_id=key_id or None,
        )


class UnavailableEvidenceVerifier:
    ready = False

    def assess(self, hold: HoldCase, evidence: EvidenceRecord) -> EvidenceAssessment:
        return EvidenceAssessment(
            verdict=EvidenceVerdict.INSUFFICIENT,
            expected_type=expected_evidence_type(hold.reason_code),
            submitted_type=evidence.submission.evidence_type,
            subject_match=False,
            time_match=evidence.recency_hours <= 168,
            verification_mode="UNVERIFIED",
            content_sha256=evidence.submission.content_sha256,
            verifier_version="evidence-verifier-unavailable@1",
        )


def assess_evidence(hold: HoldCase, evidence: EvidenceRecord) -> EvidenceAssessment:
    """Backward-compatible development entry point."""
    return _legacy_assessment(hold, evidence)


def _legacy_assessment(hold: HoldCase, evidence: EvidenceRecord) -> EvidenceAssessment:
    expected_type = expected_evidence_type(hold.reason_code)
    subject_match = evidence.type_match
    time_match = evidence.recency_hours <= 168
    if subject_match and time_match:
        verdict = EvidenceVerdict.SUPPORTED
    elif not subject_match:
        verdict = EvidenceVerdict.CONTRADICTORY
    else:
        verdict = EvidenceVerdict.INSUFFICIENT
    return EvidenceAssessment(
        verdict=verdict,
        expected_type=expected_type,
        submitted_type=evidence.submission.evidence_type,
        subject_match=subject_match,
        time_match=time_match,
        verification_mode="MOCKED",
        content_sha256=evidence.submission.content_sha256,
        verifier_version="evidence-rules@1-development-only",
    )
