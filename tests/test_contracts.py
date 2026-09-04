from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from razortrust.audit import canonical_json
from razortrust.domain import EvidenceSubmission, ProbabilityVector, utc_now
from razortrust.security import generate_release_keypair, sign_manifest, verify_manifest


def test_probability_vector_must_sum_to_one() -> None:
    with pytest.raises(ValidationError, match="sum to 1"):
        ProbabilityVector(release=0.8, evidence_needed=0.3, escalate=0.1)


def test_client_cannot_submit_server_derived_evidence_fields() -> None:
    now = utc_now()
    with pytest.raises(ValidationError, match="server-derived"):
        EvidenceSubmission(
            request_id=uuid4(),
            evidence_type="CAMPAIGN",
            submitted_at=now,
            evidence_observed_at=now - timedelta(hours=1),
            content_sha256="b" * 64,
            metadata={"evidence_type_match": True},
        )


def test_evidence_metadata_cannot_contain_document_content() -> None:
    now = utc_now()
    with pytest.raises(ValidationError, match="evidence content is not accepted"):
        EvidenceSubmission(
            request_id=uuid4(),
            evidence_type="CAMPAIGN",
            submitted_at=now,
            evidence_observed_at=now - timedelta(hours=1),
            content_sha256="b" * 64,
            metadata={"document": "raw invoice text"},
        )


def test_release_manifest_signature_detects_tampering() -> None:
    private_key, public_key = generate_release_keypair()
    manifest = {"model_version": "settlement-risk@12", "model_sha256": "c" * 64}
    signature = sign_manifest(manifest, private_key)
    assert verify_manifest(manifest, signature, public_key)
    assert not verify_manifest(
        {**manifest, "model_version": "settlement-risk@13"}, signature, public_key
    )


def test_canonical_json_uses_rfc8785_key_order_and_utf8() -> None:
    assert canonical_json({"z": 1, "é": "value", "a": True}) == (
        b'{"a":true,"z":1,"\xc3\xa9":"value"}'
    )
