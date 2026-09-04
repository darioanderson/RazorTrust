from __future__ import annotations

from razortrust.audit_checkpoint import create_audit_checkpoint, verify_audit_checkpoint
from razortrust.security import generate_release_keypair


def test_signed_audit_checkpoint_detects_tampering() -> None:
    private_key, public_key = generate_release_keypair()
    checkpoint = create_audit_checkpoint(
        ledger_id="production-audit",
        record_count=10,
        head_hash="a" * 64,
        key_id="audit-checkpoint-2026-08",
        private_key_b64=private_key,
    )

    assert verify_audit_checkpoint(checkpoint, public_key)
    assert not verify_audit_checkpoint(
        checkpoint.model_copy(update={"record_count": 11}), public_key
    )
