from __future__ import annotations

from datetime import datetime

from pydantic import Field

from .domain import StrictModel, utc_now
from .security import sign_manifest, verify_manifest


class AuditCheckpoint(StrictModel):
    schema_version: str = "1.0"
    ledger_id: str = Field(min_length=1, max_length=128)
    record_count: int = Field(ge=0)
    head_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpointed_at: datetime
    key_id: str = Field(min_length=1, max_length=128)
    signature_b64: str

    def signing_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"signature_b64"})


def create_audit_checkpoint(
    *,
    ledger_id: str,
    record_count: int,
    head_hash: str,
    key_id: str,
    private_key_b64: str,
) -> AuditCheckpoint:
    unsigned = AuditCheckpoint(
        ledger_id=ledger_id,
        record_count=record_count,
        head_hash=head_hash,
        checkpointed_at=utc_now(),
        key_id=key_id,
        signature_b64="pending",
    )
    return unsigned.model_copy(
        update={"signature_b64": sign_manifest(unsigned.signing_payload(), private_key_b64)}
    )


def verify_audit_checkpoint(checkpoint: AuditCheckpoint, public_key_b64: str) -> bool:
    return verify_manifest(checkpoint.signing_payload(), checkpoint.signature_b64, public_key_b64)
