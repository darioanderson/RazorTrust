from __future__ import annotations

import hashlib
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import rfc8785

GENESIS_HASH = "0" * 64


class AuditFormatError(ValueError):
    pass


def canonical_json(value: dict[str, Any]) -> bytes:
    """Serialize with RFC 8785 JSON Canonicalization Scheme for cross-language hashing."""
    return rfc8785.dumps(value)


def hash_event(event: dict[str, Any], previous_hash: str) -> str:
    if len(previous_hash) != 64:
        raise ValueError("previous_hash must be a SHA-256 hex digest")
    payload = {**event, "previous_hash": previous_hash}
    return hashlib.sha256(bytes.fromhex(previous_hash) + canonical_json(payload)).hexdigest()


class AuditLedger:
    """Append-only JSONL chain used by the prototype and local tests."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        required = {"schema_version", "case_id", "sequence_no", "previous_hash", "record_hash"}
        for index, record in enumerate(records):
            missing = required.difference(record)
            if missing:
                names = ", ".join(sorted(missing))
                raise AuditFormatError(f"audit record {index} is missing: {names}")
        return records

    @property
    def head_hash(self) -> str:
        records = self._records()
        return records[-1]["record_hash"] if records else GENESIS_HASH

    def append(
        self,
        *,
        case_id: UUID,
        actor_id: str,
        event_type: str,
        payload: dict[str, Any],
        actor_type: str = "SYSTEM",
        input_refs: list[str] | None = None,
        model_version: str | None = None,
        feature_schema_version: str | None = None,
        policy_version: str | None = None,
    ) -> str:
        with self._lock:
            records = self._records()
            previous_hash = records[-1]["record_hash"] if records else GENESIS_HASH
            sequence_no = 1 + sum(record["case_id"] == str(case_id) for record in records)
            event = {
                "schema_version": "1.0",
                "event_id": f"evt_{uuid4().hex}",
                "case_id": str(case_id),
                "sequence_no": sequence_no,
                "actor": {"type": actor_type, "id": actor_id},
                "event_type": event_type,
                "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "model_version": model_version,
                "feature_schema_version": feature_schema_version,
                "policy_version": policy_version,
                "input_refs": input_refs or [],
                "payload": payload,
            }
            record_hash = hash_event(event, previous_hash)
            record = {**event, "previous_hash": previous_hash, "record_hash": record_hash}
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
            return record_hash

    def records_for_case(self, case_id: UUID) -> list[dict[str, Any]]:
        return [record for record in self._records() if record["case_id"] == str(case_id)]

    def verify(self) -> tuple[bool, int]:
        previous_hash = GENESIS_HASH
        records = self._records()
        for index, record in enumerate(records):
            if record.get("previous_hash") != previous_hash:
                return False, index
            unsigned = {
                key: value
                for key, value in record.items()
                if key not in {"record_hash", "previous_hash"}
            }
            expected = hash_event(unsigned, previous_hash)
            if record.get("record_hash") != expected:
                return False, index
            previous_hash = expected
        return True, len(records)
