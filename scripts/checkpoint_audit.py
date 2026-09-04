from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from razortrust.audit import AuditLedger
from razortrust.audit_checkpoint import create_audit_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a signed audit-chain checkpoint")
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--ledger-id", required=True)
    parser.add_argument("--key-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    private_key = os.getenv("RAZORTRUST_AUDIT_CHECKPOINT_PRIVATE_KEY")
    if not private_key:
        raise ValueError("RAZORTRUST_AUDIT_CHECKPOINT_PRIVATE_KEY is required")
    ledger = AuditLedger(args.ledger)
    valid, count = ledger.verify()
    if not valid:
        raise ValueError(f"audit chain is invalid at record {count}")
    checkpoint = create_audit_checkpoint(
        ledger_id=args.ledger_id,
        record_count=count,
        head_hash=ledger.head_hash,
        key_id=args.key_id,
        private_key_b64=private_key,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("audit checkpoints are immutable")
    args.output.write_text(
        json.dumps(checkpoint.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
