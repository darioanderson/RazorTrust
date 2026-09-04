from __future__ import annotations

import hashlib
import json

import pytest

from experimental.public_fraud.partner_data import load_real_settlement_dataset
from razortrust.synthetic import generate_dataset


def test_real_settlement_loader_requires_manifest_hashes_and_counts(tmp_path) -> None:
    merchants, transactions, holds = generate_dataset(
        seed=3, merchants_per_family=1, transactions_per_merchant=12
    )
    payloads = {
        "merchants.jsonl": merchants,
        "transactions.jsonl": transactions,
        "holds.jsonl": holds,
    }
    hashes = {}
    for filename, rows in payloads.items():
        content = "".join(
            json.dumps(row.model_dump(mode="json"), sort_keys=True) + "\n" for row in rows
        )
        (tmp_path / filename).write_text(content, encoding="utf-8")
        hashes[filename] = hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()
    manifest = {
        "schema_version": "1.0",
        "dataset_id": "partner-redacted-2026-08",
        "data_origin": "PARTNER_REAL",
        "license_or_agreement": "DPA-2026-08",
        "permitted_purpose": "Settlement risk model validation",
        "merchant_count": len(merchants),
        "transaction_count": len(transactions),
        "hold_count": len(holds),
        "file_sha256": hashes,
        "content_sha256": "a" * 64,
    }
    (tmp_path / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    loaded_merchants, loaded_transactions, loaded_holds, loaded_manifest = (
        load_real_settlement_dataset(tmp_path)
    )

    assert len(loaded_merchants) == len(merchants)
    assert len(loaded_transactions) == len(transactions)
    assert len(loaded_holds) == len(holds)
    assert loaded_manifest.data_origin == "PARTNER_REAL"

    (tmp_path / "holds.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_real_settlement_dataset(tmp_path)
