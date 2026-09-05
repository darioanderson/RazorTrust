from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from ..domain import StrictModel, TransactionEvent
from ..synthetic import SyntheticHold, SyntheticMerchant

DATA_FILES = ("merchants.jsonl", "transactions.jsonl", "holds.jsonl")


class RealSettlementDatasetManifest(StrictModel):
    schema_version: str = "1.0"
    dataset_id: str = Field(min_length=1, max_length=128)
    data_origin: Literal["PUBLIC_REAL", "PARTNER_REAL"]
    license_or_agreement: str = Field(min_length=1, max_length=500)
    permitted_purpose: str = Field(min_length=1, max_length=500)
    merchant_count: int = Field(ge=1)
    transaction_count: int = Field(ge=1)
    hold_count: int = Field(ge=1)
    file_sha256: dict[str, str]
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def settlement_dataset_content_sha256(directory: str | Path) -> str:
    """Hash the exact named input bytes using the synthetic dataset convention."""
    root = Path(directory)
    digest = hashlib.sha256()
    for filename in DATA_FILES:
        content = (root / filename).read_bytes()
        digest.update(filename.encode("utf-8") + b"\0" + content)
    return digest.hexdigest()


def load_real_settlement_dataset(
    directory: str | Path,
) -> tuple[
    list[SyntheticMerchant],
    list[TransactionEvent],
    list[SyntheticHold],
    RealSettlementDatasetManifest,
]:
    """Load a consented/public settlement dataset only after complete hash validation."""
    root = Path(directory)
    manifest = RealSettlementDatasetManifest.model_validate_json(
        (root / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    if set(manifest.file_sha256) != set(DATA_FILES):
        raise ValueError("real dataset manifest must hash exactly the required data files")

    for filename in DATA_FILES:
        actual = hashlib.sha256((root / filename).read_bytes()).hexdigest()
        if manifest.file_sha256.get(filename) != actual:
            raise ValueError(f"real dataset hash mismatch: {filename}")

    actual_content_hash = settlement_dataset_content_sha256(root)
    if manifest.content_sha256 != actual_content_hash:
        raise ValueError("real dataset aggregate content hash mismatch")

    merchants = _load_jsonl(root / "merchants.jsonl", SyntheticMerchant)
    transactions = _load_jsonl(root / "transactions.jsonl", TransactionEvent)
    holds = _load_jsonl(root / "holds.jsonl", SyntheticHold)
    if (len(merchants), len(transactions), len(holds)) != (
        manifest.merchant_count,
        manifest.transaction_count,
        manifest.hold_count,
    ):
        raise ValueError("real dataset manifest counts do not match the files")

    merchant_ids = [merchant.merchant_id for merchant in merchants]
    if len(merchant_ids) != len(set(merchant_ids)):
        raise ValueError("real dataset contains duplicate merchant IDs")
    known_merchants = set(merchant_ids)
    if any(transaction.merchant_id not in known_merchants for transaction in transactions):
        raise ValueError("real dataset transaction references an unknown merchant")
    if any(hold.hold.merchant_id not in known_merchants for hold in holds):
        raise ValueError("real dataset hold references an unknown merchant")

    return merchants, transactions, holds, manifest


def _load_jsonl[ModelT: StrictModel](path: Path, model: type[ModelT]) -> list[ModelT]:
    rows: list[ModelT] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(model.model_validate(json.loads(line)))
            except (ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid {path.name} row {line_number}") from exc
    return rows
