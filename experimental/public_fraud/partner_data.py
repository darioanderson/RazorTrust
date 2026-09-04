from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from razortrust.domain import StrictModel, TransactionEvent
from razortrust.synthetic import SyntheticHold, SyntheticMerchant


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


def load_real_settlement_dataset(
    directory: str | Path,
) -> tuple[
    list[SyntheticMerchant],
    list[TransactionEvent],
    list[SyntheticHold],
    RealSettlementDatasetManifest,
]:
    root = Path(directory)
    manifest = RealSettlementDatasetManifest.model_validate_json(
        (root / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    expected_files = ("merchants.jsonl", "transactions.jsonl", "holds.jsonl")
    for filename in expected_files:
        path = root / filename
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if manifest.file_sha256.get(filename) != actual:
            raise ValueError(f"real dataset hash mismatch: {filename}")
    merchants = _load_jsonl(root / "merchants.jsonl", SyntheticMerchant)
    transactions = _load_jsonl(root / "transactions.jsonl", TransactionEvent)
    holds = _load_jsonl(root / "holds.jsonl", SyntheticHold)
    if (len(merchants), len(transactions), len(holds)) != (
        manifest.merchant_count,
        manifest.transaction_count,
        manifest.hold_count,
    ):
        raise ValueError("real dataset manifest counts do not match the files")
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
