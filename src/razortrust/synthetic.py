from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from .domain import HoldCreate, HoldDecision, MerchantBaseline, StrictModel, TransactionEvent

GENERATOR_VERSION = "3.0.0"
DATASET_VERSION = "synthetic@3"
DATASET_CREATED_AT = datetime(2026, 8, 28, tzinfo=UTC)
LEGITIMATE_FAMILIES = (
    "baseline",
    "flash_sale",
    "influencer_campaign",
    "geo_expansion",
    "product_launch",
    "auth_provider_incident",
    "seasonal_peak",
)
RISK_FAMILIES = (
    "card_testing_style",
    "refund_abuse_style",
    "synthetic_volume_spike",
    "distributed_device_ring",
    "slow_low_ring",
    "account_takeover_shift",
    "mixed_evasion",
)
DIRECT_RELEASE_FAMILIES = ("baseline", "seasonal_peak")
EVIDENCE_RESOLVABLE_FAMILIES = tuple(
    family for family in LEGITIMATE_FAMILIES if family not in DIRECT_RELEASE_FAMILIES
)


class TrueRiskState(StrEnum):
    LEGITIMATE = "LEGITIMATE"
    RISKY = "RISKY"


class SyntheticMerchant(StrictModel):
    merchant_id: str
    cohort: str
    scenario_family: str
    baseline: MerchantBaseline


class SyntheticHold(StrictModel):
    hold: HoldCreate
    scenario_family: str
    true_risk_state: TrueRiskState
    evidence_resolvable: bool
    attack_family: str | None
    operational_target: HoldDecision


class DatasetManifest(StrictModel):
    dataset_id: str
    dataset_version: str
    generator_version: str
    generator_source_sha256: str
    feature_schema_version: str
    seed: int
    merchant_count: int
    transaction_count: int
    hold_count: int
    merchants_per_family: int
    transactions_per_merchant: int
    scenario_families: list[str]
    family_counts: dict[str, int]
    file_sha256: dict[str, str]
    created_at: datetime
    content_sha256: str


def _scenario_profile(family: str) -> dict[str, float]:
    profile = {
        "volume": 1.0,
        "amount": 1.0,
        "new_device": 0.20,
        "new_geo": 0.10,
        "auth_fail": 0.03,
        "refund": 0.02,
        "chargeback": 0.005,
    }
    overrides = {
        "flash_sale": {"volume": 2.8, "new_device": 0.45},
        "influencer_campaign": {"volume": 2.0, "new_device": 0.70},
        "geo_expansion": {"volume": 1.4, "new_geo": 0.70},
        "product_launch": {"volume": 1.6, "amount": 2.2},
        "auth_provider_incident": {"auth_fail": 0.28},
        "seasonal_peak": {"volume": 2.4, "amount": 1.2},
        "card_testing_style": {
            "volume": 2.5,
            "amount": 0.15,
            "new_device": 0.90,
            "auth_fail": 0.65,
        },
        "refund_abuse_style": {"refund": 0.45},
        "synthetic_volume_spike": {"volume": 4.0, "amount": 1.8},
        "distributed_device_ring": {"new_device": 0.08},
        "slow_low_ring": {"volume": 0.9, "new_device": 0.08},
        "account_takeover_shift": {
            "amount": 2.6,
            "new_device": 0.85,
            "new_geo": 0.75,
            "auth_fail": 0.20,
        },
        "mixed_evasion": {
            "volume": 1.6,
            "amount": 1.5,
            "new_device": 0.55,
            "new_geo": 0.35,
            "auth_fail": 0.14,
            "refund": 0.12,
        },
    }
    profile.update(overrides.get(family, {}))
    return profile


def generate_dataset(
    seed: int = 42,
    merchants_per_family: int = 20,
    transactions_per_merchant: int = 80,
) -> tuple[list[SyntheticMerchant], list[TransactionEvent], list[SyntheticHold]]:
    rng = random.Random(seed)
    families = LEGITIMATE_FAMILIES + RISK_FAMILIES
    merchants: list[SyntheticMerchant] = []
    transactions: list[TransactionEvent] = []
    holds: list[SyntheticHold] = []

    for family_index, family in enumerate(families):
        for merchant_offset in range(merchants_per_family):
            merchant_number = family_index * merchants_per_family + merchant_offset
            merchant_id = f"merchant_{merchant_number:05d}"
            profile = _scenario_profile(family)
            ticket_mean = rng.uniform(500, 4500)
            volume_mean = transactions_per_merchant
            hold_time = DATASET_CREATED_AT + timedelta(days=merchant_offset % 7)
            known_devices = {f"{merchant_id}_device_{index}" for index in range(12)}
            known_geos = {"IN-MH", "IN-KA", "IN-DL"}
            baseline = MerchantBaseline(
                volume_mean=volume_mean,
                volume_std=max(5.0, volume_mean * 0.15),
                gmv_mean=volume_mean * ticket_mean,
                gmv_std=max(500.0, volume_mean * ticket_mean * 0.20),
                ticket_size_mean=ticket_mean,
                ticket_size_std=max(50.0, ticket_mean * 0.25),
                refund_rate_mean=0.02,
                refund_rate_std=0.015,
                chargeback_rate_mean=0.005,
                chargeback_rate_std=0.005,
                known_devices=known_devices,
                known_geos=known_geos,
                amount_bin_edges=[
                    0,
                    ticket_mean * 0.5,
                    ticket_mean,
                    ticket_mean * 2,
                    ticket_mean * 6,
                ],
                amount_bin_probabilities=[0.15, 0.35, 0.40, 0.10],
            )
            merchants.append(
                SyntheticMerchant(
                    merchant_id=merchant_id,
                    cohort=rng.choice(["retail", "travel", "digital_goods"]),
                    scenario_family=family,
                    baseline=baseline,
                )
            )

            count = max(12, round(transactions_per_merchant * profile["volume"]))
            ring_family = family in {"distributed_device_ring", "slow_low_ring"}
            ring_group = merchant_offset // 4
            ring_id = f"ring_{family_index:02d}_{ring_group:04d}" if ring_family else None
            for transaction_index in range(count):
                event_time = hold_time - timedelta(
                    seconds=(count - transaction_index) * max(2, 86400 // count)
                )
                is_new_device = rng.random() < profile["new_device"]
                if ring_family:
                    # Adjacent merchants overlap partially instead of sharing one global signature.
                    ring_slot = (merchant_offset + transaction_index) % 6
                    device = f"{ring_id}_device_{ring_slot}"
                elif is_new_device:
                    device = f"{merchant_id}_new_device_{transaction_index}"
                else:
                    device = f"{merchant_id}_device_{transaction_index % 12}"
                geo = (
                    rng.choice(["GB-LND", "SG-01", "AE-DU"])
                    if rng.random() < profile["new_geo"]
                    else rng.choice(sorted(known_geos))
                )
                auth_failed = rng.random() < profile["auth_fail"]
                refund = not auth_failed and rng.random() < profile["refund"]
                chargeback = not auth_failed and rng.random() < profile["chargeback"]
                amount = max(
                    10.0, rng.lognormvariate(math.log(ticket_mean * profile["amount"]), 0.35)
                )
                transactions.append(
                    TransactionEvent(
                        transaction_id=f"txn_{merchant_number:05d}_{transaction_index:05d}",
                        merchant_id=merchant_id,
                        customer_id=(
                            f"{ring_id}_customer_{transaction_index % 16}"
                            if ring_family
                            else f"{merchant_id}_customer_{transaction_index % 48}"
                        ),
                        ring_id=ring_id,
                        timestamp=event_time,
                        amount=round(amount, 2),
                        device_fingerprint=device,
                        customer_geo=geo,
                        auth_status="FAILED" if auth_failed else "APPROVED",
                        refund_timestamp=event_time + timedelta(hours=rng.randint(1, 48))
                        if refund
                        else None,
                        chargeback_timestamp=event_time + timedelta(days=rng.randint(2, 7))
                        if chargeback
                        else None,
                    )
                )

            true_risk_state = (
                TrueRiskState.LEGITIMATE if family in LEGITIMATE_FAMILIES else TrueRiskState.RISKY
            )
            evidence_resolvable = family in EVIDENCE_RESOLVABLE_FAMILIES
            if family in DIRECT_RELEASE_FAMILIES:
                operational_target = HoldDecision.RELEASE
            elif evidence_resolvable:
                operational_target = HoldDecision.EVIDENCE_NEEDED
            else:
                operational_target = HoldDecision.ESCALATE
            holds.append(
                SyntheticHold(
                    hold=HoldCreate(
                        request_id=uuid5(NAMESPACE_URL, f"razortrust:{seed}:{merchant_id}"),
                        merchant_id=merchant_id,
                        source_event_id=f"settlement_{merchant_number:05d}",
                        triggered_at=hold_time,
                        reason_code=family.upper(),
                    ),
                    scenario_family=family,
                    true_risk_state=true_risk_state,
                    evidence_resolvable=evidence_resolvable,
                    attack_family=family if family in RISK_FAMILIES else None,
                    operational_target=operational_target,
                )
            )
    return merchants, transactions, holds


def write_dataset(
    output_dir: str | Path,
    seed: int = 42,
    merchants_per_family: int = 20,
    transactions_per_merchant: int = 80,
) -> DatasetManifest:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    merchants, transactions, holds = generate_dataset(
        seed, merchants_per_family, transactions_per_merchant
    )
    payloads: dict[str, Sequence[StrictModel]] = {
        "merchants.jsonl": merchants,
        "transactions.jsonl": transactions,
        "holds.jsonl": holds,
    }
    digest = hashlib.sha256()
    file_sha256: dict[str, str] = {}
    for filename, rows in payloads.items():
        content = (
            b"\n".join(
                json.dumps(row.model_dump(mode="json"), sort_keys=True).encode("utf-8")
                for row in rows
            )
            + b"\n"
        )
        file_sha256[filename] = hashlib.sha256(content).hexdigest()
        _write_immutable(output / filename, content)
        digest.update(filename.encode("utf-8") + b"\0" + content)

    manifest = DatasetManifest(
        dataset_id=f"synthetic-v3-seed-{seed}",
        dataset_version=DATASET_VERSION,
        generator_version=GENERATOR_VERSION,
        generator_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        feature_schema_version="1.0.0",
        seed=seed,
        merchant_count=len(merchants),
        transaction_count=len(transactions),
        hold_count=len(holds),
        merchants_per_family=merchants_per_family,
        transactions_per_merchant=transactions_per_merchant,
        scenario_families=list(LEGITIMATE_FAMILIES + RISK_FAMILIES),
        family_counts={
            family: merchants_per_family for family in LEGITIMATE_FAMILIES + RISK_FAMILIES
        },
        file_sha256=file_sha256,
        created_at=DATASET_CREATED_AT,
        content_sha256=digest.hexdigest(),
    )
    _write_immutable(
        output / "dataset_manifest.json",
        (manifest.model_dump_json(indent=2) + "\n").encode("utf-8"),
    )
    return manifest


def _write_immutable(path: Path, content: bytes) -> None:
    """Permit deterministic retries but reject mutation of an existing dataset."""
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"refusing to replace generated dataset file: {path}")
        return
    path.write_bytes(content)
