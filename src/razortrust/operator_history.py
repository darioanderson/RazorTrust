from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field

from .domain import HoldCase, HoldCreate, StrictModel
from .layer_execution import LayerExecutionContext, LayerStage
from .live_features import (
    FeatureContractPreview,
    build_feature_contract_preview,
    hash_device_pseudonym,
)
from .workflow import HoldService

SOURCE_CONTRACT_VERSION = "operator-real-history-contract@1"
MAX_IMPORT_BYTES = 10 * 1024 * 1024
MAX_IMPORT_ROWS = 10_000
SOURCE_KINDS = {
    "OPERATOR_SUPPLIED_REAL_HISTORY",
    "PUBLIC_REAL_WORLD_ANONYMIZED",
}
STATUS_VALUES = {"authorized", "captured", "refunded", "failed"}
GEO_RE = re.compile(r"^[A-Z]{2}(?:-[A-Z0-9]{1,12}){0,2}$")

REQUIRED_COLUMNS = (
    "transaction_id",
    "order_id",
    "event_at",
    "observed_at",
    "amount_subunits",
    "currency",
    "status",
    "method",
    "device_pseudonym",
    "customer_geo",
    "refund_amount_subunits",
    "refund_event_at",
    "refund_observed_at",
    "dispute_phase",
    "dispute_status",
    "dispute_event_at",
    "dispute_observed_at",
)
OPTIONAL_COLUMNS = ("fraud_label",)


class HistoryImportValidationError(ValueError):
    pass


class HistoryManifest(StrictModel):
    dataset_id: str
    dataset_name: str
    account_id: str
    source_kind: str
    source_description: str
    source_url: str | None = None
    provider_verified: bool = False
    user_attested_real_data: bool
    original_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalized_records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_contract_version: str = SOURCE_CONTRACT_VERSION
    row_count: int
    fraud_label_count: int
    first_event_at: datetime
    last_event_at: datetime
    imported_at: datetime
    production_action_eligible: bool = False


class HistoryCandidate(StrictModel):
    dataset_id: str
    dataset_name: str
    source_kind: str
    transaction_id: str
    account_id: str
    event_at: datetime
    observed_at: datetime
    amount_subunits: int
    currency: str
    status: str
    method: str
    fraud_label: int | None
    source_file_sha256: str


class CsvCompatibilityReport(StrictModel):
    recognized_family: str
    recognized: bool
    source_kind: str
    live_13_feature_contract_compatible: bool
    present_columns: list[str]
    missing_required_columns: list[str]
    safe_direct_mappings: dict[str, str]
    forbidden_inferences: list[str]
    notes: list[str]


@dataclass(frozen=True)
class PreparedHistoryCase:
    hold: HoldCase
    preview: FeatureContractPreview
    context: LayerExecutionContext


def _parse_datetime(value: str, field: str, row_number: int) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HistoryImportValidationError(
            f"row {row_number}: {field} must be ISO-8601"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise HistoryImportValidationError(
            f"row {row_number}: {field} must include timezone"
        )
    return parsed.astimezone(UTC)


def _optional_datetime(value: str, field: str, row_number: int) -> datetime | None:
    stripped = value.strip()
    if not stripped:
        return None
    return _parse_datetime(stripped, field, row_number)


def _parse_int(value: str, field: str, row_number: int) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise HistoryImportValidationError(
            f"row {row_number}: {field} must be an integer"
        ) from exc


def _parse_fraud_label(value: str, row_number: int) -> int | None:
    stripped = value.strip()
    if not stripped:
        return None
    if stripped not in {"0", "1"}:
        raise HistoryImportValidationError(
            f"row {row_number}: fraud_label must be blank, 0, or 1"
        )
    return int(stripped)


def _canonical_jsonl(records: list[dict[str, Any]]) -> bytes:
    lines = [
        json.dumps(record, sort_keys=True, separators=(",", ":"))
        for record in records
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def audit_csv_bytes(data: bytes) -> CsvCompatibilityReport:
    try:
        text_value = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HistoryImportValidationError("CSV must be UTF-8") from exc

    reader = csv.reader(io.StringIO(text_value))
    try:
        headers = [item.strip() for item in next(reader)]
    except StopIteration:
        headers = []
    header_set = set(headers)

    if {"TransactionID", "TransactionDT", "TransactionAmt", "isFraud"} <= header_set:
        family = "IEEE_CIS_VESTA"
        mappings = {
            "TransactionID": "transaction_id",
            "TransactionDT": "relative event time only",
            "TransactionAmt": "amount",
            "isFraud": "source fraud label",
        }
        forbidden = [
            "Do not infer RazorTrust payment status from isFraud.",
            "Do not infer coarse geography from anonymized address/card fields.",
            "Do not infer refund, chargeback, auth, or observed_at fields.",
        ]
        notes = [
            "Recognized as IEEE-CIS/Vesta-style schema.",
            "Useful as public real-world fraud ground truth, not a full live feature source.",
        ]
        kind = "PUBLIC_REAL_WORLD_ANONYMIZED"
    elif {"Time", "Amount", "Class"} <= header_set and any(
        f"V{i}" in header_set for i in range(1, 29)
    ):
        family = "ULB_EUROPEAN_CREDITCARD_2013"
        mappings = {
            "Time": "relative transaction time only",
            "Amount": "amount",
            "Class": "source fraud label",
        }
        forbidden = [
            "Do not reverse PCA columns V1-V28 into device, geo, auth, refund, or dispute.",
            "Do not invent merchant identity or observed_at timestamps.",
        ]
        notes = [
            "Recognized as the ULB European credit-card fraud benchmark.",
            "PCA anonymization prevents the live 13-feature provenance contract.",
        ]
        kind = "PUBLIC_REAL_WORLD_ANONYMIZED"
    elif set(REQUIRED_COLUMNS) <= header_set:
        family = "RAZORTRUST_FULL_HISTORY_CONTRACT"
        mappings = {name: name for name in REQUIRED_COLUMNS}
        forbidden = []
        notes = [
            "Full RazorTrust historical-input contract detected.",
            "Rows still undergo point-in-time and data-quality validation.",
        ]
        kind = "OPERATOR_SUPPLIED_REAL_HISTORY"
    else:
        family = "UNKNOWN"
        mappings = {}
        forbidden = ["No semantic field inference is performed for unknown schemas."]
        notes = ["Schema not recognized; import is blocked until the full contract is supplied."]
        kind = "UNKNOWN"

    missing = sorted(set(REQUIRED_COLUMNS) - header_set)
    return CsvCompatibilityReport(
        recognized_family=family,
        recognized=family != "UNKNOWN",
        source_kind=kind,
        live_13_feature_contract_compatible=not missing,
        present_columns=headers,
        missing_required_columns=missing,
        safe_direct_mappings=mappings,
        forbidden_inferences=forbidden,
        notes=notes,
    )


class OperatorHistoryStore:
    def __init__(self, root: str | Path | None = None) -> None:
        configured = (
            root
            if root is not None
            else os.getenv(
                "RAZORTRUST_OPERATOR_HISTORY_PATH",
                "/app/var/operator-history",
            )
        )
        self.root = Path(configured)

    def _dataset_dir(self, dataset_id: str) -> Path:
        if not re.fullmatch(r"hist_[0-9a-f]{16}", dataset_id):
            raise HistoryImportValidationError("invalid dataset_id")
        return self.root / dataset_id

    def import_csv_bytes(
        self,
        data: bytes,
        *,
        dataset_name: str,
        account_id: str,
        source_kind: str,
        source_description: str,
        source_url: str | None,
        user_attested_real_data: bool,
    ) -> HistoryManifest:
        if len(data) > MAX_IMPORT_BYTES:
            raise HistoryImportValidationError("CSV exceeds 10 MiB import limit")
        if not dataset_name.strip() or len(dataset_name) > 160:
            raise HistoryImportValidationError("dataset_name must be 1-160 characters")
        if not account_id.strip() or len(account_id) > 128:
            raise HistoryImportValidationError("account_id must be 1-128 characters")
        if source_kind not in SOURCE_KINDS:
            raise HistoryImportValidationError(
                f"source_kind must be one of {sorted(SOURCE_KINDS)}"
            )
        if not source_description.strip() or len(source_description) > 500:
            raise HistoryImportValidationError(
                "source_description must be 1-500 characters"
            )
        if not user_attested_real_data:
            raise HistoryImportValidationError(
                "user_attested_real_data=true is required; demo history is rejected"
            )

        compatibility = audit_csv_bytes(data)
        if not compatibility.live_13_feature_contract_compatible:
            missing = ", ".join(compatibility.missing_required_columns)
            raise HistoryImportValidationError(
                f"CSV is not live-13-feature compatible; missing: {missing}"
            )

        original_sha = hashlib.sha256(data).hexdigest()
        dataset_id = f"hist_{original_sha[:16]}"
        directory = self._dataset_dir(dataset_id)
        if directory.exists():
            return self.get_manifest(dataset_id)

        text_data = data.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text_data))
        if reader.fieldnames is None:
            raise HistoryImportValidationError("CSV header is missing")
        headers = {name.strip() for name in reader.fieldnames if name}
        unknown = headers - set(REQUIRED_COLUMNS) - set(OPTIONAL_COLUMNS)
        if unknown:
            raise HistoryImportValidationError(
                f"unknown CSV columns are not allowed: {sorted(unknown)}"
            )

        records: list[dict[str, Any]] = []
        seen_transactions: set[str] = set()
        fraud_count = 0

        for row_number, raw in enumerate(reader, start=2):
            if len(records) >= MAX_IMPORT_ROWS:
                raise HistoryImportValidationError("CSV exceeds 10,000 row import limit")
            transaction_id = (raw.get("transaction_id") or "").strip()
            order_id = (raw.get("order_id") or "").strip()
            if not transaction_id or len(transaction_id) > 128:
                raise HistoryImportValidationError(
                    f"row {row_number}: invalid transaction_id"
                )
            if transaction_id in seen_transactions:
                raise HistoryImportValidationError(
                    f"row {row_number}: duplicate transaction_id {transaction_id}"
                )
            seen_transactions.add(transaction_id)
            if not order_id or len(order_id) > 128:
                raise HistoryImportValidationError(f"row {row_number}: invalid order_id")

            event_at = _parse_datetime(raw["event_at"].strip(), "event_at", row_number)
            observed_at = _parse_datetime(
                raw["observed_at"].strip(), "observed_at", row_number
            )
            if observed_at < event_at:
                raise HistoryImportValidationError(
                    f"row {row_number}: observed_at cannot precede event_at"
                )

            amount = _parse_int(
                raw["amount_subunits"].strip(), "amount_subunits", row_number
            )
            if amount <= 0:
                raise HistoryImportValidationError(
                    f"row {row_number}: amount_subunits must be positive"
                )
            currency = raw["currency"].strip().upper()
            if not re.fullmatch(r"[A-Z]{3}", currency):
                raise HistoryImportValidationError(
                    f"row {row_number}: currency must be a 3-letter code"
                )
            status = raw["status"].strip().lower()
            if status not in STATUS_VALUES:
                raise HistoryImportValidationError(
                    f"row {row_number}: unsupported status {status!r}"
                )
            method = raw["method"].strip().lower()
            if not method or len(method) > 64:
                raise HistoryImportValidationError(f"row {row_number}: invalid method")
            device = raw["device_pseudonym"].strip()
            if len(device) < 16 or len(device) > 256:
                raise HistoryImportValidationError(
                    f"row {row_number}: device_pseudonym must be 16-256 characters"
                )
            geo = raw["customer_geo"].strip().upper()
            if not GEO_RE.fullmatch(geo):
                raise HistoryImportValidationError(
                    f"row {row_number}: customer_geo must be coarse, for example IN-DL"
                )

            refund_amount = _parse_int(
                raw["refund_amount_subunits"].strip() or "0",
                "refund_amount_subunits",
                row_number,
            )
            if refund_amount < 0 or refund_amount > amount:
                raise HistoryImportValidationError(
                    f"row {row_number}: refund amount must be between 0 and amount"
                )
            refund_event = _optional_datetime(
                raw["refund_event_at"], "refund_event_at", row_number
            )
            refund_observed = _optional_datetime(
                raw["refund_observed_at"], "refund_observed_at", row_number
            )
            if refund_amount > 0 and (refund_event is None or refund_observed is None):
                raise HistoryImportValidationError(
                    f"row {row_number}: refund timestamps required when refund_amount > 0"
                )
            if refund_observed and refund_event and refund_observed < refund_event:
                raise HistoryImportValidationError(
                    f"row {row_number}: refund_observed_at cannot precede refund_event_at"
                )

            dispute_phase = raw["dispute_phase"].strip().lower()
            dispute_status = raw["dispute_status"].strip().lower()
            dispute_event = _optional_datetime(
                raw["dispute_event_at"], "dispute_event_at", row_number
            )
            dispute_observed = _optional_datetime(
                raw["dispute_observed_at"], "dispute_observed_at", row_number
            )
            dispute_present = any(
                (dispute_phase, dispute_status, dispute_event, dispute_observed)
            )
            if dispute_present and not all(
                (dispute_phase, dispute_status, dispute_event, dispute_observed)
            ):
                raise HistoryImportValidationError(
                    f"row {row_number}: dispute fields must be supplied together"
                )
            if dispute_observed and dispute_event and dispute_observed < dispute_event:
                raise HistoryImportValidationError(
                    f"row {row_number}: dispute_observed_at cannot precede dispute_event_at"
                )

            fraud_label = _parse_fraud_label(raw.get("fraud_label", ""), row_number)
            fraud_count += int(fraud_label == 1)

            records.append(
                {
                    "transaction_id": transaction_id,
                    "order_id": order_id,
                    "event_at": event_at.isoformat(),
                    "observed_at": observed_at.isoformat(),
                    "amount_subunits": amount,
                    "currency": currency,
                    "status": status,
                    "method": method,
                    "device_fingerprint_sha256": hash_device_pseudonym(
                        account_id, device
                    ),
                    "customer_geo": geo,
                    "refund_amount_subunits": refund_amount,
                    "refund_event_at": refund_event.isoformat() if refund_event else None,
                    "refund_observed_at": (
                        refund_observed.isoformat() if refund_observed else None
                    ),
                    "dispute_phase": dispute_phase or None,
                    "dispute_status": dispute_status or None,
                    "dispute_event_at": (
                        dispute_event.isoformat() if dispute_event else None
                    ),
                    "dispute_observed_at": (
                        dispute_observed.isoformat() if dispute_observed else None
                    ),
                    "fraud_label": fraud_label,
                }
            )

        if not records:
            raise HistoryImportValidationError("CSV contains no data rows")

        records.sort(key=lambda row: (row["event_at"], row["transaction_id"]))
        normalized = _canonical_jsonl(records)
        normalized_sha = hashlib.sha256(normalized).hexdigest()
        manifest = HistoryManifest(
            dataset_id=dataset_id,
            dataset_name=dataset_name.strip(),
            account_id=account_id.strip(),
            source_kind=source_kind,
            source_description=source_description.strip(),
            source_url=source_url.strip() if source_url else None,
            user_attested_real_data=True,
            original_file_sha256=original_sha,
            normalized_records_sha256=normalized_sha,
            row_count=len(records),
            fraud_label_count=fraud_count,
            first_event_at=datetime.fromisoformat(records[0]["event_at"]),
            last_event_at=datetime.fromisoformat(records[-1]["event_at"]),
            imported_at=datetime.now(UTC),
        )

        directory.mkdir(parents=True, exist_ok=False)
        (directory / "records.jsonl").write_bytes(normalized)
        (directory / "manifest.json").write_text(
            json.dumps(manifest.model_dump(mode="json"), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest

    def list_manifests(self) -> list[HistoryManifest]:
        if not self.root.exists():
            return []
        manifests = []
        for path in sorted(self.root.glob("hist_*/manifest.json")):
            try:
                manifests.append(
                    HistoryManifest.model_validate_json(path.read_text(encoding="utf-8"))
                )
            except (OSError, ValueError):
                continue
        return sorted(manifests, key=lambda item: item.imported_at, reverse=True)

    def get_manifest(self, dataset_id: str) -> HistoryManifest:
        path = self._dataset_dir(dataset_id) / "manifest.json"
        if not path.is_file():
            raise LookupError("history dataset was not found")
        return HistoryManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def _records(self, dataset_id: str) -> list[dict[str, Any]]:
        path = self._dataset_dir(dataset_id) / "records.jsonl"
        if not path.is_file():
            raise LookupError("history dataset records were not found")
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        manifest = self.get_manifest(dataset_id)
        actual = hashlib.sha256(_canonical_jsonl(records)).hexdigest()
        if actual != manifest.normalized_records_sha256:
            raise HistoryImportValidationError("normalized history hash verification failed")
        return records

    def candidates(self, *, limit: int = 100) -> list[HistoryCandidate]:
        items: list[HistoryCandidate] = []
        for manifest in self.list_manifests():
            for record in reversed(self._records(manifest.dataset_id)):
                items.append(
                    HistoryCandidate(
                        dataset_id=manifest.dataset_id,
                        dataset_name=manifest.dataset_name,
                        source_kind=manifest.source_kind,
                        transaction_id=record["transaction_id"],
                        account_id=manifest.account_id,
                        event_at=datetime.fromisoformat(record["event_at"]),
                        observed_at=datetime.fromisoformat(record["observed_at"]),
                        amount_subunits=record["amount_subunits"],
                        currency=record["currency"],
                        status=record["status"],
                        method=record["method"],
                        fraud_label=record.get("fraud_label"),
                        source_file_sha256=manifest.original_file_sha256,
                    )
                )
        items.sort(key=lambda item: item.observed_at, reverse=True)
        return items[:limit]

    def _preview(
        self, dataset_id: str, transaction_id: str
    ) -> tuple[FeatureContractPreview, dict[str, Any], HistoryManifest]:
        manifest = self.get_manifest(dataset_id)
        records = self._records(dataset_id)
        target = next(
            (row for row in records if row["transaction_id"] == transaction_id), None
        )
        if target is None:
            raise LookupError("transaction was not found in history dataset")

        as_of = datetime.fromisoformat(target["observed_at"]).astimezone(UTC)
        payment_rows = []
        telemetry_rows = []
        refund_rows = []
        dispute_rows = []
        order_by_payment: dict[str, str] = {}

        for row in records:
            event_at = datetime.fromisoformat(row["event_at"]).astimezone(UTC)
            observed_at = datetime.fromisoformat(row["observed_at"]).astimezone(UTC)
            if observed_at > as_of:
                continue

            refund_observed = (
                datetime.fromisoformat(row["refund_observed_at"]).astimezone(UTC)
                if row["refund_observed_at"]
                else None
            )
            refund_known = bool(refund_observed and refund_observed <= as_of)
            known_refund_amount = row["refund_amount_subunits"] if refund_known else 0

            payment_rows.append(
                SimpleNamespace(
                    payment_id=row["transaction_id"],
                    order_id=row["order_id"],
                    account_id=manifest.account_id,
                    status=row["status"],
                    amount=row["amount_subunits"],
                    currency=row["currency"],
                    method=row["method"],
                    captured=row["status"] in {"captured", "refunded"},
                    amount_refunded=known_refund_amount,
                    refund_status=(
                        "full"
                        if known_refund_amount == row["amount_subunits"]
                        and known_refund_amount > 0
                        else "partial"
                        if known_refund_amount > 0
                        else None
                    ),
                    provider_created_at=event_at,
                    observed_at=observed_at,
                )
            )
            telemetry_rows.append(
                SimpleNamespace(
                    order_id=row["order_id"],
                    payment_id=row["transaction_id"],
                    device_fingerprint_sha256=row["device_fingerprint_sha256"],
                    customer_geo=row["customer_geo"],
                    observed_at=observed_at,
                )
            )
            order_by_payment[row["transaction_id"]] = row["order_id"]

            if refund_known:
                refund_event = datetime.fromisoformat(row["refund_event_at"]).astimezone(UTC)
                refund_rows.append(
                    SimpleNamespace(
                        refund_id=f"hist_refund_{row['transaction_id']}",
                        payment_id=row["transaction_id"],
                        account_id=manifest.account_id,
                        amount=row["refund_amount_subunits"],
                        currency=row["currency"],
                        status="processed",
                        provider_created_at=refund_event,
                        observed_at=refund_observed,
                    )
                )

            if row["dispute_observed_at"]:
                dispute_observed = datetime.fromisoformat(
                    row["dispute_observed_at"]
                ).astimezone(UTC)
                if dispute_observed <= as_of:
                    dispute_event = datetime.fromisoformat(
                        row["dispute_event_at"]
                    ).astimezone(UTC)
                    dispute_rows.append(
                        SimpleNamespace(
                            dispute_id=f"hist_dispute_{row['transaction_id']}",
                            payment_id=row["transaction_id"],
                            account_id=manifest.account_id,
                            status=row["dispute_status"],
                            phase=row["dispute_phase"],
                            provider_created_at=dispute_event,
                            observed_at=dispute_observed,
                        )
                    )

        preview = build_feature_contract_preview(
            account_id=manifest.account_id,
            as_of=as_of,
            payment_rows=payment_rows,
            refund_rows=refund_rows,
            dispute_rows=dispute_rows,
            telemetry_rows=telemetry_rows,
            order_by_payment=order_by_payment,
        ).model_copy(
            update={
                "provider": "OPERATOR_HISTORY",
                "source_contract_version": SOURCE_CONTRACT_VERSION,
            }
        )
        return preview, target, manifest

    async def prepare_case(
        self,
        dataset_id: str,
        transaction_id: str,
        hold_service: HoldService,
    ) -> PreparedHistoryCase:
        preview, target, manifest = self._preview(dataset_id, transaction_id)
        request_id = uuid5(
            NAMESPACE_URL, f"razortrust:history:{dataset_id}:{transaction_id}"
        )
        hold, _created = await hold_service.create_hold(
            HoldCreate(
                request_id=request_id,
                merchant_id=manifest.account_id,
                source_event_id=transaction_id,
                triggered_at=preview.as_of,
                reason_code="IMPORTED_REAL_HISTORY_REVIEW",
            )
        )

        source_label = target.get("fraud_label")
        ground_truth = LayerStage(
            layer="SOURCE_GROUND_TRUTH",
            status="RESULT" if source_label is not None else "NOT_AVAILABLE",
            output={
                "dataset_id": dataset_id,
                "source_kind": manifest.source_kind,
                "source_url": manifest.source_url,
                "source_file_sha256": manifest.original_file_sha256,
                "transaction_id": transaction_id,
                "source_fraud_label": source_label,
                "fraud_label_semantics": (
                    "source-provided evaluation label; not Razorpay dispute confirmation"
                ),
                "provider_verified": False,
                "evaluation_only": True,
                "used_for_model_scoring": False,
                "used_for_policy": False,
            },
        )
        context = LayerExecutionContext(
            source_mode=manifest.source_kind,
            input_provenance={
                "dataset_id": dataset_id,
                "dataset_name": manifest.dataset_name,
                "source_description": manifest.source_description,
                "source_url": manifest.source_url,
                "source_file_sha256": manifest.original_file_sha256,
                "normalized_records_sha256": manifest.normalized_records_sha256,
                "provider_verified": False,
                "user_attested_real_data": manifest.user_attested_real_data,
            },
            secure_ingestion_status="RESULT",
            secure_ingestion_output={
                "source_contract_version": SOURCE_CONTRACT_VERSION,
                "original_file_sha256": manifest.original_file_sha256,
                "normalized_records_sha256": manifest.normalized_records_sha256,
                "row_count": manifest.row_count,
                "raw_device_identifiers_stored": False,
                "provider_signature_verified": False,
                "provider_verified": False,
                "note": (
                    "Imported history is hash-verified and source-provenanced, "
                    "but is not claimed to be provider verified."
                ),
            },
            ground_truth_stage=ground_truth,
        )
        return PreparedHistoryCase(hold=hold, preview=preview, context=context)
