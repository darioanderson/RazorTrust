from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import skops.io as sio
from pydantic import Field
from xgboost import XGBClassifier

from ..audit import canonical_json
from ..costs import DEFAULT_COST_MATRIX
from ..domain import StrictModel, Thresholds
from ..features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION
from ..security import sign_manifest, verify_manifest
from .modeling import ModelBundle, TrainingResult

MANIFEST_NAME = "release-manifest.jcs.json"
SIGNATURE_NAME = "release-manifest.sig"
_ALLOWED_TYPE_PREFIXES = ("razortrust.", "sklearn.", "xgboost.", "numpy.", "scipy.")


class ReleaseManifest(StrictModel):
    schema_version: str = "2.0"
    release_id: str
    model_version: str
    files: dict[str, str]
    feature_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cost_matrix_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_method: str
    release_threshold: float
    escalate_threshold: float
    created_at: datetime
    signer_key_id: str


class ReleaseVerificationError(ValueError):
    pass


def save_model_release(
    directory: str | Path,
    result: TrainingResult,
    *,
    private_key_b64: str,
    signer_key_id: str,
    split_manifest_sha256: str,
    training_dataset_sha256: str,
    evaluation_report_sha256: str,
    release_id: str,
) -> ReleaseManifest:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(f"model release directory must be empty: {destination}")

    classifier_path = destination / "xgboost.ubj"
    anomaly_path = destination / "anomaly.skops"
    calibrator_path = destination / "calibrator.skops"
    result.bundle.classifier.save_model(classifier_path)
    sio.dump(result.bundle.anomaly_scorer, anomaly_path)
    if result.bundle.calibrator is not None:
        sio.dump(result.bundle.calibrator, calibrator_path)

    metadata = {
        "model_version": result.bundle.model_version,
        "calibration_method": result.bundle.calibration_method,
        "thresholds": result.bundle.thresholds.model_dump(mode="json"),
        "policy_version": result.bundle.policy_version,
        "cost_matrix_version": result.bundle.cost_matrix_version,
        "cost_matrix_sha256": result.bundle.cost_matrix_sha256,
        "anomaly_model_version": str(getattr(result.bundle.anomaly_scorer, "version", "unknown")),
        "anomaly_reference_mode": str(
            getattr(result.bundle.anomaly_scorer, "reference_mode", "legacy_in_sample")
        ),
    }
    feature_schema = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "columns": list(FEATURE_COLUMNS),
    }
    _write_json(destination / "metadata.json", metadata)
    _write_json(destination / "feature_schema.json", feature_schema)
    _write_json(destination / "cost_matrix.json", DEFAULT_COST_MATRIX.model_dump(mode="json"))

    artifact_names = [
        "xgboost.ubj",
        "anomaly.skops",
        "metadata.json",
        "feature_schema.json",
        "cost_matrix.json",
    ]
    if calibrator_path.exists():
        artifact_names.append("calibrator.skops")
    file_hashes = {name: _sha256(destination / name) for name in artifact_names}
    manifest = ReleaseManifest(
        release_id=release_id,
        model_version=result.bundle.model_version,
        files=file_hashes,
        feature_schema_sha256=file_hashes["feature_schema.json"],
        split_manifest_sha256=split_manifest_sha256,
        training_dataset_sha256=training_dataset_sha256,
        evaluation_report_sha256=evaluation_report_sha256,
        cost_matrix_sha256=file_hashes["cost_matrix.json"],
        calibration_method=result.bundle.calibration_method,
        release_threshold=result.bundle.thresholds.release,
        escalate_threshold=result.bundle.thresholds.escalate,
        created_at=datetime.now(UTC),
        signer_key_id=signer_key_id,
    )
    manifest_data = manifest.model_dump(mode="json")
    (destination / MANIFEST_NAME).write_bytes(canonical_json(manifest_data))
    signature = sign_manifest(manifest_data, private_key_b64)
    (destination / SIGNATURE_NAME).write_text(signature + "\n", encoding="ascii")
    return manifest


def load_verified_model_release(directory: str | Path, *, public_key_b64: str) -> ModelBundle:
    source = Path(directory)
    try:
        manifest_data = json.loads((source / MANIFEST_NAME).read_bytes())
        signature = (source / SIGNATURE_NAME).read_text(encoding="ascii").strip()
        manifest = ReleaseManifest.model_validate(manifest_data)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ReleaseVerificationError("model release manifest is missing or invalid") from exc
    if not verify_manifest(manifest_data, signature, public_key_b64):
        raise ReleaseVerificationError("model release signature is invalid")
    for name, expected_hash in manifest.files.items():
        path = source / name
        if not path.is_file() or _sha256(path) != expected_hash:
            raise ReleaseVerificationError(f"model release artifact hash mismatch: {name}")

    feature_schema = _read_json(source / "feature_schema.json")
    if feature_schema != {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "columns": list(FEATURE_COLUMNS),
    }:
        raise ReleaseVerificationError("model release feature schema is unsupported")
    cost_matrix = _read_json(source / "cost_matrix.json")
    if cost_matrix != DEFAULT_COST_MATRIX.model_dump(mode="json"):
        raise ReleaseVerificationError("model release cost matrix is unsupported")

    metadata = _read_json(source / "metadata.json")
    anomaly = _safe_skops_load(source / "anomaly.skops")
    calibrator = (
        _safe_skops_load(source / "calibrator.skops")
        if "calibrator.skops" in manifest.files
        else None
    )
    classifier = XGBClassifier()
    classifier.load_model(source / "xgboost.ubj")
    return ModelBundle(
        model_version=str(metadata["model_version"]),
        anomaly_scorer=anomaly,
        classifier=classifier,
        calibrator=calibrator,
        calibration_method=str(metadata["calibration_method"]),
        thresholds=Thresholds.model_validate(metadata["thresholds"]),
        policy_version=str(metadata["policy_version"]),
        cost_matrix_version=str(metadata["cost_matrix_version"]),
        cost_matrix_sha256=str(metadata["cost_matrix_sha256"]),
    )


def _safe_skops_load(path: Path) -> Any:
    untrusted = sio.get_untrusted_types(file=path)
    rejected = [name for name in untrusted if not name.startswith(_ALLOWED_TYPE_PREFIXES)]
    if rejected:
        raise ReleaseVerificationError(f"model release contains unapproved types: {rejected}")
    return sio.load(path, trusted=untrusted)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(canonical_json(value))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ReleaseVerificationError(f"expected a JSON object: {path.name}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
