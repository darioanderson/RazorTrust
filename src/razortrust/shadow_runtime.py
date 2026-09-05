from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .risk import RiskModel, UnavailableRiskModel

EXPECTED_CHAMPION = "xgb-if-settlement@2"


@dataclass(frozen=True, slots=True)
class ShadowAnalysisRuntime:
    """Independent analysis runtime with no money-moving authority."""

    model: RiskModel
    status: str
    expected_model_version: str
    model_version: str
    release_path: str | None
    metadata: dict[str, Any]
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.status == "READY"


def load_shadow_analysis_runtime() -> ShadowAnalysisRuntime:
    release_path = os.getenv("RAZORTRUST_SHADOW_MODEL_RELEASE_PATH")
    public_key_path = os.getenv("RAZORTRUST_SHADOW_MODEL_PUBLIC_KEY_PATH")
    expected = os.getenv("RAZORTRUST_SHADOW_MODEL_EXPECTED_VERSION", EXPECTED_CHAMPION)

    model: RiskModel
    if not release_path or not public_key_path:
        reason = "shadow analysis model release is not configured"
        model = UnavailableRiskModel(reason)
        return ShadowAnalysisRuntime(
            model=model,
            status="NOT_CONFIGURED",
            expected_model_version=expected,
            model_version=model.version,
            release_path=release_path,
            metadata={},
            error=reason,
        )

    try:
        release = Path(release_path)
        public_key = Path(public_key_path).read_text(encoding="ascii").strip()
        if not public_key:
            raise ValueError("shadow model public key is empty")

        from .ml.runtime import TrainedRiskModel

        model = TrainedRiskModel.load_release(release, public_key_b64=public_key)
        if model.version != expected:
            raise ValueError(
                f"shadow model version mismatch: expected {expected}, loaded {model.version}"
            )

        metadata_path = release / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("signed model metadata must be a JSON object")
        if str(metadata.get("model_version")) != expected:
            raise ValueError("signed model metadata version does not match expected champion")

        return ShadowAnalysisRuntime(
            model=model,
            status="READY",
            expected_model_version=expected,
            model_version=model.version,
            release_path=str(release),
            metadata=metadata,
        )
    except (ImportError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        reason = f"{type(exc).__name__}: {exc}"
        model = UnavailableRiskModel(reason)
        return ShadowAnalysisRuntime(
            model=model,
            status="ERROR",
            expected_model_version=expected,
            model_version=model.version,
            release_path=release_path,
            metadata={},
            error=reason,
        )
