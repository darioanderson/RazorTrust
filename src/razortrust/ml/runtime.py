from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..domain import (
    FeatureVector,
    ModelPrediction,
    ProbabilityVector,
)
from ..features import FEATURE_COLUMNS
from ..novelty_policy import DEFAULT_NOVELTY_POLICY
from .explain import top_contributions
from .modeling import INDEX_TO_LABEL, MODEL_COLUMNS, ModelBundle
from .release import load_verified_model_release


class TrainedRiskModel:
    """Runtime adapter for a validated offline model bundle."""

    def __init__(self, bundle: ModelBundle) -> None:
        _configure_matplotlib_cache()
        import shap

        self.bundle = bundle
        self.version = bundle.model_version
        self.thresholds = bundle.thresholds
        self._explainer = shap.TreeExplainer(bundle.classifier)

    @classmethod
    def load_release(cls, path: str | Path, *, public_key_b64: str) -> TrainedRiskModel:
        return cls(load_verified_model_release(path, public_key_b64=public_key_b64))

    def predict(self, features: FeatureVector, cohort: str = "pooled") -> ModelPrediction:
        frame = pd.DataFrame([features.model_dump()], columns=FEATURE_COLUMNS)
        probabilities = self.bundle.predict_proba(frame, pd.Series([cohort]))[0]
        candidate_index = int(np.argmax(probabilities))
        candidate = INDEX_TO_LABEL[candidate_index]
        model_frame = frame.copy()
        anomaly_details = self.bundle.anomaly_scorer.score_details(frame, pd.Series([cohort]))
        model_frame[MODEL_COLUMNS[-1]] = anomaly_details.percentiles
        explanation = self._explainer(model_frame.loc[:, list(MODEL_COLUMNS)])
        values = np.asarray(explanation.values)
        if values.ndim != 3:
            raise ValueError(f"unexpected multiclass SHAP shape: {values.shape}")
        contributions = values[0, :, candidate_index]
        top_features = top_contributions(
            list(MODEL_COLUMNS),
            model_frame.loc[0, list(MODEL_COLUMNS)].to_numpy(dtype=float),
            contributions,
            candidate,
        )
        anomaly_score = float(model_frame.loc[0, MODEL_COLUMNS[-1]])
        novelty_override = (
            float(probabilities[0]) >= self.thresholds.release
            and anomaly_score >= DEFAULT_NOVELTY_POLICY.strong_signal_percentile
        )
        return ModelPrediction(
            probabilities=ProbabilityVector(
                release=float(probabilities[0]),
                evidence_needed=float(probabilities[1]),
                escalate=float(probabilities[2]),
            ),
            calibrated=self.bundle.calibration_method != "uncalibrated",
            calibration_method=self.bundle.calibration_method,
            anomaly_score=anomaly_score,
            anomaly_raw_score=float(anomaly_details.raw_scores[0]),
            anomaly_reference_max=float(anomaly_details.reference_max_scores[0]),
            anomaly_tail_excess=float(anomaly_details.tail_excesses[0]),
            anomaly_reference_size=int(anomaly_details.reference_sizes[0]),
            anomaly_model_version=anomaly_details.scorer_version,
            anomaly_reference_mode=anomaly_details.reference_mode,
            novelty_override=novelty_override,
            top_features=top_features,
            model_version=self.version,
        )


def _configure_matplotlib_cache() -> None:
    config_path = Path(tempfile.gettempdir()) / "razortrust-matplotlib"
    config_path.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(config_path))
