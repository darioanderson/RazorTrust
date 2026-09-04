from __future__ import annotations

from typing import Literal, cast

import numpy as np

from ..domain import FeatureContribution, HoldDecision


class ExplanationFidelityError(ValueError):
    pass


def top_contributions(
    feature_names: list[str],
    observed_values: np.ndarray,
    contribution_values: np.ndarray,
    model_output: HoldDecision,
    *,
    limit: int = 3,
    requested_features: set[str] | None = None,
) -> list[FeatureContribution]:
    """Return faithful top-ranked model contributions for one case and output class."""
    if len(feature_names) != len(observed_values) or len(feature_names) != len(contribution_values):
        raise ValueError("feature, value, and contribution arrays must have equal length")
    ranked_indices = np.argsort(np.abs(contribution_values))[::-1][:limit]
    ranked_features = {feature_names[index] for index in ranked_indices}
    if requested_features is not None and not requested_features <= ranked_features:
        unsupported = ", ".join(sorted(requested_features - ranked_features))
        raise ExplanationFidelityError(
            f"requested explanation factors are not top model contributions: {unsupported}"
        )
    direction = cast(
        Literal["toward_RELEASE", "toward_EVIDENCE_NEEDED", "toward_ESCALATE"],
        f"toward_{model_output.value}",
    )
    return [
        FeatureContribution(
            feature=feature_names[index],
            observed_value=float(observed_values[index]),
            contribution_value=float(contribution_values[index]),
            direction=direction,
            reference="Tree SHAP contribution for the candidate model output",
            attribution_method="tree_shap",
            model_output_explained=model_output,
            template_id=f"{feature_names[index]}_v1",
        )
        for index in ranked_indices
    ]
