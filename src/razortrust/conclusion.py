from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from .domain import StrictModel


class DecisionConclusion(StrictModel):
    headline: str
    recommendation: str
    source_statement: str
    data_quality_statement: str
    model_statement: str
    dispute_statement: str
    strongest_signals: list[str] = []
    unresolved_questions: list[str] = []
    journey: list[str] = []
    next_step: str
    human_authorization_required: bool = True
    automatic_release_enabled: bool = False
    production_action_eligible: bool = False


def build_decision_conclusion(
    *,
    stages: Sequence[Any],
    recommendation: str,
    source_mode: str,
) -> DecisionConclusion:
    """Summarize only facts already emitted by the execution layers."""
    by_layer = {stage.layer: stage for stage in stages}
    journey = [f"{stage.layer}: {stage.status}" for stage in stages]
    blockers = _unique(blocker for stage in stages for blocker in getattr(stage, "blockers", []))

    quality = by_layer.get("DATA_QUALITY_FIREWALL") or by_layer.get("DATASET_INTEGRITY")
    if quality is None:
        data_quality = "No explicit data-quality result was returned."
    elif quality.status in {"PASS", "RESULT"}:
        data_quality = "The declared data-quality and integrity checks completed."
    else:
        data_quality = f"Data quality is {quality.status}; model use is fail-closed."

    core = by_layer.get("CORE_ML") or by_layer.get("BENCHMARK_XGBOOST")
    model_statement = _model_statement(core, by_layer)
    strongest_signals = _strongest_signals(by_layer)
    dispute_statement = _dispute_statement(by_layer, source_mode)

    unresolved = list(blockers)
    for layer_name in ("CORE_ML", "OPA_GUARDRAILS", "OPA"):
        stage = by_layer.get(layer_name)
        if stage is not None and stage.status in {"BLOCKED", "ERROR", "NOT_AVAILABLE"}:
            unresolved.append(f"{layer_name}: {stage.status}")
    unresolved = _unique(unresolved)

    normalized = recommendation.upper()
    if normalized == "RELEASE":
        next_step = "A human reviewer must approve RELEASE before any settlement action."
    elif normalized == "EVIDENCE_NEEDED":
        next_step = "Request only the missing case evidence, then return the case to human review."
    elif normalized == "ESCALATE":
        next_step = "Route the case to a human risk analyst; no automatic release is permitted."
    else:
        next_step = "Keep this research result separate from production and complete human review."

    return DecisionConclusion(
        headline=f"{recommendation}: human decision still required",
        recommendation=recommendation,
        source_statement=f"The case was evaluated from source mode {source_mode}.",
        data_quality_statement=data_quality,
        model_statement=model_statement,
        dispute_statement=dispute_statement,
        strongest_signals=strongest_signals,
        unresolved_questions=unresolved,
        journey=journey,
        next_step=next_step,
    )


def _model_statement(stage: Any | None, by_layer: dict[str, Any]) -> str:
    if stage is None:
        return "No model-scoring layer returned a result."
    if stage.status not in {"RESULT", "SCORED"}:
        return f"The model layer is {stage.status}; its output cannot authorize an action."
    output = stage.output or {}
    probabilities = output.get("probabilities")
    if isinstance(probabilities, dict) and probabilities:
        numeric = {
            str(name): float(value)
            for name, value in probabilities.items()
            if isinstance(value, int | float)
        }
        if numeric:
            name, value = max(numeric.items(), key=lambda item: item[1])
            calibration = "calibrated" if output.get("calibrated") else "reported"
            return f"The highest {calibration} model probability is {name} at {value:.1%}."
    calibration_stage = by_layer.get("PROBABILITY_CALIBRATION")
    if calibration_stage is not None and calibration_stage.output:
        calibrated = calibration_stage.output.get("probabilities")
        if isinstance(calibrated, dict):
            fraud = calibrated.get("fraud")
            if isinstance(fraud, int | float):
                return f"The calibrated research fraud probability is {float(fraud):.1%}."
    raw = output.get("raw_probability_fraud")
    if isinstance(raw, int | float):
        return f"The research XGBoost fraud score is {float(raw):.1%} before calibration."
    return "The model completed scoring; inspect its layer output for the numeric result."


def _strongest_signals(by_layer: dict[str, Any]) -> list[str]:
    explanation = by_layer.get("SHAP_EXPLANATION") or by_layer.get("SHAP")
    if explanation is None or not explanation.output:
        return []
    values = explanation.output.get("top_features")
    if not isinstance(values, list):
        return []
    signals: list[str] = []
    for item in values[:3]:
        if not isinstance(item, dict):
            continue
        feature = item.get("feature")
        direction = item.get("direction")
        if isinstance(feature, str):
            signals.append(f"{feature}: {direction}" if isinstance(direction, str) else feature)
    return signals


def _dispute_statement(by_layer: dict[str, Any], source_mode: str) -> str:
    truth = by_layer.get("PROVIDER_GROUND_TRUTH")
    if truth is not None:
        if truth.status == "RESULT":
            return (
                "Provider dispute/refund outcomes were linked after scoring for evaluation only; "
                "they did not leak into this decision."
            )
        return (
            f"Provider dispute ground truth is {truth.status}; this case must not be described "
            "as confirmed fraud."
        )
    source_truth = by_layer.get("SOURCE_GROUND_TRUTH")
    if source_truth is not None:
        if source_mode != "PUBLIC_REAL_WORLD_ANONYMIZED":
            return (
                "The imported source label is evaluation-only and is not provider-verified "
                "dispute confirmation."
            )
        return (
            "The public source label is transaction-fraud ground truth, not Razorpay dispute "
            "or settlement ground truth."
        )
    return (
        f"Source mode {source_mode} returned no mature dispute ground-truth layer; "
        "the recommendation is not a confirmed-fraud finding."
    )


def _unique(values: Iterable[object]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))
