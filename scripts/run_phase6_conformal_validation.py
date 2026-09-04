from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from razortrust.ml.conformal_validation import (
    label_indices,
    shift_model_matrix,
    summarize_prediction_sets,
)
from razortrust.ml.dataset import build_training_frame, model_matrix
from razortrust.ml.modeling import train_model_bundle
from razortrust.ml.monitoring import batch_drift_report
from razortrust.ml.splits import create_split_manifest
from razortrust.ml.uncertainty import ApsConformalAbstainer
from razortrust.synthetic import generate_dataset


def _parse_float_list(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one float")
    return parsed


def _parse_str_list(value: str) -> list[str]:
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one feature")
    return parsed


def _frame_hash(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _build_frame(
    *,
    seed: int,
    merchants_per_family: int,
    transactions_per_merchant: int,
) -> pd.DataFrame:
    merchants, transactions, holds = generate_dataset(
        seed=seed,
        merchants_per_family=merchants_per_family,
        transactions_per_merchant=transactions_per_merchant,
    )
    frame = build_training_frame(merchants, transactions, holds).reset_index(drop=True)
    labels = label_indices(frame)
    if set(np.unique(labels).tolist()) != {0, 1, 2}:
        raise RuntimeError(
            f"seed {seed} does not contain all three operational classes"
        )
    return frame


def _coverage_gate(
    report: dict[str, object],
    confidence_level: float,
) -> dict[str, object]:
    interval = report["coverage_wilson_95"]
    routing = report["routing"]
    target_within_interval = (
        float(interval["lower"])
        <= confidence_level
        <= float(interval["upper"])
    )
    routing_invariants = (
        int(routing["ambiguous_release_invariant_violations"]) == 0
        and int(routing["empty_release_invariant_violations"]) == 0
        and int(routing["full_release_invariant_violations"]) == 0
    )
    return {
        "nominal_target_within_empirical_wilson_95": target_within_interval,
        "conservative_routing_invariants_hold": routing_invariants,
        "pass": target_within_interval and routing_invariants,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 6 fresh synthetic conformal validation comparing the current "
            "cross-fitted-probability style with strict fixed-predictor split conformal"
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-seed", type=int, default=20260920)
    parser.add_argument("--conformal-seed", type=int, default=20260921)
    parser.add_argument("--test-seed", type=int, default=20260922)
    parser.add_argument("--merchants-per-family", type=int, default=40)
    parser.add_argument("--transactions-per-merchant", type=int, default=80)
    parser.add_argument("--estimators", type=int, default=300)
    parser.add_argument("--confidence-level", type=float, default=0.90)
    parser.add_argument(
        "--shift-features",
        type=_parse_str_list,
        default=["volume_delta_z", "gmv_delta_z", "ticket_size_delta_z"],
    )
    parser.add_argument(
        "--shift-strengths",
        type=_parse_float_list,
        default=[0.50, 1.00],
    )
    args = parser.parse_args()

    seeds = {
        args.development_seed,
        args.conformal_seed,
        args.test_seed,
    }
    if len(seeds) != 3:
        raise SystemExit("development, conformal, and test seeds must be distinct")
    if not 0.5 < args.confidence_level < 1.0:
        raise SystemExit("confidence level must be between 0.5 and 1.0")

    development = _build_frame(
        seed=args.development_seed,
        merchants_per_family=args.merchants_per_family,
        transactions_per_merchant=args.transactions_per_merchant,
    )
    conformal_frame = _build_frame(
        seed=args.conformal_seed,
        merchants_per_family=args.merchants_per_family,
        transactions_per_merchant=args.transactions_per_merchant,
    )
    test_frame = _build_frame(
        seed=args.test_seed,
        merchants_per_family=args.merchants_per_family,
        transactions_per_merchant=args.transactions_per_merchant,
    )

    development_hash = _frame_hash(development)
    conformal_hash = _frame_hash(conformal_frame)
    test_hash = _frame_hash(test_frame)

    split_manifest = create_split_manifest(
        development,
        development_hash,
        seed=args.development_seed,
    )
    result = train_model_bundle(
        development,
        split_manifest,
        seed=args.development_seed,
        n_estimators=args.estimators,
    )

    conformal_matrix = model_matrix(conformal_frame)
    test_matrix = model_matrix(test_frame)
    conformal_labels = label_indices(conformal_frame)
    test_labels = label_indices(test_frame)

    fixed_predictor_conformal_probabilities = result.bundle.predict_proba(
        conformal_matrix,
        conformal_frame["cohort"],
    )
    test_probabilities = result.bundle.predict_proba(
        test_matrix,
        test_frame["cohort"],
    )

    strict_split = ApsConformalAbstainer(
        confidence_level=args.confidence_level,
        seed=args.development_seed + 100,
    ).conformalize(
        fixed_predictor_conformal_probabilities,
        conformal_labels,
    )
    strict_report = summarize_prediction_sets(
        strict_split,
        test_probabilities,
        test_labels,
    )

    legacy_style = ApsConformalAbstainer(
        confidence_level=args.confidence_level,
        seed=args.development_seed + 200,
    ).conformalize(
        result.policy_probabilities,
        result.policy_labels,
    )
    legacy_report = summarize_prediction_sets(
        legacy_style,
        test_probabilities,
        test_labels,
    )

    strict_gate = _coverage_gate(strict_report, args.confidence_level)

    shifted_results: list[dict[str, object]] = []
    for strength in args.shift_strengths:
        shifted_matrix = shift_model_matrix(
            test_matrix,
            reference=conformal_matrix,
            features=args.shift_features,
            strength_reference_std=float(strength),
        )
        shifted_probabilities = result.bundle.predict_proba(
            shifted_matrix,
            test_frame["cohort"],
        )
        shifted_report = summarize_prediction_sets(
            strict_split,
            shifted_probabilities,
            test_labels,
        )
        feature_drift = batch_drift_report(conformal_matrix, shifted_matrix)

        shifted_results.append(
            {
                "strength_reference_std": float(strength),
                "features": list(args.shift_features),
                "feature_drift": feature_drift.model_dump(mode="json"),
                "coverage_and_routing": shifted_report,
                "coverage_delta_vs_unshifted": round(
                    float(shifted_report["empirical_coverage"])
                    - float(strict_report["empirical_coverage"]),
                    8,
                ),
                "average_set_size_delta_vs_unshifted": round(
                    float(shifted_report["average_set_size"])
                    - float(strict_report["average_set_size"]),
                    8,
                ),
                "ambiguous_set_rate_delta_vs_unshifted": round(
                    float(shifted_report["ambiguous_set_rate"])
                    - float(strict_report["ambiguous_set_rate"]),
                    8,
                ),
                "coverage_guarantee_claimed_under_shift": False,
            }
        )

    decision = (
        "PHASE6_RETAIN_CONFORMAL_RESEARCH_VALUE"
        if strict_gate["pass"]
        else "PHASE6_BLOCKED_BY_BASELINE_CONFORMAL_VALIDATION"
    )

    report = {
        "schema_version": "1.0",
        "phase": 6,
        "benchmark_label": "SYNTHETIC / RESEARCH CONFORMAL VALIDATION",
        "decision": decision,
        "confidence_level": args.confidence_level,
        "development": {
            "seed": args.development_seed,
            "rows": len(development),
            "frame_sha256": development_hash,
            "purpose": (
                "model training + model probability calibration + cost-policy "
                "selection inside existing RazorTrust development pipeline"
            ),
        },
        "strict_split_conformalization": {
            "seed": args.conformal_seed,
            "rows": len(conformal_frame),
            "frame_sha256": conformal_hash,
            "predictor_semantics": (
                "same frozen final ModelBundle generates probabilities for "
                "conformalization and research test"
            ),
        },
        "research_test": {
            "seed": args.test_seed,
            "rows": len(test_frame),
            "frame_sha256": test_hash,
            "sealed_test_used": False,
        },
        "baseline_comparison": {
            "strict_fixed_predictor_split_conformal": strict_report,
            "legacy_cross_fitted_probability_style_diagnostic": legacy_report,
            "legacy_style_formal_split_guarantee_claimed": False,
            "methodology_note": (
                "Legacy-style scores use cross-fitted policy probabilities, while "
                "test probabilities use the final calibrator refit on the full "
                "development calibration partition. This comparison is diagnostic."
            ),
        },
        "pre_registered_baseline_gate": {
            "rule": (
                "90% nominal target must fall inside the empirical Wilson 95% "
                "coverage interval, and ambiguous/empty/full sets must never "
                "route to RELEASE"
            ),
            "result": strict_gate,
        },
        "class_coverage_semantics": (
            "class-specific coverage is diagnostic only; standard APS targets "
            "marginal coverage rather than class-conditional coverage"
        ),
        "controlled_covariate_shift": {
            "features": list(args.shift_features),
            "strengths_reference_std": list(args.shift_strengths),
            "results": shifted_results,
        },
        "safety": {
            "production_action_eligible": False,
            "serving_change_authorized": False,
            "automatic_release_enabled": False,
            "automatic_promotion": False,
            "automatic_retraining": False,
            "sealed_test_used": False,
            "stress_set_used": False,
            "champion_remains": "xgb-if-settlement@2",
            "active_enforcement_runtime": "human-only@1",
            "release_is_system_recommendation_only": True,
        },
        "limitations": [
            "Fresh synthetic mechanics data is not Razorpay settlement-hold ground truth.",
            "Standard APS provides marginal rather than class-conditional coverage.",
            "Controlled feature shifts are robustness probes, not real fraud mechanisms.",
            "Coverage guarantees are not claimed after detected distribution shift.",
            "This research runner does not modify the production conformal implementation.",
        ],
    }

    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "phase6_conformal_validation_report.json"
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
