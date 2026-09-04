from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from razortrust.ml.conformal_v11_validation import (
    corrected_v11_gate,
    exact_undercoverage_test,
)
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
    row_hashes = pd.util.hash_pandas_object(
        frame,
        index=True,
    ).to_numpy(dtype=np.uint64)
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    frame = build_training_frame(
        merchants,
        transactions,
        holds,
    ).reset_index(drop=True)

    labels = label_indices(frame)
    if set(np.unique(labels).tolist()) != {0, 1, 2}:
        raise RuntimeError(
            f"seed {seed} does not contain all three operational classes"
        )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 6 V1.1 fresh split-conformal confirmation with a corrected "
            "one-sided undercoverage gate"
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-v1-report", type=Path, required=True)
    parser.add_argument("--development-seed", type=int, default=20260923)
    parser.add_argument("--conformal-seed", type=int, default=20260924)
    parser.add_argument("--test-seed", type=int, default=20260925)
    parser.add_argument("--merchants-per-family", type=int, default=40)
    parser.add_argument("--transactions-per-merchant", type=int, default=80)
    parser.add_argument("--estimators", type=int, default=300)
    parser.add_argument("--confidence-level", type=float, default=0.90)
    parser.add_argument("--undercoverage-alpha", type=float, default=0.05)
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
        raise SystemExit(
            "development, conformal, and test seeds must be distinct"
        )
    if not 0.5 < args.confidence_level < 1.0:
        raise SystemExit(
            "confidence level must be between 0.5 and 1.0"
        )
    if not 0.0 < args.undercoverage_alpha < 0.5:
        raise SystemExit(
            "undercoverage alpha must be between 0 and 0.5"
        )

    if not args.prior_v1_report.exists():
        raise SystemExit(
            f"prior V1 report not found: {args.prior_v1_report}"
        )
    prior_v1 = json.loads(
        args.prior_v1_report.read_text(encoding="utf-8")
    )
    if prior_v1.get("decision") != (
        "PHASE6_BLOCKED_BY_BASELINE_CONFORMAL_VALIDATION"
    ):
        raise SystemExit(
            "prior V1 report does not contain the expected frozen blocked decision"
        )
    if prior_v1.get("safety", {}).get("sealed_test_used") is not False:
        raise SystemExit(
            "prior V1 report does not prove SEALED TEST remained unused"
        )

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

    conformal_probabilities = result.bundle.predict_proba(
        conformal_matrix,
        conformal_frame["cohort"],
    )
    test_probabilities = result.bundle.predict_proba(
        test_matrix,
        test_frame["cohort"],
    )

    abstainer = ApsConformalAbstainer(
        confidence_level=args.confidence_level,
        seed=args.development_seed + 100,
    ).conformalize(
        conformal_probabilities,
        conformal_labels,
    )

    baseline = summarize_prediction_sets(
        abstainer,
        test_probabilities,
        test_labels,
    )
    undercoverage = exact_undercoverage_test(
        abstainer,
        test_probabilities,
        test_labels,
        target_coverage=args.confidence_level,
        significance_level=args.undercoverage_alpha,
    )
    gate = corrected_v11_gate(
        empirical_coverage=float(baseline["empirical_coverage"]),
        target_coverage=args.confidence_level,
        undercoverage_test=undercoverage,
        routing=baseline["routing"],
    )

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
        shifted_summary = summarize_prediction_sets(
            abstainer,
            shifted_probabilities,
            test_labels,
        )
        shifted_undercoverage = exact_undercoverage_test(
            abstainer,
            shifted_probabilities,
            test_labels,
            target_coverage=args.confidence_level,
            significance_level=args.undercoverage_alpha,
        )
        feature_drift = batch_drift_report(
            conformal_matrix,
            shifted_matrix,
        )

        shifted_results.append(
            {
                "strength_reference_std": float(strength),
                "features": list(args.shift_features),
                "feature_drift": feature_drift.model_dump(mode="json"),
                "coverage_and_routing": shifted_summary,
                "undercoverage_diagnostic": shifted_undercoverage,
                "coverage_delta_vs_unshifted": round(
                    float(shifted_summary["empirical_coverage"])
                    - float(baseline["empirical_coverage"]),
                    8,
                ),
                "average_set_size_delta_vs_unshifted": round(
                    float(shifted_summary["average_set_size"])
                    - float(baseline["average_set_size"]),
                    8,
                ),
                "coverage_guarantee_claimed_under_shift": False,
            }
        )

    decision = (
        "PHASE6_V11_RETAIN_CONFORMAL_RESEARCH_VALUE"
        if gate["pass"]
        else "PHASE6_V11_BLOCKED_BY_FRESH_UNDERCOVERAGE_GATE"
    )

    report = {
        "schema_version": "1.1",
        "phase": "6-V1.1",
        "benchmark_label": (
            "SYNTHETIC / RESEARCH FRESH CONFORMAL CONFIRMATION"
        ),
        "decision": decision,
        "confidence_level": args.confidence_level,
        "prior_v1_frozen_evidence": {
            "path": str(args.prior_v1_report),
            "sha256": _file_sha256(args.prior_v1_report),
            "decision": prior_v1["decision"],
            "v1_gate_rewritten": False,
        },
        "fresh_seeds": {
            "development": args.development_seed,
            "conformalization": args.conformal_seed,
            "research_test": args.test_seed,
        },
        "data_isolation": {
            "development_frame_sha256": development_hash,
            "conformal_frame_sha256": conformal_hash,
            "research_test_frame_sha256": test_hash,
            "development_rows": len(development),
            "conformal_rows": len(conformal_frame),
            "research_test_rows": len(test_frame),
            "same_fixed_predictor_for_conformal_and_test_probabilities": True,
            "sealed_test_used": False,
            "stress_set_used": False,
        },
        "baseline": baseline,
        "undercoverage_test": undercoverage,
        "pre_registered_v11_gate": {
            "rule": [
                "empirical marginal coverage >= 0.90",
                (
                    "one-sided exact binomial test does not find significant "
                    "evidence of coverage < 0.90 at alpha=0.05"
                ),
                "ambiguous prediction set routes to RELEASE = 0",
                "empty prediction set routes to RELEASE = 0",
                "full prediction set routes to RELEASE = 0",
                "SEALED TEST remains unused",
                "no production promotion",
            ],
            "result": gate,
        },
        "class_coverage_semantics": (
            "class-specific coverage remains diagnostic only; standard APS "
            "targets marginal rather than class-conditional coverage"
        ),
        "controlled_covariate_shift": {
            "features": list(args.shift_features),
            "strengths_reference_std": list(args.shift_strengths),
            "results": shifted_results,
            "coverage_guarantee_claimed_under_detected_shift": False,
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
            (
                "Fresh synthetic mechanics data is not Razorpay "
                "settlement-hold ground truth."
            ),
            (
                "The exact binomial check is a validation diagnostic and "
                "does not replace conformal finite-sample theory."
            ),
            (
                "Standard APS targets marginal rather than "
                "class-conditional coverage."
            ),
            (
                "Controlled covariate shifts are robustness probes, "
                "not real fraud mechanisms."
            ),
            (
                "Coverage guarantees are not claimed after detected "
                "distribution shift."
            ),
        ],
    }

    args.output.mkdir(parents=True, exist_ok=True)
    report_path = (
        args.output / "phase6_v11_fresh_confirmation_report.json"
    )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
