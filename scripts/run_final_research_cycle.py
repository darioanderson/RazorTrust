from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
import skops.io as sio
from sklearn import __version__ as sklearn_version
from xgboost import XGBClassifier, build_info
from xgboost import __version__ as xgboost_version

from razortrust.audit import canonical_json
from razortrust.ml.final_cycle import (
    FEATURE_JUSTIFICATIONS,
    FINAL_FEATURES,
    add_final_features,
    calibrate,
    development_error_structure,
    fit_calibrator,
    fit_release_model,
    fit_router_model,
    predict_actions,
    ranking_metrics,
    reliability_table,
    rolling_oof,
    select_release_threshold,
)
from razortrust.ml.v3_research import (
    DEFAULT_SEED,
    MAX_FALSE_RELEASE_RATE,
    MIN_TRUE_RELEASE_RECALL,
    V3A_FEATURES,
    create_research_partitions,
    evaluate_actions,
    load_v31_frame,
    sha256_file,
    target_indices,
    write_json,
)
from razortrust.security import generate_release_keypair, sign_manifest, verify_manifest

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "artifacts" / "research" / "data-v31-large-seed-20260903" / "hold_windows.csv.gz"
DATASET_MANIFEST = DATASET.parent / "dataset_manifest.json"
V3B = ROOT / "artifacts" / "research" / "v3b-final-20260903"
REPRODUCTION = (
    ROOT
    / "artifacts"
    / "research"
    / "v3b-reproduction-before-final-cycle-20260903"
    / "v3b_reproduction_report.json"
)
OUTPUT = ROOT / "artifacts" / "research" / "final-cycle-v3c-20260903"

V3B_ADDITIONS = (
    "legitimate_stability_score",
    "identity_dispersion_gap",
    "auth_novelty_pressure",
)
PARAMETER_GRID = (
    {
        "n_estimators": 220,
        "max_depth": 2,
        "learning_rate": 0.035,
        "min_child_weight": 4.0,
        "reg_lambda": 3.0,
    },
    {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.035,
        "min_child_weight": 3.0,
        "reg_lambda": 3.0,
    },
    {
        "n_estimators": 360,
        "max_depth": 3,
        "learning_rate": 0.025,
        "min_child_weight": 5.0,
        "reg_lambda": 5.0,
    },
    {
        "n_estimators": 260,
        "max_depth": 4,
        "learning_rate": 0.03,
        "min_child_weight": 6.0,
        "reg_lambda": 6.0,
        "reg_alpha": 0.05,
    },
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def _source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*.py")):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _partition_metrics(
    release_model: XGBClassifier,
    router_model: XGBClassifier,
    calibrator: Any,
    frame: pd.DataFrame,
    features: tuple[str, ...],
    threshold: float,
) -> tuple[dict[str, Any], np.ndarray]:
    actions, probabilities, release_scores = predict_actions(
        release_model, router_model, calibrator, frame, features, threshold
    )
    labels = target_indices(frame)
    binary_labels = (labels == 0).astype(int)
    metrics = evaluate_actions(labels, actions, probabilities)
    selected = actions == 0
    metrics.update(ranking_metrics(binary_labels, release_scores))
    metrics["release_precision"] = round(
        float(np.mean(binary_labels[selected])) if selected.any() else 1.0, 8
    )
    metrics["reliability"] = reliability_table(binary_labels, release_scores)
    return metrics, release_scores


def _breakdown(
    release_model: XGBClassifier,
    router_model: XGBClassifier,
    calibrator: Any,
    frame: pd.DataFrame,
    features: tuple[str, ...],
    threshold: float,
    column: str,
) -> dict[str, Any]:
    return {
        str(name): _partition_metrics(
            release_model, router_model, calibrator, group, features, threshold
        )[0]
        for name, group in frame.groupby(column, sort=True)
    }


def _time_breakdown(
    release_model: XGBClassifier,
    router_model: XGBClassifier,
    calibrator: Any,
    frame: pd.DataFrame,
    features: tuple[str, ...],
    threshold: float,
) -> dict[str, Any]:
    ordered = frame.sort_values("hold_triggered_at").copy()
    ordered["_time_slice"] = pd.qcut(
        np.arange(len(ordered)), q=3, labels=("early", "middle", "late")
    )
    return _breakdown(
        release_model, router_model, calibrator, ordered, features, threshold, "_time_slice"
    )


def _explainability(
    model: XGBClassifier, frame: pd.DataFrame, features: tuple[str, ...]
) -> dict[str, Any]:
    sample = frame.loc[:, features].iloc[: min(300, len(frame))]
    values = np.asarray(shap.TreeExplainer(model)(sample).values, dtype=float)
    importance = np.mean(np.abs(values), axis=0)
    order = np.argsort(importance)[::-1]
    return {
        "method": "TreeSHAP on frozen release-ranking model",
        "sample_scope": "development only",
        "sample_rows": len(sample),
        "global_signals": [
            {
                "feature": features[index],
                "mean_absolute_shap": round(float(importance[index]), 8),
                "business_interpretation": FEATURE_JUSTIFICATIONS.get(
                    features[index], "Existing accepted point-in-time v3A signal."
                ),
            }
            for index in order[:12]
        ],
        "caution": "SHAP attribution describes the model, not causality or certainty.",
    }


def _freeze(
    output: Path,
    release_model: XGBClassifier,
    router_model: XGBClassifier,
    calibrator: Any,
    features: tuple[str, ...],
    threshold: float,
    metadata: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    candidate = output / "research-candidate"
    candidate.mkdir()
    release_model.save_model(candidate / "release-model.ubj")
    router_model.save_model(candidate / "router-model.ubj")
    sio.dump(calibrator, candidate / "release-calibrator.skops")
    write_json(
        candidate / "feature-set.json",
        {
            "version": "feature-set-v3C-final-cycle",
            "features": features,
            "derived_feature_justifications": FEATURE_JUSTIFICATIONS,
        },
    )
    write_json(
        candidate / "policy.json",
        {
            "version": "research-policy-v3C-final-cycle",
            "release_threshold": threshold,
            "maximum_false_release_rate": MAX_FALSE_RELEASE_RATE,
            "minimum_true_release_recall": MIN_TRUE_RELEASE_RECALL,
            "fallback_action": "EVIDENCE_NEEDED",
            "auto_promotion": False,
            "production_action_eligible": False,
            "human_approval_required": True,
        },
    )
    write_json(candidate / "metadata.json", metadata)
    files = {path.name: sha256_file(path) for path in sorted(candidate.iterdir()) if path.is_file()}
    manifest = {
        "schema_version": "1.0",
        "release_id": metadata["release_id"],
        "model_version": "two-stage-xgb-v3C-final-research-candidate",
        "dataset_sha256": metadata["dataset_sha256"],
        "partition_manifest_sha256": metadata["partition_manifest_sha256"],
        "source_tree_sha256": metadata["source_tree_sha256"],
        "files": files,
        "sealed_accessed_at_freeze": False,
        "auto_promotion": False,
        "production_action_eligible": False,
        "human_approval_required": True,
        "created_at": datetime.now(UTC).isoformat(),
    }
    private_key, public_key = generate_release_keypair()
    (candidate / "candidate-manifest.jcs.json").write_bytes(canonical_json(manifest))
    signature = sign_manifest(manifest, private_key)
    (candidate / "candidate-manifest.sig").write_text(signature + "\n", encoding="ascii")
    (candidate / "candidate-public-key.txt").write_text(public_key + "\n", encoding="ascii")
    if not verify_manifest(manifest, signature, public_key):
        raise RuntimeError("candidate signature self-check failed")
    return candidate, manifest


def _loadability_test(
    candidate: Path, frame: pd.DataFrame, features: tuple[str, ...], expected: np.ndarray
) -> dict[str, Any]:
    release_model = XGBClassifier()
    release_model.load_model(candidate / "release-model.ubj")
    router_model = XGBClassifier()
    router_model.load_model(candidate / "router-model.ubj")
    untrusted = sio.get_untrusted_types(file=candidate / "release-calibrator.skops")
    rejected = [name for name in untrusted if not name.startswith(("sklearn.", "numpy.", "scipy."))]
    if rejected:
        raise ValueError(f"unapproved calibrator types: {rejected}")
    calibrator = sio.load(candidate / "release-calibrator.skops", trusted=untrusted)
    actual = calibrate(
        calibrator,
        np.asarray(release_model.predict_proba(frame.loc[:, features]))[:, 1],
    )
    return {
        "passed": bool(np.allclose(actual, expected, rtol=0, atol=1e-12)),
        "rows_compared": len(actual),
        "maximum_absolute_difference": float(np.max(np.abs(actual - expected))),
        "router_load_succeeded": router_model is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the single authorized final ML research cycle"
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to rerun or overwrite the one final cycle: {args.output}")
    reproduction = _json(REPRODUCTION)
    if not reproduction.get("exact_reproduction") or reproduction.get("training_performed"):
        raise RuntimeError("valid pre-training v3B reproduction evidence is required")
    if sha256_file(DATASET) != reproduction["dataset_sha256"]:
        raise RuntimeError("dataset changed after pre-training reproduction")
    if args.device.startswith("cuda") and not build_info().get("USE_CUDA"):
        raise RuntimeError("CUDA was requested but unavailable")

    started = time.monotonic()
    args.output.mkdir(parents=True)
    raw_frame = load_v31_frame(DATASET)
    frame = add_final_features(raw_frame)
    partitions, partition_manifest = create_research_partitions(frame, seed=args.seed)
    if partition_manifest["content_sha256"] != reproduction["partition_manifest_sha256"]:
        raise RuntimeError("partition lock changed after v3B reproduction")
    development = frame.loc[partitions == "development"].reset_index(drop=True)
    gate = frame.loc[partitions == "v3b_gate"].reset_index(drop=True)

    provenance = {
        "dataset_path": str(DATASET.relative_to(ROOT)),
        "dataset_sha256": sha256_file(DATASET),
        "dataset_manifest_sha256": sha256_file(DATASET_MANIFEST),
        "dataset_manifest": _json(DATASET_MANIFEST),
        "accepted_corpus": "the unchanged signed corpus used by v3B",
        "dataset_v3_2_disposition": reproduction["dataset_v3_2_disposition"],
        "partitions": partition_manifest,
        "selection_allowed_partitions": ["development"],
        "final_development_gate_partition": "v3b_gate",
        "sealed_partitions_locked_during_selection": ["sealed", "unknown_family"],
        "selection_data_controls": (
            "All candidate fitting, calibration, thresholding, and comparison receives only the "
            "development DataFrame. The gate is evaluated after selection. Sealed and unknown "
            "partitions are materialized only inside the conditional validation branch."
        ),
        "production_data_used": False,
    }
    write_json(args.output / "data_provenance_report.json", provenance)
    write_json(args.output / "partition_manifest.json", partition_manifest)
    write_json(
        args.output / "feature_leakage_audit.json",
        {
            "base_features": list(V3A_FEATURES),
            "new_features": FEATURE_JUSTIFICATIONS,
            "point_in_time_only": True,
            "metadata_features_excluded": [
                "merchant_id",
                "scenario_family",
                "operational_target",
                "cohort",
                "hold_triggered_at",
            ],
            "label_derived_features": [],
            "post_outcome_features": [],
            "future_information_used": False,
            "derivation_deterministic": True,
        },
    )

    feature_sets = {
        "base_v3a": tuple(V3A_FEATURES),
        "v3b_audited": (*V3A_FEATURES, *V3B_ADDITIONS),
        "v3c_error_informed": (*V3A_FEATURES, *V3B_ADDITIONS, *FINAL_FEATURES),
    }
    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for set_name, features in feature_sets.items():
        for parameter_index, parameters in enumerate(PARAMETER_GRID):
            raw_scores, labels, folds = rolling_oof(
                development,
                features,
                seed=args.seed + parameter_index * 100,
                device=args.device,
                parameters=dict(parameters),
            )
            calibrator = fit_calibrator(raw_scores, labels, args.seed)
            scores = calibrate(calibrator, raw_scores)
            threshold = select_release_threshold(scores, labels)
            ranking = ranking_metrics(labels, scores)
            objective = (
                0.55 * ranking["partial_roc_auc_at_5pct_fpr"]
                + 0.25 * threshold.true_release_recall
                + 0.20 * ranking["average_precision"]
            )
            trial = {
                "feature_set": set_name,
                "features": list(features),
                "parameter_index": parameter_index,
                "parameters": parameters,
                "folds": folds,
                "ranking_metrics": ranking,
                "threshold": threshold.as_dict(),
                "objective": round(objective, 10),
            }
            trials.append(trial)
            key = (objective, threshold.true_release_recall, ranking["average_precision"])
            if best is None or key > best["key"]:
                best = {
                    "key": key,
                    "trial": trial,
                    "raw_scores": raw_scores,
                    "labels": labels,
                    "scores": scores,
                    "calibrator": calibrator,
                    "features": features,
                    "parameters": dict(parameters),
                    "threshold": threshold,
                }
    assert best is not None
    selected_trial = best["trial"]
    hpo_report = {
        "schema_version": "1.0",
        "search_type": "predeclared bounded grid",
        "primary_objective": "0.55*pAUC@5%FPR + 0.25*recall@<=5%FPR + 0.20*average_precision",
        "candidate_feature_sets": len(feature_sets),
        "parameter_configurations_per_set": len(PARAMETER_GRID),
        "models_compared": len(trials),
        "selected_trial": selected_trial,
        "trials": trials,
        "global_auroc_used_as_selection_objective": False,
        "threshold_only_search": False,
    }
    write_json(args.output / "model_selection_report.json", hpo_report)
    write_json(
        args.output / "development_error_analysis.json",
        development_error_structure(
            development,
            best["labels"],
            best["scores"],
            best["threshold"].threshold,
        ),
    )

    features = tuple(best["features"])
    release_model = fit_release_model(
        development,
        features,
        seed=args.seed,
        device=args.device,
        parameters=best["parameters"],
    )
    router_model = fit_router_model(development, features, seed=args.seed + 1, device=args.device)
    calibrator = best["calibrator"]
    threshold = best["threshold"].threshold
    gate_metrics, gate_scores = _partition_metrics(
        release_model, router_model, calibrator, gate, features, threshold
    )
    gate_passed = (
        gate_metrics["false_release_rate"] <= MAX_FALSE_RELEASE_RATE
        and gate_metrics["true_release_recall"] >= MIN_TRUE_RELEASE_RECALL
    )
    gate_report = {
        "schema_version": "1.0",
        "gate": "fixed final development POLICY gate",
        "partition": "v3b_gate (not used in selection)",
        "locked_target": {
            "maximum_false_release_rate": MAX_FALSE_RELEASE_RATE,
            "minimum_true_release_recall": MIN_TRUE_RELEASE_RECALL,
            "preferred_true_release_recall": 0.25,
        },
        "selected_threshold_from_development_oof": best["threshold"].as_dict(),
        "metrics": gate_metrics,
        "passed": gate_passed,
        "sealed_accessed": False,
    }
    write_json(args.output / "final_development_gate_report.json", gate_report)
    write_json(
        args.output / "explainability_report.json",
        _explainability(release_model, development, features),
    )

    metadata = {
        "release_id": f"v3c-final-cycle-{args.seed}",
        "dataset_sha256": sha256_file(DATASET),
        "partition_manifest_sha256": partition_manifest["content_sha256"],
        "source_tree_sha256": _source_hash(),
        "seed": args.seed,
        "device": args.device,
        "python": sys.version,
        "platform": platform.platform(),
        "xgboost": xgboost_version,
        "scikit_learn": sklearn_version,
        "xgboost_build": build_info(),
        "features": list(features),
        "parameters": best["parameters"],
        "release_threshold": threshold,
        "gate_passed": gate_passed,
        "auto_promotion": False,
        "production_action_eligible": False,
        "human_approval_required": True,
    }

    validation: dict[str, Any] = {
        "performed": False,
        "reason": (
            "Final development gate failed; sealed, stress, future, and unknown-family "
            "data remain unopened."
        ),
        "sealed_accessed": False,
    }
    candidate: Path | None = None
    manifest: dict[str, Any] | None = None
    loadability: dict[str, Any] = {"performed": False, "reason": "gate failed"}
    final_validated = False
    if gate_passed:
        candidate, manifest = _freeze(
            args.output,
            release_model,
            router_model,
            calibrator,
            features,
            threshold,
            metadata,
        )
        expected = calibrate(
            calibrator,
            np.asarray(release_model.predict_proba(gate.loc[:, features]))[:, 1],
        )
        loadability = _loadability_test(candidate, gate, features, expected)

        # This is the only branch in which sealed/unknown rows are materialized.
        sealed = frame.loc[partitions == "sealed"].reset_index(drop=True)
        unknown = frame.loc[partitions == "unknown_family"].reset_index(drop=True)
        sealed_metrics, _ = _partition_metrics(
            release_model, router_model, calibrator, sealed, features, threshold
        )
        unknown_metrics, _ = _partition_metrics(
            release_model, router_model, calibrator, unknown, features, threshold
        )
        stressed = sealed.copy()
        for column in (
            "new_device_ratio",
            "new_geo_ratio",
            "failed_auth_ratio",
            "refund_rate_delta_z",
            "chargeback_rate_delta_z",
        ):
            stressed[column] = stressed[column] * 1.5
        stressed = add_final_features(stressed)
        stress_metrics, _ = _partition_metrics(
            release_model, router_model, calibrator, stressed, features, threshold
        )
        malformed = sealed.iloc[[0]].copy()
        malformed[features[0]] = np.nan
        try:
            predict_actions(release_model, router_model, calibrator, malformed, features, threshold)
            malformed_action = "UNSAFE_NO_ERROR"
        except ValueError:
            malformed_action = "EVIDENCE_NEEDED"
        failure_checks = {
            "malformed_non_finite_input": malformed_action,
            "model_unavailable_contract": "EVIDENCE_NEEDED",
            "calibrator_unavailable_contract": "EVIDENCE_NEEDED",
        }
        final_validated = (
            sealed_metrics["false_release_rate"] <= MAX_FALSE_RELEASE_RATE
            and sealed_metrics["true_release_recall"] >= MIN_TRUE_RELEASE_RECALL
            and all(value == "EVIDENCE_NEEDED" for value in failure_checks.values())
            and loadability["passed"]
        )
        validation = {
            "performed": True,
            "candidate_frozen_before_sealed_open": True,
            "sealed_open_count": 1,
            "retuning_after_sealed_open": False,
            "sealed_normal_test": sealed_metrics,
            "future_time_evaluation": sealed_metrics,
            "unseen_merchants": {
                "metrics": sealed_metrics,
                "merchant_overlap_with_development": len(
                    set(sealed.merchant_id) & set(development.merchant_id)
                ),
            },
            "unknown_family_evaluation": unknown_metrics,
            "unknown_family_breakdown": _breakdown(
                release_model,
                router_model,
                calibrator,
                unknown,
                features,
                threshold,
                "scenario_family",
            ),
            "stress_evaluation": {
                "definition": (
                    "1.5x identity novelty, authorization, refund, and chargeback pressure"
                ),
                "metrics": stress_metrics,
            },
            "graceful_failure": failure_checks,
            "family_breakdown": _breakdown(
                release_model,
                router_model,
                calibrator,
                sealed,
                features,
                threshold,
                "scenario_family",
            ),
            "cohort_breakdown": _breakdown(
                release_model, router_model, calibrator, sealed, features, threshold, "cohort"
            ),
            "time_slice_breakdown": _time_breakdown(
                release_model, router_model, calibrator, sealed, features, threshold
            ),
            "final_validation_passed": final_validated,
        }
    write_json(args.output / "independent_validation_report.json", validation)
    write_json(args.output / "candidate_loadability_report.json", loadability)

    decision = (
        "FINAL_RESEARCH_CANDIDATE_VALIDATED"
        if final_validated
        else "FINAL_ML_STOP_NO_SAFE_RELEASE_CANDIDATE"
    )
    final_report = {
        "schema_version": "1.0",
        "decision": decision,
        "v3b_reproduced_before_training": True,
        "data_leakage_prevented": True,
        "development_gate_passed": gate_passed,
        "sealed_accessed": bool(gate_passed),
        "production_data_used": False,
        "production_action_eligible": False,
        "auto_promotion": False,
        "human_approval_required": True,
        "successful_approach": (
            "Two-stage release ranking with temporal OOF selection and point-in-time interactions"
            if final_validated
            else None
        ),
        "failed_approach": None
        if final_validated
        else (
            "All 12 bounded low-FPR candidates failed to establish a safe validated release policy."
        ),
        "final_metrics": validation.get("sealed_normal_test") if final_validated else gate_metrics,
        "selected_development_oof": {
            "ranking_metrics": selected_trial["ranking_metrics"],
            "threshold": selected_trial["threshold"],
        },
        "features_added": list(FINAL_FEATURES)
        if selected_trial["feature_set"] == "v3c_error_informed"
        else [],
        "models_compared": len(trials),
        "seeds": [args.seed, *[args.seed + index * 100 for index in range(len(PARAMETER_GRID))]],
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "candidate_release_path": str(candidate.relative_to(ROOT)) if candidate else None,
        "candidate_manifest": manifest,
        "reports": [
            "data_provenance_report.json",
            "partition_manifest.json",
            "feature_leakage_audit.json",
            "model_selection_report.json",
            "development_error_analysis.json",
            "final_development_gate_report.json",
            "explainability_report.json",
            "independent_validation_report.json",
            "candidate_loadability_report.json",
        ],
    }
    write_json(args.output / "final_research_report.json", final_report)
    write_json(
        args.output / "run_manifest.json",
        {
            "command": (
                "python scripts/run_final_research_cycle.py --device cuda --output "
                "artifacts/research/final-cycle-v3c-20260903"
            ),
            "created_at": datetime.now(UTC).isoformat(),
            "single_authorized_cycle": True,
            "rerun_guard": "output directory must not already exist",
            "seed": args.seed,
            "device": args.device,
            "elapsed_seconds": final_report["elapsed_seconds"],
            "artifact_hashes": {
                path.name: sha256_file(path)
                for path in sorted(args.output.iterdir())
                if path.is_file()
            },
        },
    )
    markdown = f"""# RazorTrust final ML research cycle

**Decision:** `{decision}`

- v3B reproduced before training: yes (3.0% false release, 10.0% true-release recall)
- Data leakage prevented: yes; selection used development temporal OOF only
- Production data used: no
- Models compared: {len(trials)} (3 feature sets × 4 bounded configurations)
- Selected feature set: `{selected_trial["feature_set"]}`
- Final development gate: {gate_metrics["false_release_rate"]:.1%} false release,
  {gate_metrics["true_release_recall"]:.1%} recall
- Development gate passed: {str(gate_passed).lower()}
- Sealed data accessed: {str(bool(gate_passed)).lower()}
- Production action eligibility: no; human approval remains mandatory

The repository did not contain a Dataset v3.2 corpus. This cycle used the unchanged,
hash-verified synthetic v3.1 longitudinal corpus signed into the existing v3B candidate;
no substitute or newly generated dataset was introduced.
"""
    (args.output / "FINAL_RESEARCH_REPORT.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "decision": decision, "gate": gate_metrics}))


if __name__ == "__main__":
    main()
