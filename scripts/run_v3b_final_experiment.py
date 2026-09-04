from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "work" / "matplotlib"),
)

import numpy as np
import optuna
import pandas as pd
import shap
import skops.io as sio
from sklearn import __version__ as sklearn_version
from xgboost import XGBClassifier, build_info
from xgboost import __version__ as xgboost_version

from razortrust.audit import canonical_json
from razortrust.ml.v3_research import (
    DEFAULT_SEED,
    MAX_FALSE_RELEASE_RATE,
    MIN_TRUE_RELEASE_RECALL,
    V3A_FEATURES,
    V3B_CANDIDATES,
    add_v3b_candidates,
    calibrate_probabilities,
    create_research_partitions,
    cross_validated_probabilities,
    evaluate_actions,
    fail_closed_action,
    fit_classifier,
    fit_probability_calibrator,
    load_v31_frame,
    predict_probabilities,
    select_thresholds,
    sha256_file,
    target_indices,
    threshold_actions,
    write_json,
)
from razortrust.security import generate_release_keypair, sign_manifest, verify_manifest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    ROOT / "artifacts" / "research" / "data-v31-large-seed-20260903" / "hold_windows.csv.gz"
)
DEFAULT_AUDIT = (
    ROOT / "artifacts" / "research" / "v3a-error-audit-20260903" / "v3a_error_audit.json"
)


def _trial_parameters(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 180, 500, step=40),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.2, 8.0, log=True),
    }


def _hpo(
    development: pd.DataFrame,
    features: tuple[str, ...],
    *,
    seed: int,
    device: str,
    trials: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = target_indices(development)

    def objective(trial: optuna.Trial) -> float:
        parameters = _trial_parameters(trial)
        raw = cross_validated_probabilities(
            development,
            features,
            seed=seed,
            device=device,
            parameters=parameters,
        )
        calibrator = fit_probability_calibrator(raw, labels, seed=seed)
        calibrated = calibrate_probabilities(calibrator, raw)
        try:
            selection = select_thresholds(calibrated, labels)
        except ValueError:
            return 1_000_000.0
        trial.set_user_attr("thresholds", selection.as_dict())
        return selection.expected_cost_units

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    summaries = [
        {
            "number": trial.number,
            "value": trial.value,
            "state": trial.state.name,
            "parameters": trial.params,
            "thresholds": trial.user_attrs.get("thresholds"),
        }
        for trial in study.trials
    ]
    return dict(study.best_trial.params), summaries


def _evaluate_partition(
    estimator: XGBClassifier,
    calibrator: Any,
    frame: pd.DataFrame,
    features: tuple[str, ...],
    release_threshold: float,
    escalate_threshold: float,
) -> tuple[dict[str, Any], np.ndarray]:
    probabilities = predict_probabilities(estimator, calibrator, frame, features)
    labels = target_indices(frame)
    actions = threshold_actions(probabilities, release_threshold, escalate_threshold)
    return evaluate_actions(labels, actions, probabilities), probabilities


def _family_metrics(
    estimator: XGBClassifier,
    calibrator: Any,
    frame: pd.DataFrame,
    features: tuple[str, ...],
    release_threshold: float,
    escalate_threshold: float,
) -> dict[str, Any]:
    result = {}
    for family, group in frame.groupby("scenario_family", sort=True):
        metrics, _ = _evaluate_partition(
            estimator,
            calibrator,
            group,
            features,
            release_threshold,
            escalate_threshold,
        )
        result[str(family)] = metrics
    return result


def _explainability(
    estimator: XGBClassifier,
    frame: pd.DataFrame,
    features: tuple[str, ...],
) -> dict[str, Any]:
    sample = frame.loc[:, features].iloc[: min(300, len(frame))]
    values = shap.TreeExplainer(estimator)(sample).values
    array = np.asarray(values, dtype=float)
    if array.ndim == 3:
        importance = np.mean(np.abs(array), axis=(0, 2))
    else:
        importance = np.mean(np.abs(array), axis=0)
    ordering = np.argsort(importance)[::-1]
    return {
        "method": "TreeSHAP",
        "sample_rows": len(sample),
        "global_signals": [
            {
                "feature": features[index],
                "mean_absolute_shap": round(float(importance[index]), 8),
                "plain_english": _plain_english(features[index]),
            }
            for index in ordering[:10]
        ],
        "uncertainty_note": (
            "Calibrated class probabilities describe model uncertainty; they are not "
            "a guarantee of correctness. Missing or invalid evidence fails closed."
        ),
    }


def _plain_english(feature: str) -> str:
    phrases = {
        "legitimate_stability_score": "Operational behavior is stable and lacks harm indicators.",
        "identity_dispersion_gap": "New-device activity is compared with device diversity.",
        "auth_novelty_pressure": "Authorization failures coincide with new identity signals.",
        "refund_rate_delta_z": "Refund activity differs from the merchant baseline.",
        "chargeback_rate_delta_z": "Chargeback activity differs from the merchant baseline.",
        "failed_auth_ratio": "A share of payment attempts failed authorization.",
        "new_device_ratio": "Payments arrived from devices unseen in the baseline.",
        "new_geo_ratio": "Payments arrived from geographies unseen in the baseline.",
        "volume_delta_z": "Payment volume differs from the merchant baseline.",
        "gmv_delta_z": "Payment value differs from the merchant baseline.",
    }
    return phrases.get(feature, "This behavior differs from the merchant's historical baseline.")


def _source_tree_sha256(source: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(source.rglob("*.py"))
    for path in paths:
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _freeze_candidate(
    output: Path,
    estimator: XGBClassifier,
    calibrator: Any,
    features: tuple[str, ...],
    reports: list[Path],
    metadata: dict[str, Any],
) -> None:
    release = output / "research-candidate"
    release.mkdir(parents=True, exist_ok=False)
    estimator.save_model(release / "xgboost.ubj")
    sio.dump(calibrator, release / "calibrator.skops")
    write_json(release / "feature-set.json", {"version": "feature-set-v3B", "features": features})
    write_json(
        release / "policy.json",
        {
            "version": "research-policy-v3B",
            "maximum_false_release_rate": MAX_FALSE_RELEASE_RATE,
            "minimum_true_release_recall": MIN_TRUE_RELEASE_RECALL,
            "auto_promotion": False,
            "real_action_requires_human_approval": True,
        },
    )
    write_json(release / "metadata.json", metadata)
    copied_reports = []
    for report in reports:
        destination = release / report.name
        destination.write_bytes(report.read_bytes())
        copied_reports.append(destination)
    artifact_paths = sorted(
        [path for path in release.iterdir() if path.is_file()], key=lambda path: path.name
    )
    manifest = {
        "schema_version": "1.0",
        "release_id": metadata["release_id"],
        "model_version": "xgb-settlement-v3B-research-candidate",
        "feature_set_version": "feature-set-v3B",
        "policy_version": "research-policy-v3B",
        "dataset_sha256": metadata["dataset_sha256"],
        "partition_manifest_sha256": metadata["partition_manifest_sha256"],
        "source_tree_sha256": metadata["source_tree_sha256"],
        "files": {path.name: sha256_file(path) for path in artifact_paths},
        "auto_promotion": False,
        "human_approval_required": True,
        "signer_key_id": "ephemeral-research-demo-key-20260903",
        "created_at": datetime.now(UTC).isoformat(),
    }
    private_key, public_key = generate_release_keypair()
    (release / "candidate-manifest.jcs.json").write_bytes(canonical_json(manifest))
    signature = sign_manifest(manifest, private_key)
    (release / "candidate-manifest.sig").write_text(signature + "\n", encoding="ascii")
    (release / "candidate-public-key.txt").write_text(public_key + "\n", encoding="ascii")
    if not verify_manifest(manifest, signature, public_key):
        raise RuntimeError("candidate signature self-verification failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the one final feature-set v3B experiment")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite final experiment: {args.output}")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    recommended = tuple(audit.get("recommended_v3b_features", []))
    if not recommended or not set(recommended).issubset(V3B_CANDIDATES):
        raise ValueError("v3B features must be the non-empty, audited candidate subset")
    dataset_sha256 = sha256_file(args.dataset)
    if audit.get("dataset_sha256") != dataset_sha256:
        raise ValueError("v3A audit and v3B dataset hashes do not match")

    frame = add_v3b_candidates(load_v31_frame(args.dataset))
    partition, partition_manifest = create_research_partitions(frame, seed=args.seed)
    if audit.get("partition_manifest_sha256") != partition_manifest["content_sha256"]:
        raise ValueError("v3A and v3B partition manifests do not match")
    development = frame.loc[partition == "development"].reset_index(drop=True)
    gate = frame.loc[partition == "v3b_gate"].reset_index(drop=True)
    features = (*V3A_FEATURES, *recommended)

    best_parameters, trials = _hpo(
        development,
        features,
        seed=args.seed,
        device=args.device,
        trials=args.trials,
    )
    development_labels = target_indices(development)
    development_raw = cross_validated_probabilities(
        development,
        features,
        seed=args.seed,
        device=args.device,
        parameters=best_parameters,
    )
    calibrator = fit_probability_calibrator(development_raw, development_labels, seed=args.seed)
    development_probabilities = calibrate_probabilities(calibrator, development_raw)
    thresholds = select_thresholds(development_probabilities, development_labels)
    estimator = fit_classifier(
        development,
        features,
        seed=args.seed,
        device=args.device,
        parameters=best_parameters,
    )
    gate_metrics, _ = _evaluate_partition(
        estimator,
        calibrator,
        gate,
        features,
        thresholds.release_threshold,
        thresholds.escalate_threshold,
    )
    passed = (
        gate_metrics["false_release_rate"] <= MAX_FALSE_RELEASE_RATE
        and gate_metrics["true_release_recall"] >= MIN_TRUE_RELEASE_RECALL
    )

    args.output.mkdir(parents=True)
    write_json(args.output / "partition_manifest.json", partition_manifest)
    hpo_report = {
        "schema_version": "1.0",
        "device": args.device,
        "xgboost_build": build_info(),
        "trial_count": args.trials,
        "best_parameters": best_parameters,
        "trials": trials,
    }
    write_json(args.output / "hpo_report.json", hpo_report)
    gate_report = {
        "schema_version": "1.0",
        "experiment": "feature_set_v3B_final_gate",
        "features": list(features),
        "audit_justified_additions": list(recommended),
        "locked_target": {
            "maximum_false_release_rate": MAX_FALSE_RELEASE_RATE,
            "minimum_true_release_recall": MIN_TRUE_RELEASE_RECALL,
        },
        "thresholds_selected_on_development_oof": thresholds.as_dict(),
        "gate_metrics": gate_metrics,
        "passed": passed,
        "stopping_decision": "CONTINUE_TO_INDEPENDENT_VALIDATION" if passed else "STOP_ML",
        "sealed_accessed": False,
    }
    write_json(args.output / "v3b_gate_report.json", gate_report)

    validation_report: dict[str, Any] = {
        "performed": False,
        "reason": (
            "v3B failed its fixed gate; sealed, stress, and unknown-family data remain untouched."
        ),
    }
    explainability: dict[str, Any] = {"performed": False, "reason": "v3B gate failed"}
    if passed:
        sealed = frame.loc[partition == "sealed"].reset_index(drop=True)
        unknown = frame.loc[partition == "unknown_family"].reset_index(drop=True)
        sealed_metrics, _ = _evaluate_partition(
            estimator,
            calibrator,
            sealed,
            features,
            thresholds.release_threshold,
            thresholds.escalate_threshold,
        )
        unknown_metrics, _ = _evaluate_partition(
            estimator,
            calibrator,
            unknown,
            features,
            thresholds.release_threshold,
            thresholds.escalate_threshold,
        )
        malformed = sealed.iloc[[0]].copy()
        malformed.loc[:, features[0]] = np.nan
        failure_checks = {
            "model_unavailable": fail_closed_action(
                None, calibrator, sealed.iloc[[0]], features, thresholds
            )[0],
            "calibrator_unavailable": fail_closed_action(
                estimator, None, sealed.iloc[[0]], features, thresholds
            )[0],
            "malformed_non_finite_input": fail_closed_action(
                estimator, calibrator, malformed, features, thresholds
            )[0],
        }
        validation_report = {
            "performed": True,
            "sealed_normal_test": sealed_metrics,
            "future_time_evaluation": {
                "metrics": sealed_metrics,
                "chronological_within_each_known_family": True,
            },
            "unseen_merchants": {
                "metrics": sealed_metrics,
                "merchant_overlap_with_development": len(
                    set(sealed.merchant_id) & set(development.merchant_id)
                ),
            },
            "unknown_family_evaluation": unknown_metrics,
            "unknown_family_breakdown": _family_metrics(
                estimator,
                calibrator,
                unknown,
                features,
                thresholds.release_threshold,
                thresholds.escalate_threshold,
            ),
            "stress_and_graceful_failure": failure_checks,
            "all_failure_checks_fail_closed": all(
                action == "EVIDENCE_NEEDED" for action in failure_checks.values()
            ),
            "calibration": {
                "method": "multinomial_logit_on_development_oof_probabilities",
                "sealed_log_loss": sealed_metrics["log_loss"],
                "sealed_multiclass_brier": sealed_metrics["multiclass_brier"],
                "sealed_expected_calibration_error": sealed_metrics["expected_calibration_error"],
            },
            "economic_cost": {
                "units": "NORMALIZED_COST_UNITS",
                "sealed_expected_cost": sealed_metrics["expected_cost_units"],
                "unknown_family_expected_cost": unknown_metrics["expected_cost_units"],
            },
        }
        explainability = _explainability(estimator, sealed, features)
    write_json(args.output / "independent_validation_report.json", validation_report)
    write_json(args.output / "explainability_report.json", explainability)

    metadata = {
        "release_id": f"v3b-research-{args.seed}",
        "dataset_sha256": dataset_sha256,
        "partition_manifest_sha256": partition_manifest["content_sha256"],
        "source_tree_sha256": _source_tree_sha256(ROOT / "src"),
        "stopping_decision": gate_report["stopping_decision"],
        "gate_passed": passed,
        "device": args.device,
        "seed": args.seed,
        "python": sys.version,
        "platform": platform.platform(),
        "xgboost": xgboost_version,
        "scikit_learn": sklearn_version,
        "command": (
            "python scripts/run_v3b_final_experiment.py --device cuda "
            f"--trials {args.trials} --seed {args.seed} --output <NEW_OUTPUT_DIR>"
        ),
        "auto_promotion": False,
        "production_action_eligible": False,
        "human_approval_required": True,
    }
    write_json(args.output / "reproducibility_report.json", metadata)
    reports = [
        args.output / "partition_manifest.json",
        args.output / "hpo_report.json",
        args.output / "v3b_gate_report.json",
        args.output / "independent_validation_report.json",
        args.output / "explainability_report.json",
        args.output / "reproducibility_report.json",
        args.audit,
    ]
    _freeze_candidate(args.output, estimator, calibrator, features, reports, metadata)
    print(json.dumps({"output": str(args.output), "passed": passed, "gate": gate_metrics}))


if __name__ == "__main__":
    main()
