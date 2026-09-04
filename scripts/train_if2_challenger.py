from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Ensure direct script execution uses the current mounted src-layout tree.
# pytest already injects src via pyproject.toml, but `python scripts/...` does not.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from razortrust.ml.dataset import build_training_frame
from razortrust.ml.evaluation import evaluate_sealed_test
from razortrust.ml.modeling import train_model_bundle
from razortrust.ml.release import save_model_release
from razortrust.ml.runtime import TrainedRiskModel
from razortrust.ml.splits import create_split_manifest, write_split_manifest
from razortrust.security import generate_release_keypair
from razortrust.shadow_scoring import fixture_feature_vector
from razortrust.synthetic import generate_dataset, write_dataset

# These are the exact best Optuna parameters printed by the original
# xgb-if-settlement@2 mechanics training run. Reusing them isolates this
# challenger primarily to the new merchant-group OOF Isolation Forest reference
# semantics rather than silently changing the XGBoost hyper-parameters too.
CHAMPION_XGB_PARAMS: dict[str, Any] = {
    "max_depth": 7,
    "learning_rate": 0.08551063117612584,
    "min_child_weight": 4.5010685388160665,
    "gamma": 1.0310241792544639,
    "subsample": 0.6649399890790841,
    "colsample_bytree": 0.6518297162203942,
    "reg_alpha": 0.00014075274223774678,
    "reg_lambda": 0.45206635415803376,
}

CHALLENGER_MODEL_VERSION = "xgb-if-settlement@3-if2-challenger"
COMPARISON_SCOPE = "SYNTHETIC_MECHANICS_ONLY_REPEATED_HOLDOUT_NOT_FOR_PROMOTION"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an IF@2 RazorTrust challenger without replacing the signed champion"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--merchants-per-family", type=int, default=40)
    parser.add_argument("--transactions-per-merchant", type=int, default=80)
    parser.add_argument("--estimators", type=int, default=300)
    parser.add_argument("--signer-key-id", default="development-if2-challenger-key")
    parser.add_argument(
        "--development-signing-key",
        action="store_true",
        help="Generate an ephemeral development signing key for this non-production challenger",
    )
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    champion_release = ROOT / "artifacts" / "tier0" / "model-release"
    champion_public_key_path = ROOT / "artifacts" / "tier0" / "model-release-public-key.txt"
    champion_report_path = ROOT / "artifacts" / "tier0" / "sealed_test_report.json"
    for path in (champion_release, champion_public_key_path, champion_report_path):
        if not path.exists():
            raise FileNotFoundError(f"required champion artifact missing: {path}")

    merchants, transactions, holds = generate_dataset(
        seed=args.seed,
        merchants_per_family=args.merchants_per_family,
        transactions_per_merchant=args.transactions_per_merchant,
    )
    dataset_manifest = write_dataset(
        output / "dataset",
        seed=args.seed,
        merchants_per_family=args.merchants_per_family,
        transactions_per_merchant=args.transactions_per_merchant,
    )
    frame = build_training_frame(merchants, transactions, holds)
    split_manifest = create_split_manifest(frame, dataset_manifest.content_sha256, seed=args.seed)
    write_split_manifest(output / "split_manifest.json", split_manifest)

    result = train_model_bundle(
        frame,
        split_manifest,
        seed=args.seed,
        n_estimators=args.estimators,
        xgb_params=CHAMPION_XGB_PARAMS,
    )
    if getattr(result.bundle.anomaly_scorer, "version", "") != "isolation-forest@2":
        raise RuntimeError("challenger training did not use isolation-forest@2")
    if getattr(result.bundle.anomaly_scorer, "reference_mode", "") != "merchant_group_oof":
        raise RuntimeError("challenger training did not use merchant_group_oof reference scores")

    # Give the challenger a distinct model identity. Nothing here changes the
    # model currently mounted by docker-compose.ml.yml.
    result.bundle.model_version = CHALLENGER_MODEL_VERSION
    result.report.model_version = CHALLENGER_MODEL_VERSION

    training_payload = {
        **result.report.model_dump(mode="json"),
        "dataset_source": "SYNTHETIC_MECHANICS_ONLY",
        "dataset_sha256": dataset_manifest.content_sha256,
        "challenger": True,
        "promotion_eligible": False,
        "comparison_scope": COMPARISON_SCOPE,
        "xgb_parameters_source": "ORIGINAL_CHAMPION_OPTUNA_BEST_TRIAL",
        "xgb_parameters": CHAMPION_XGB_PARAMS,
    }
    _write_json(output / "training_report.json", training_payload)

    evaluation = evaluate_sealed_test(frame, result.bundle, split_manifest)
    evaluation_payload = {
        **evaluation.model_dump(mode="json"),
        "dataset_source": "SYNTHETIC_MECHANICS_ONLY",
        "dataset_sha256": dataset_manifest.content_sha256,
        "challenger": True,
        "promotion_eligible": False,
        "comparison_scope": COMPARISON_SCOPE,
        "warning": (
            "This synthetic sealed set has already been observed during prior mechanics work. "
            "Use only for engineering comparison; do not use it to promote a production model."
        ),
    }
    _write_json(output / "sealed_test_report.json", evaluation_payload)

    if not args.development_signing_key:
        raise ValueError("pass --development-signing-key for this mechanics challenger")
    private_key, public_key = generate_release_keypair()
    release_dir = output / "model-release"
    save_model_release(
        release_dir,
        result,
        private_key_b64=private_key,
        signer_key_id=args.signer_key_id,
        split_manifest_sha256=split_manifest.content_sha256,
        training_dataset_sha256=dataset_manifest.content_sha256,
        evaluation_report_sha256=evaluation.content_sha256,
        release_id=f"if2-challenger-seed-{args.seed}",
    )
    (output / "model-release-public-key.txt").write_text(public_key + "\n", encoding="ascii")

    champion_public_key = champion_public_key_path.read_text(encoding="ascii").strip()
    champion_model = TrainedRiskModel.load_release(
        champion_release, public_key_b64=champion_public_key
    )
    challenger_model = TrainedRiskModel.load_release(release_dir, public_key_b64=public_key)

    fixture_diagnostics: dict[str, Any] = {}
    for fixture_name in ("normal_baseline", "novel_risk_burst"):
        vector = fixture_feature_vector(fixture_name)
        champion_prediction = champion_model.predict(vector)
        challenger_prediction = challenger_model.predict(vector)
        fixture_diagnostics[fixture_name] = {
            "champion": _prediction_summary(champion_prediction),
            "challenger": _prediction_summary(challenger_prediction),
        }
    _write_json(output / "fixture_diagnostics.json", fixture_diagnostics)

    champion_report = json.loads(champion_report_path.read_text(encoding="utf-8"))
    comparison = {
        "comparison_scope": COMPARISON_SCOPE,
        "promotion_eligible": False,
        "champion": {
            "model_version": champion_model.version,
            "anomaly_model_version": getattr(
                champion_model.bundle.anomaly_scorer, "version", "isolation-forest@1"
            ),
            "anomaly_reference_mode": getattr(
                champion_model.bundle.anomaly_scorer, "reference_mode", "legacy_in_sample"
            ),
            **_metric_subset(champion_report),
        },
        "challenger": {
            "model_version": challenger_model.version,
            "anomaly_model_version": getattr(
                challenger_model.bundle.anomaly_scorer, "version", "unknown"
            ),
            "anomaly_reference_mode": getattr(
                challenger_model.bundle.anomaly_scorer, "reference_mode", "unknown"
            ),
            **_metric_subset(evaluation_payload),
        },
        "fixture_diagnostics": fixture_diagnostics,
        "generated_at": datetime.now(UTC).isoformat(),
        "decision": "NO_PROMOTION_EVALUATION_ONLY",
        "next_gate": (
            "Evaluate AI-v3 candidates on a fresh locked merchant-family benchmark and then shadow data; "
            "do not promote from this repeatedly observed synthetic holdout."
        ),
    }
    comparison["content_sha256"] = _sha256_json(comparison)
    _write_json(output / "challenger_comparison.json", comparison)

    print(
        json.dumps(
            {
                "status": "OK",
                "output": str(output),
                "challenger_model_version": challenger_model.version,
                "challenger_anomaly_model_version": getattr(
                    challenger_model.bundle.anomaly_scorer, "version", "unknown"
                ),
                "challenger_anomaly_reference_mode": getattr(
                    challenger_model.bundle.anomaly_scorer, "reference_mode", "unknown"
                ),
                "promotion_eligible": False,
                "comparison_scope": COMPARISON_SCOPE,
                "comparison_sha256": comparison["content_sha256"],
            },
            indent=2,
        )
    )


def _prediction_summary(prediction: Any) -> dict[str, Any]:
    probabilities = prediction.probabilities.model_dump(mode="json")
    return {
        "model_version": prediction.model_version,
        "calibration_method": prediction.calibration_method,
        "probabilities": probabilities,
        "anomaly_score": prediction.anomaly_score,
        "anomaly_raw_score": prediction.anomaly_raw_score,
        "anomaly_reference_max": prediction.anomaly_reference_max,
        "anomaly_tail_excess": prediction.anomaly_tail_excess,
        "anomaly_reference_size": prediction.anomaly_reference_size,
        "anomaly_model_version": prediction.anomaly_model_version,
        "anomaly_reference_mode": prediction.anomaly_reference_mode,
        "novelty_override": prediction.novelty_override,
    }


def _metric_subset(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "log_loss": report.get("log_loss"),
        "multiclass_brier": report.get("multiclass_brier"),
        "top_label_ece": report.get("top_label_ece"),
        "expected_cost_units": report.get("expected_cost_units"),
        "false_release_rate": report.get("false_release_rate"),
        "classes": report.get("classes"),
        "cost_matrix_version": report.get("cost_matrix_version"),
        "cost_matrix_sha256": report.get("cost_matrix_sha256"),
    }


def _sha256_json(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
