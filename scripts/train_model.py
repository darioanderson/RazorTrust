from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import mlflow

from razortrust.ml.dataset import build_training_frame, model_matrix
from razortrust.ml.evaluation import evaluate_sealed_test
from razortrust.ml.modeling import train_model_bundle
from razortrust.ml.real_data import load_real_settlement_dataset
from razortrust.ml.release import save_model_release
from razortrust.ml.splits import create_split_manifest, write_split_manifest
from razortrust.ml.tuning import tune_xgboost
from razortrust.ml.uncertainty import ApsConformalAbstainer
from razortrust.security import generate_release_keypair
from razortrust.synthetic import generate_dataset, write_dataset

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the RazorTrust Tier 0 model")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "tier0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merchants-per-family", type=int, default=40)
    parser.add_argument("--transactions-per-merchant", type=int, default=80)
    parser.add_argument("--estimators", type=int, default=300)
    parser.add_argument("--hpo-trials", type=int, default=0)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Explicitly permit the synthetic mechanics-check generator",
    )
    source.add_argument(
        "--dataset-dir",
        type=Path,
        help=(
            "Hash-verified PUBLIC_REAL or PARTNER_REAL settlement dataset directory; "
            "raw records are read in place and are not copied into model artifacts"
        ),
    )
    parser.add_argument("--signer-key-id", default="development-release-key")
    parser.add_argument(
        "--development-signing-key",
        action="store_true",
        help=(
            "Generate an ephemeral local signing key; never use this option for "
            "production releases."
        ),
    )
    default_tracking_uri = f"sqlite:///{(ROOT / 'var' / 'mlflow.db').resolve().as_posix()}"
    parser.add_argument("--tracking-uri", default=default_tracking_uri)
    args = parser.parse_args()
    source_tree_sha256 = _source_tree_sha256(ROOT / "src" / "razortrust")

    if args.dataset_dir is not None:
        merchants, transactions, holds, dataset_manifest = load_real_settlement_dataset(
            args.dataset_dir
        )
        dataset_source = dataset_manifest.data_origin
        dataset_id = dataset_manifest.dataset_id
    else:
        data_dir = args.output / "dataset"
        merchants, transactions, holds = generate_dataset(
            seed=args.seed,
            merchants_per_family=args.merchants_per_family,
            transactions_per_merchant=args.transactions_per_merchant,
        )
        dataset_manifest = write_dataset(
            data_dir,
            seed=args.seed,
            merchants_per_family=args.merchants_per_family,
            transactions_per_merchant=args.transactions_per_merchant,
        )
        dataset_source = "SYNTHETIC_MECHANICS_ONLY"
        dataset_id = dataset_manifest.dataset_id
    frame = build_training_frame(merchants, transactions, holds)
    split_manifest = create_split_manifest(frame, dataset_manifest.content_sha256, seed=args.seed)
    split_path = args.output / "split_manifest.json"
    write_split_manifest(split_path, split_manifest)

    (ROOT / "var").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("razortrust-tier0")
    with mlflow.start_run(run_name=f"tier0-seed-{args.seed}"):
        mlflow.log_params(
            {
                "seed": args.seed,
                "estimators": args.estimators,
                "source_tree_sha256": source_tree_sha256,
                "dataset_source": dataset_source,
                "dataset_id": dataset_id,
                "dataset_sha256": dataset_manifest.content_sha256,
                "split_sha256": split_manifest.content_sha256,
            }
        )
        if args.dataset_dir is None:
            mlflow.log_params(
                {
                    "merchants_per_family": args.merchants_per_family,
                    "transactions_per_merchant": args.transactions_per_merchant,
                }
            )
        tuned_parameters = (
            tune_xgboost(frame, split_manifest, n_trials=args.hpo_trials, seed=args.seed)
            if args.hpo_trials > 0
            else {}
        )
        mlflow.log_params({f"xgb_{key}": value for key, value in tuned_parameters.items()})
        result = train_model_bundle(
            frame,
            split_manifest,
            seed=args.seed,
            n_estimators=args.estimators,
            xgb_params=tuned_parameters,
        )
        run_provenance = {
            "dataset_source": dataset_source,
            "dataset_id": dataset_id,
            "dataset_sha256": dataset_manifest.content_sha256,
            "raw_dataset_copied_to_artifacts": args.dataset_dir is None,
        }
        training_path = args.output / "training_report.json"
        _write_report(
            training_path,
            {**result.report.model_dump(mode="json"), **run_provenance},
            source_tree_sha256,
        )
        evaluation = evaluate_sealed_test(frame, result.bundle, split_manifest)
        evaluation_path = args.output / "sealed_test_report.json"
        _write_report(
            evaluation_path,
            {**evaluation.model_dump(mode="json"), **run_provenance},
            source_tree_sha256,
        )
        test_frame = frame[frame["merchant_id"].isin(split_manifest.test_merchants)].reset_index(
            drop=True
        )
        test_probabilities = result.bundle.predict_proba(
            model_matrix(test_frame), test_frame["cohort"]
        )
        conformal = ApsConformalAbstainer(confidence_level=0.90, seed=args.seed).conformalize(
            result.policy_probabilities, result.policy_labels
        )
        conformal_summary = conformal.evaluate(
            test_probabilities,
            test_frame["operational_target"]
            .map({"RELEASE": 0, "EVIDENCE_NEEDED": 1, "ESCALATE": 2})
            .to_numpy(dtype=int),
        )
        conformal_path = args.output / "conformal_test_report.json"
        _write_report(
            conformal_path,
            {**conformal_summary.model_dump(mode="json"), **run_provenance},
            source_tree_sha256,
        )
        private_key = os.getenv("RAZORTRUST_RELEASE_PRIVATE_KEY")
        public_key = os.getenv("RAZORTRUST_RELEASE_PUBLIC_KEY")
        if not private_key:
            if not args.development_signing_key:
                raise ValueError(
                    "set RAZORTRUST_RELEASE_PRIVATE_KEY or pass --development-signing-key"
                )
            private_key, public_key = generate_release_keypair()
        if not public_key:
            raise ValueError(
                "RAZORTRUST_RELEASE_PUBLIC_KEY is required with an external private key"
            )
        release_path = args.output / "model-release"
        release_manifest = save_model_release(
            release_path,
            result,
            private_key_b64=private_key,
            signer_key_id=args.signer_key_id,
            split_manifest_sha256=split_manifest.content_sha256,
            training_dataset_sha256=dataset_manifest.content_sha256,
            evaluation_report_sha256=evaluation.content_sha256,
            release_id=f"tier0-seed-{args.seed}",
        )
        public_key_path = args.output / "model-release-public-key.txt"
        _write_immutable(public_key_path, public_key + "\n")
        mlflow.log_metrics(
            {
                "test_log_loss": evaluation.log_loss,
                "test_multiclass_brier": evaluation.multiclass_brier,
                "test_expected_cost_units": evaluation.expected_cost_units,
                "test_false_release_rate": evaluation.false_release_rate,
            }
        )
        for artifact in (
            split_path,
            training_path,
            evaluation_path,
            conformal_path,
            public_key_path,
        ):
            mlflow.log_artifact(str(artifact))
        mlflow.log_artifacts(str(release_path), artifact_path="model-release")

    summary = {
        "model_release": str(release_path),
        "classifier_sha256": release_manifest.files["xgboost.ubj"],
        "public_key": str(public_key_path),
        "training_report": str(training_path),
        "sealed_test_report": str(evaluation_path),
        "conformal_test_report": str(conformal_path),
        "evaluation_sha256": evaluation.content_sha256,
        "source_tree_sha256": source_tree_sha256,
        "dataset_source": dataset_source,
        "dataset_id": dataset_id,
        "dataset_sha256": dataset_manifest.content_sha256,
    }
    print(json.dumps(summary, indent=2))


def _write_immutable(path: Path, content: str) -> None:
    encoded = content.encode("utf-8")
    if path.exists() and path.read_bytes() != encoded:
        raise FileExistsError(f"refusing to replace immutable artifact: {path}")
    if not path.exists():
        path.write_bytes(encoded)


def _write_report(path: Path, payload: dict[str, object], source_tree_sha256: str) -> None:
    stamped = {
        **payload,
        "source_provenance": source_tree_sha256,
        "source_tree_sha256": source_tree_sha256,
    }
    _write_immutable(path, json.dumps(stamped, indent=2) + "\n")


def _source_tree_sha256(source_root: Path) -> str:
    """Hash Python source paths and bytes in a stable order."""
    source_files = sorted(source_root.rglob("*.py"), key=lambda path: path.as_posix())
    if not source_files:
        raise ValueError(f"no Python source files found under {source_root}")
    digest = hashlib.sha256()
    for path in source_files:
        relative_path = path.relative_to(source_root).as_posix().encode("utf-8")
        digest.update(relative_path)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


if __name__ == "__main__":
    main()
