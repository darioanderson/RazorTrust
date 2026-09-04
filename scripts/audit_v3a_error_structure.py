from __future__ import annotations

import argparse
from pathlib import Path

from razortrust.ml.v3_research import (
    DEFAULT_SEED,
    V3A_FEATURES,
    audit_v3a_errors,
    calibrate_probabilities,
    create_research_partitions,
    cross_validated_probabilities,
    fit_probability_calibrator,
    load_v31_frame,
    select_thresholds,
    sha256_file,
    target_indices,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    ROOT / "artifacts" / "research" / "data-v31-large-seed-20260903" / "hold_windows.csv.gz"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the sealed-safe v3A error-structure audit")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    frame = load_v31_frame(args.dataset)
    partition, manifest = create_research_partitions(frame, seed=args.seed)
    development = frame.loc[partition == "development"].reset_index(drop=True)
    raw_probabilities = cross_validated_probabilities(
        development, V3A_FEATURES, seed=args.seed, device=args.device
    )
    labels = target_indices(development)
    calibrator = fit_probability_calibrator(raw_probabilities, labels, seed=args.seed)
    probabilities = calibrate_probabilities(calibrator, raw_probabilities)
    thresholds = select_thresholds(probabilities, labels)
    report = audit_v3a_errors(development, probabilities, thresholds)
    report["dataset_sha256"] = sha256_file(args.dataset)
    report["partition_manifest_sha256"] = manifest["content_sha256"]
    report["device"] = args.device

    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "partition_manifest.json", manifest)
    write_json(args.output / "v3a_error_audit.json", report)
    print(args.output / "v3a_error_audit.json")


if __name__ == "__main__":
    main()
