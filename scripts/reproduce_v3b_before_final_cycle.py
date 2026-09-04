from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import skops.io as sio
from xgboost import XGBClassifier

from razortrust.ml.v3_research import (
    add_v3b_candidates,
    calibrate_probabilities,
    create_research_partitions,
    evaluate_actions,
    load_v31_frame,
    sha256_file,
    target_indices,
    threshold_actions,
    write_json,
)
from razortrust.security import verify_manifest

ROOT = Path(__file__).resolve().parents[1]
V3B = ROOT / "artifacts" / "research" / "v3b-final-20260903"
DATASET = ROOT / "artifacts" / "research" / "data-v31-large-seed-20260903" / "hold_windows.csv.gz"
OUTPUT = ROOT / "artifacts" / "research" / "v3b-reproduction-before-final-cycle-20260903"


def _load_skops(path: Path) -> Any:
    untrusted = sio.get_untrusted_types(file=path)
    allowed = ("razortrust.", "sklearn.", "xgboost.", "numpy.", "scipy.")
    rejected = [name for name in untrusted if not name.startswith(allowed)]
    if rejected:
        raise ValueError(f"unapproved serialized types: {rejected}")
    return sio.load(path, trusted=untrusted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite reproduction evidence: {args.output}")

    candidate = V3B / "research-candidate"
    manifest = json.loads((candidate / "candidate-manifest.jcs.json").read_text(encoding="utf-8"))
    signature = (candidate / "candidate-manifest.sig").read_text(encoding="ascii").strip()
    public_key = (candidate / "candidate-public-key.txt").read_text(encoding="ascii").strip()
    if not verify_manifest(manifest, signature, public_key):
        raise ValueError("v3B candidate signature is invalid")
    mismatches = {
        name: {"expected": expected, "actual": sha256_file(candidate / name)}
        for name, expected in manifest["files"].items()
        if not (candidate / name).is_file() or sha256_file(candidate / name) != expected
    }
    if mismatches:
        raise ValueError(f"v3B candidate artifact hash mismatch: {mismatches}")
    if sha256_file(DATASET) != manifest["dataset_sha256"]:
        raise ValueError("accepted dataset no longer matches the signed v3B candidate")

    gate_report = json.loads((V3B / "v3b_gate_report.json").read_text(encoding="utf-8"))
    features = tuple(json.loads((candidate / "feature-set.json").read_text())["features"])
    thresholds = gate_report["thresholds_selected_on_development_oof"]
    frame = add_v3b_candidates(load_v31_frame(DATASET))
    partitions, partition_manifest = create_research_partitions(frame)
    if partition_manifest["content_sha256"] != manifest["partition_manifest_sha256"]:
        raise ValueError("partition reconstruction does not match signed v3B candidate")
    gate = frame.loc[partitions == "v3b_gate"].reset_index(drop=True)

    model = XGBClassifier()
    model.load_model(candidate / "xgboost.ubj")
    calibrator = _load_skops(candidate / "calibrator.skops")
    raw = np.asarray(model.predict_proba(gate.loc[:, features]), dtype=float)
    probabilities = calibrate_probabilities(calibrator, raw)
    labels = target_indices(gate)
    actions = threshold_actions(
        probabilities, thresholds["release_threshold"], thresholds["escalate_threshold"]
    )
    reproduced = evaluate_actions(labels, actions, probabilities)
    expected = gate_report["gate_metrics"]
    exact_fields = (
        "row_count",
        "false_release_count",
        "false_release_rate",
        "true_release_recall",
        "confusion",
    )
    matches = all(reproduced[field] == expected[field] for field in exact_fields)
    if not matches:
        raise RuntimeError("v3B gate result did not reproduce exactly")

    report = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": "mandatory pre-training reproduction for the authorized final cycle",
        "training_performed": False,
        "artifact_signature_valid": True,
        "artifact_hashes_valid": True,
        "dataset_sha256": sha256_file(DATASET),
        "dataset_manifest_version": "synthetic-v3.1-longitudinal",
        "dataset_v3_2_disposition": (
            "No Dataset v3.2 corpus exists in the supplied repository; the unchanged, signed "
            "accepted corpus that produced v3B is used. No replacement dataset was created."
        ),
        "partition_manifest_sha256": partition_manifest["content_sha256"],
        "evaluated_partition": "v3b_gate",
        "sealed_accessed": False,
        "stress_accessed": False,
        "unknown_family_accessed": False,
        "expected_metrics": expected,
        "reproduced_metrics": reproduced,
        "exact_reproduction": True,
        "comparison_fields": list(exact_fields),
    }
    args.output.mkdir(parents=True)
    write_json(args.output / "v3b_reproduction_report.json", report)
    print(json.dumps({"output": str(args.output), "exact_reproduction": True}))


if __name__ == "__main__":
    main()
