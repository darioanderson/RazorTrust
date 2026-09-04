from __future__ import annotations

import argparse
import json
from pathlib import Path

from razortrust.features import FEATURE_COLUMNS
from razortrust.ml.dataset import build_training_frame, model_matrix
from razortrust.ml.graph_evaluation import (
    build_graph_training_frame,
    evaluate_graph_statistics_gate,
)
from razortrust.ml.loafo import evaluate_leave_one_attack_family_out
from razortrust.ml.monitoring import batch_drift_report, write_evidently_drift_report
from razortrust.synthetic import generate_dataset

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run gated RazorTrust research evaluations")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "research")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merchants-per-family", type=int, default=8)
    parser.add_argument("--with-autoencoder", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    dataset = generate_dataset(
        seed=args.seed,
        merchants_per_family=args.merchants_per_family,
        transactions_per_merchant=24,
    )
    frame = build_training_frame(*dataset)
    loafo = evaluate_leave_one_attack_family_out(
        frame,
        novelty_threshold=0.95,
        use_autoencoder=args.with_autoencoder,
        seed=args.seed,
    )
    graph_frame = build_graph_training_frame(*dataset)
    graph_gate = evaluate_graph_statistics_gate(graph_frame, seed=args.seed)

    reference = model_matrix(frame[frame["true_risk_state"].astype(str) == "LEGITIMATE"])
    shifted = reference.copy()
    shifted.loc[:, list(FEATURE_COLUMNS)] += 0.75
    drift = batch_drift_report(reference, shifted)
    write_evidently_drift_report(
        reference,
        shifted,
        html_path=str(args.output / "evidently-drift.html"),
        json_path=str(args.output / "evidently-drift.json"),
    )
    report = {
        "schema_version": "1.0",
        "seed": args.seed,
        "loafo": loafo.model_dump(mode="json"),
        "graph_gate": graph_gate.model_dump(mode="json"),
        "graphsage_entry_allowed": graph_gate.gate_passed,
        "drift": drift.model_dump(mode="json"),
        "gated_components": {
            "graphsage": "ALLOWED" if graph_gate.gate_passed else "BLOCKED_BY_ABLATION",
            "temporal_gnn": "BLOCKED_UNTIL_GRAPHSAGE_VALUE_IS_PROVEN",
            "sequence_neural_model": "BLOCKED_UNTIL_ENGINEERED_TEMPORAL_BASELINE_IS_BEATEN",
            "sequence_transformer": "BLOCKED_UNTIL_GRU_OR_LSTM_LEAVES_A_MEASURED_GAP",
        },
    }
    path = args.output / "research-evaluation.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
