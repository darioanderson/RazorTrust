from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from razortrust.ml.sequence_models import (
    FALSE_HOLD_COST,
    FALSE_RELEASE_COST,
    build_sequence_examples,
    evaluate_sequence_challenger,
    evaluate_temporal_feature_gate,
    sequence_gate_passed,
)
from razortrust.synthetic import generate_dataset

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run gated Phase 4 temporal/LSTM/Transformer research"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "research" / "phase4-sequence",
    )
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--merchants-per-family", type=int, default=20)
    parser.add_argument("--transactions-per-merchant", type=int, default=80)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    merchants, transactions, holds = generate_dataset(
        seed=args.seed,
        merchants_per_family=args.merchants_per_family,
        transactions_per_merchant=args.transactions_per_merchant,
    )
    baseline = evaluate_temporal_feature_gate(
        merchants,
        transactions,
        holds,
        seed=args.seed,
        folds=args.folds,
    )

    challengers: dict[str, dict[str, object]] = {}
    retained: list[str] = []
    if baseline.gate_passed:
        examples = build_sequence_examples(transactions, holds)
        for model_type in ("lstm", "transformer"):
            challenger = evaluate_sequence_challenger(
                examples,
                model_type=model_type,
                seed=args.seed,
                folds=args.folds,
                epochs=args.epochs,
            )
            passed = sequence_gate_passed(
                temporal_pr_auc=baseline.temporal_pr_auc,
                temporal_expected_cost=baseline.temporal_expected_cost,
                temporal_risk_recall=baseline.temporal_risk_recall,
                challenger=challenger,
            )
            challengers[model_type] = {**asdict(challenger), "gate_passed": passed}
            if passed:
                retained.append(model_type)
        decision = "RETAIN_FOR_RESEARCH" if retained else "REJECT_NEURAL_SEQUENCE_MODELS"
    else:
        decision = "BLOCKED_BY_TEMPORAL_ABLATION"

    report = {
        "schema_version": "1.0",
        "phase": 4,
        "decision": decision,
        "seed": args.seed,
        "merchants_per_family": args.merchants_per_family,
        "transactions_per_merchant": args.transactions_per_merchant,
        "folds": args.folds,
        "epochs": args.epochs,
        "case_count": len(holds),
        "transaction_count": len(transactions),
        "evaluation_mode": "GROUPED_DEVELOPMENT_POINT_IN_TIME_NO_SEALED_TEST",
        "fixed_threshold": 0.5,
        "costs": {
            "false_release": FALSE_RELEASE_COST,
            "false_hold": FALSE_HOLD_COST,
        },
        "temporal_ablation": asdict(baseline),
        "challengers": challengers,
        "retained": retained,
        "safety": {
            "timestamp_rule": "transaction.timestamp < hold.triggered_at",
            "sealed_test_used": False,
            "stress_set_used": False,
            "production_action_eligible": False,
            "serving_change_authorized": False,
            "automatic_promotion": False,
            "champion_remains": "xgb-if-settlement@2",
        },
    }
    output = args.output / "phase4_sequence_gate_report.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
