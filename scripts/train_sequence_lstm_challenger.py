from __future__ import annotations

import argparse
import json
from pathlib import Path

from razortrust.ml.sequence_lstm import (
    build_point_in_time_sequence_tensor,
    fit_lstm_sequences,
    predict_lstm_probability,
)
from razortrust.synthetic import TrueRiskState, generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the Phase-4 sequence LSTM research challenger"
    )
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--merchants-per-family", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/research/phase4-sequence-lstm")
    )
    args = parser.parse_args()

    merchants, transactions, holds = generate_dataset(
        seed=args.seed,
        merchants_per_family=args.merchants_per_family,
        transactions_per_merchant=32,
    )
    del merchants
    examples = [
        build_point_in_time_sequence_tensor(
            hold.hold.merchant_id, transactions, hold.hold.triggered_at
        )
        for hold in holds
    ]
    labels = [int(hold.true_risk_state == TrueRiskState.RISKY) for hold in holds]
    model = fit_lstm_sequences(examples, labels, epochs=args.epochs, seed=args.seed)
    probabilities = [predict_lstm_probability(model, example) for example in examples]
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": 4,
        "model_version": "sequence-lstm-risk@1",
        "evaluation_mode": "RESEARCH_ONLY_POINT_IN_TIME_SEQUENCE",
        "case_count": len(labels),
        "mean_probability": sum(probabilities) / len(probabilities),
        "production_action_eligible": False,
        "serving_change_authorized": False,
        "champion_remains": "xgb-if-settlement@2",
    }
    (args.output / "sequence_lstm_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
