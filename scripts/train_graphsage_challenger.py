from __future__ import annotations

import argparse
import json
from pathlib import Path

from razortrust.ml.graphsage import (
    build_point_in_time_graph_snapshot,
    fit_graphsage_snapshots,
    predict_graphsage_probability,
)
from razortrust.synthetic import TrueRiskState, generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Phase-3 GraphSAGE research challenger")
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--merchants-per-family", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--output", type=Path, default=Path("artifacts/research/phase3-graphsage"))
    args = parser.parse_args()

    merchants, transactions, holds = generate_dataset(
        seed=args.seed,
        merchants_per_family=args.merchants_per_family,
        transactions_per_merchant=24,
    )
    del merchants
    snapshots = [
        build_point_in_time_graph_snapshot(
            hold.hold.merchant_id, transactions, hold.hold.triggered_at
        )
        for hold in holds
    ]
    labels = [int(hold.true_risk_state == TrueRiskState.RISKY) for hold in holds]
    model = fit_graphsage_snapshots(snapshots, labels, epochs=args.epochs, seed=args.seed)
    probabilities = [predict_graphsage_probability(model, snapshot) for snapshot in snapshots]
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": 3,
        "model_version": "graphsage-risk@1",
        "evaluation_mode": "RESEARCH_ONLY_POINT_IN_TIME_SNAPSHOTS",
        "case_count": len(labels),
        "mean_probability": sum(probabilities) / len(probabilities),
        "production_action_eligible": False,
        "serving_change_authorized": False,
        "champion_remains": "xgb-if-settlement@2",
    }
    (args.output / "graphsage_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
