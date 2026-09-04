from __future__ import annotations

import argparse
import json
from pathlib import Path

from razortrust.ml.longitudinal_data import (
    LongitudinalConfig,
    future_mutation_invariance_check,
    write_longitudinal_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RazorTrust synthetic longitudinal dataset v3.1"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--merchants-per-family", type=int, default=80)
    parser.add_argument("--history-days", type=int, default=60)
    parser.add_argument("--baseline-days", type=int, default=30)
    parser.add_argument("--confounder-probability", type=float, default=0.18)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LongitudinalConfig(
        seed=args.seed,
        merchants_per_family=args.merchants_per_family,
        history_days=args.history_days,
        baseline_days=args.baseline_days,
        confounder_probability=args.confounder_probability,
    )
    if not future_mutation_invariance_check(config):
        raise RuntimeError("future-data mutation changed point-in-time features")
    report = write_longitudinal_dataset(Path(args.output), config)
    report = {**report, "future_mutation_invariance": "PASS"}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
