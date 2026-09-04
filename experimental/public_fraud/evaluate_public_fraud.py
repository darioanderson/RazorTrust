from __future__ import annotations

import argparse
import json
from pathlib import Path

from experimental.public_fraud.public_fraud import evaluate_public_fraud
from scripts.train_model import _source_tree_sha256

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RazorTrust on the public ULB dataset")
    parser.add_argument("dataset", type=Path, help="creditcard.csv or its Kaggle ZIP archive")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "public-fraud")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--estimators", type=int, default=300)
    args = parser.parse_args()
    report = evaluate_public_fraud(
        args.dataset,
        source_tree_sha256=_source_tree_sha256(ROOT / "src" / "razortrust"),
        output_dir=args.output,
        seed=args.seed,
        estimators=args.estimators,
    )
    print(json.dumps(report.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
