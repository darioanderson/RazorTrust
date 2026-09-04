from __future__ import annotations

import argparse
from pathlib import Path

from razortrust.synthetic import write_dataset

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the seeded RazorTrust synthetic dataset."
    )
    parser.add_argument("--output", default=str(ROOT / "data" / "generated"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merchants-per-family", type=int, default=20)
    parser.add_argument("--transactions-per-merchant", type=int, default=80)
    args = parser.parse_args()
    manifest = write_dataset(
        args.output,
        seed=args.seed,
        merchants_per_family=args.merchants_per_family,
        transactions_per_merchant=args.transactions_per_merchant,
    )
    print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
