from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from razortrust.ml.cost_calibration import OutcomeCostRecord, calibrate_cost_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate a candidate cost matrix from outcomes")
    parser.add_argument("outcomes", type=Path, help="CSV with decisions and observed loss columns")
    parser.add_argument("--version", required=True)
    parser.add_argument("--minimum-cell-count", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.outcomes)
    records = [OutcomeCostRecord.model_validate(row) for row in frame.to_dict(orient="records")]
    report = calibrate_cost_matrix(
        records,
        version=args.version,
        minimum_cell_count=args.minimum_cell_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "content_sha256": report.content_sha256}))


if __name__ == "__main__":
    main()
