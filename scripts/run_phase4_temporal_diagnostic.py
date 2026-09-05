from __future__ import annotations

import argparse
import json
from pathlib import Path

from razortrust.ml.temporal_diagnostics import run_seed_diagnostic, summarize_seed_results
from razortrust.synthetic import generate_dataset

ROOT = Path(__file__).resolve().parents[1]


def _parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase 4 temporal feature construction before closing the phase"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "research" / "phase4-temporal-diagnostic",
    )
    parser.add_argument(
        "--seeds", type=_parse_int_list, default=[20260904, 20260905, 20260906, 20260907, 20260908]
    )
    parser.add_argument("--windows", type=_parse_int_list, default=[1, 6, 24, 72])
    parser.add_argument("--merchants-per-family", type=int, default=4)
    parser.add_argument("--transactions-per-merchant", type=int, default=32)
    parser.add_argument("--folds", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    seed_results = []
    for seed in args.seeds:
        merchants, transactions, holds = generate_dataset(
            seed=seed,
            merchants_per_family=args.merchants_per_family,
            transactions_per_merchant=args.transactions_per_merchant,
        )
        result = run_seed_diagnostic(
            merchants,
            transactions,
            holds,
            seed=seed,
            folds=args.folds,
            windows=args.windows,
        )
        seed_results.append(result)
        (args.output / f"seed-{seed}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    summary = summarize_seed_results(seed_results, args.windows)
    engineering_failures = []
    for seed_result in seed_results:
        for window, data in seed_result["windows"].items():
            audit = data["audit"]
            if audit["point_in_time_violations"] or audit["window_violations"]:
                engineering_failures.append(
                    f"seed={seed_result['seed']} window={window}: "
                    "timestamp/window invariant failure"
                )
            for feature, stats in audit["features"].items():
                if stats["nan_count"] or stats["inf_count"]:
                    engineering_failures.append(
                        f"seed={seed_result['seed']} window={window} "
                        f"feature={feature}: non-finite values"
                    )

    harmful_features = []
    for seed_result in seed_results:
        data = seed_result["windows"].get("24")
        if not data:
            continue
        metrics = data["metrics"]
        full_pr = metrics["temporal_full"]["pr_auc"]
        full_cost = metrics["temporal_full"]["expected_cost"]
        for feature in (
            "burst_score_10m",
            "amount_autocorrelation",
            "auth_failure_run_max",
            "new_device_transition_rate",
        ):
            drop = metrics[f"drop_one::{feature}"]
            if drop["pr_auc"] > full_pr + 0.01 and drop["expected_cost"] <= full_cost:
                harmful_features.append(
                    {
                        "seed": seed_result["seed"],
                        "feature": feature,
                        "evidence": "drop-one improves PR-AUC by >0.01 without worsening cost",
                    }
                )

    w24 = summary["windows"].get("24", {})
    if engineering_failures:
        decision = "ENGINEERING_ISSUE_FOUND"
    elif harmful_features:
        decision = "SPECIFIC_TEMPORAL_FEATURE_SUSPECTED"
    elif w24 and w24["pr_auc_improved_seed_count"] not in (0, len(seed_results)):
        decision = "SEED_SENSITIVE_TEMPORAL_RESULT"
    elif any(
        window_data["pr_auc_improved_seed_count"] >= max(2, len(seed_results) // 2 + 1)
        and window_data["expected_cost_delta_mean"] < 0
        for window_data in summary["windows"].values()
    ):
        decision = "WINDOW_CHOICE_SUSPECTED"
    else:
        decision = "NEGATIVE_TEMPORAL_RESULT_REPLICATED"

    report = {
        "schema_version": "1.0",
        "phase": 4,
        "diagnostic": "temporal_feature_construction_and_stability",
        "decision": decision,
        "seeds": args.seeds,
        "windows_hours": args.windows,
        "merchants_per_family": args.merchants_per_family,
        "transactions_per_merchant": args.transactions_per_merchant,
        "folds": args.folds,
        "costs": {"false_release": 100, "false_hold": 25},
        "summary": summary,
        "engineering_failures": engineering_failures,
        "harmful_feature_evidence": harmful_features,
        "notes": {
            "scaling": (
                "XGBoost gbtree/hist is split-based. Feature distributions are audited, "
                "but standardization is not treated as a primary remedy."
            ),
            "missingness": (
                "Current temporal construction returns zeros when no events exist. The "
                "diagnostic separately tests history flags/counts to detect whether "
                "zero/no-history conflation matters."
            ),
            "fold_safety": (
                "Temporal features are deterministic pre-cutoff summaries and have no "
                "train-fitted scaler/encoder/target transform outside the grouped folds."
            ),
            "sealed_test_used": False,
            "stress_set_used": False,
            "production_action_eligible": False,
            "serving_change_authorized": False,
            "champion_remains": "xgb-if-settlement@2",
        },
    }
    output = args.output / "phase4_temporal_diagnostic_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
