from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from razortrust.costs import DEFAULT_COST_MATRIX
from razortrust.domain import HoldDecision
from razortrust.ml.dataset import build_training_frame, model_matrix
from razortrust.ml.modeling import IsolationForestScorer
from razortrust.ml.novelty_guard import CalibratedRawNoveltyGuard, apply_guard_route
from razortrust.ml.runtime import TrainedRiskModel
from razortrust.ml.splits import create_split_manifest, write_split_manifest
from razortrust.synthetic import (
    TrueRiskState,
    generate_dataset,
    write_dataset,
)

LABEL_INDEX = {
    HoldDecision.RELEASE: 0,
    HoldDecision.EVIDENCE_NEEDED: 1,
    HoldDecision.ESCALATE: 2,
}
ALPHA_GRID = (0.005, 0.01, 0.02, 0.05)
ROUTES: tuple[Literal["EVIDENCE_NEEDED", "ESCALATE"], ...] = (
    "EVIDENCE_NEEDED",
    "ESCALATE",
)
SCOPE = "SYNTHETIC_RESEARCH_ONLY_FRESH_SEED_NO_PROMOTION"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="R4D.3 raw Isolation-Forest novelty-guard calibration"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--merchants-per-family", type=int, default=60)
    parser.add_argument("--transactions-per-merchant", type=int, default=80)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    champion_release = ROOT / "artifacts" / "tier0" / "model-release"
    champion_public_key_path = ROOT / "artifacts" / "tier0" / "model-release-public-key.txt"
    for path in (champion_release, champion_public_key_path):
        if not path.exists():
            raise FileNotFoundError(f"required champion artifact missing: {path}")

    manifest = write_dataset(
        output / "dataset",
        seed=args.seed,
        merchants_per_family=args.merchants_per_family,
        transactions_per_merchant=args.transactions_per_merchant,
    )
    merchants, transactions, holds = generate_dataset(
        seed=args.seed,
        merchants_per_family=args.merchants_per_family,
        transactions_per_merchant=args.transactions_per_merchant,
    )
    frame = build_training_frame(merchants, transactions, holds)
    split = create_split_manifest(frame, manifest.content_sha256, seed=args.seed)
    write_split_manifest(output / "split_manifest.json", split)

    train = _partition(frame, split.train_merchants)
    calibration = _partition(frame, split.calibration_merchants)
    test = _partition(frame, split.test_merchants)

    if2 = IsolationForestScorer(seed=args.seed).fit(
        model_matrix(train),
        train["cohort"],
        train["true_risk_state"].astype(str) == TrueRiskState.LEGITIMATE,
        train["merchant_id"].astype(str),
    )
    if getattr(if2, "version", "") != "isolation-forest@2":
        raise RuntimeError("R4D.3 requires isolation-forest@2 training semantics")
    if getattr(if2, "reference_mode", "") != "merchant_group_oof":
        raise RuntimeError("R4D.3 requires merchant_group_oof reference semantics")

    cal_details = if2.score_details(model_matrix(calibration), calibration["cohort"])
    test_details = if2.score_details(model_matrix(test), test["cohort"])

    champion = TrainedRiskModel.load_release(
        champion_release,
        public_key_b64=champion_public_key_path.read_text(encoding="ascii").strip(),
    )
    cal_probabilities = champion.bundle.predict_proba(
        model_matrix(calibration), calibration["cohort"]
    )
    test_probabilities = champion.bundle.predict_proba(model_matrix(test), test["cohort"])
    cal_base_actions = _threshold_actions(
        cal_probabilities, champion.thresholds.release, champion.thresholds.escalate
    )
    test_base_actions = _threshold_actions(
        test_probabilities, champion.thresholds.release, champion.thresholds.escalate
    )

    cal_true = _true_actions(calibration)
    test_true = _true_actions(test)
    cal_legit = calibration["true_risk_state"].astype(str).to_numpy() == TrueRiskState.LEGITIMATE
    test_legit = test["true_risk_state"].astype(str).to_numpy() == TrueRiskState.LEGITIMATE
    test_risky = ~test_legit

    candidates: list[dict[str, Any]] = []
    guards: dict[float, CalibratedRawNoveltyGuard] = {}
    for alpha in ALPHA_GRID:
        guard = CalibratedRawNoveltyGuard.fit(
            cal_details.raw_scores[cal_legit], target_false_alarm_rate=alpha
        )
        guards[alpha] = guard
        cal_guard_scores = guard.score(cal_details.raw_scores)
        for route in ROUTES:
            guarded = apply_guard_route(cal_base_actions, cal_guard_scores.signal, route=route)
            candidates.append(
                _candidate_metrics(
                    alpha=alpha,
                    route=route,
                    actions=guarded,
                    base_actions=cal_base_actions,
                    true_actions=cal_true,
                    legitimate_mask=cal_legit,
                    risky_mask=~cal_legit,
                    novelty_signal=cal_guard_scores.signal,
                    calibration=guard.calibration.as_dict(),
                )
            )

    selected = min(
        candidates,
        key=lambda row: (
            float(row["expected_cost_units"]),
            float(row["legitimate_release_block_rate"]),
            0 if row["route"] == "EVIDENCE_NEEDED" else 1,
            float(row["target_false_alarm_rate"]),
        ),
    )
    selected_alpha = float(selected["target_false_alarm_rate"])
    selected_route = str(selected["route"])
    selected_guard = guards[selected_alpha]
    test_guard_scores = selected_guard.score(test_details.raw_scores)
    test_guarded_actions = apply_guard_route(
        test_base_actions,
        test_guard_scores.signal,
        route=selected_route,  # type: ignore[arg-type]
    )

    base_test = _action_metrics(test_base_actions, test_true)
    guarded_test = _candidate_metrics(
        alpha=selected_alpha,
        route=selected_route,
        actions=test_guarded_actions,
        base_actions=test_base_actions,
        true_actions=test_true,
        legitimate_mask=test_legit,
        risky_mask=test_risky,
        novelty_signal=test_guard_scores.signal,
        calibration=selected_guard.calibration.as_dict(),
    )

    family_metrics = _family_metrics(
        test,
        raw_scores=test_details.raw_scores,
        signal=test_guard_scores.signal,
        base_actions=test_base_actions,
        guarded_actions=test_guarded_actions,
    )

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "scope": SCOPE,
        "promotion_eligible": False,
        "serving_change_authorized": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "dataset_sha256": manifest.content_sha256,
        "split_manifest_sha256": split.content_sha256,
        "champion_model_version": champion.version,
        "champion_remains_unchanged": True,
        "standalone_novelty_model_version": getattr(if2, "version", "unknown"),
        "standalone_novelty_reference_mode": getattr(if2, "reference_mode", "unknown"),
        "important_semantics": {
            "raw_score": "-IsolationForest.score_samples; higher means more anomalous in RazorTrust",
            "novelty_signal": "raw score at or above a held-out legitimate calibration threshold",
            "novelty_override": "policy event only when an otherwise-RELEASE action is changed",
            "empirical_tail_probability": "diagnostic rank only; not a fraud probability or production conformal claim",
        },
        "selection_protocol": {
            "alpha_grid": list(ALPHA_GRID),
            "routes": list(ROUTES),
            "selection_partition": "merchant-isolated calibration",
            "evaluation_partition": "merchant-isolated untouched test",
            "criterion": "minimum normalized expected cost; tie-break benign release blocks then EVIDENCE route",
        },
        "candidate_calibration_results": candidates,
        "selected_guard": selected,
        "test_base_champion": base_test,
        "test_selected_guard": guarded_test,
        "test_delta": {
            "expected_cost_units": round(
                float(guarded_test["expected_cost_units"] - base_test["expected_cost_units"]), 8
            ),
            "false_release_rate": round(
                float(guarded_test["false_release_rate"] - base_test["false_release_rate"]), 8
            ),
        },
        "family_metrics": family_metrics,
        "notes": [
            "IF@2 is evaluated here as an independent safety guard; its percentile is NOT fed into the champion XGBoost classifier.",
            "This fresh synthetic benchmark is research/mechanics evidence only and cannot promote a production financial action.",
            "The current synthetic generator still has family-template limitations; Phase 2 will add held-out-family and benign-novelty stress tests.",
        ],
    }
    report["content_sha256"] = _sha256_json(report)
    _write_json(output / "r4d3_novelty_guard_report.json", report)
    _write_json(output / "selected_guard_calibration.json", selected_guard.calibration.as_dict())

    print(
        json.dumps(
            {
                "status": "OK",
                "scope": SCOPE,
                "promotion_eligible": False,
                "serving_change_authorized": False,
                "champion_model_version": champion.version,
                "standalone_novelty_model_version": getattr(if2, "version", "unknown"),
                "selected_target_false_alarm_rate": selected_alpha,
                "selected_route": selected_route,
                "selected_raw_threshold": selected_guard.calibration.raw_threshold,
                "calibration_expected_cost_units": selected["expected_cost_units"],
                "test_champion_expected_cost_units": base_test["expected_cost_units"],
                "test_guard_expected_cost_units": guarded_test["expected_cost_units"],
                "test_champion_false_release_rate": base_test["false_release_rate"],
                "test_guard_false_release_rate": guarded_test["false_release_rate"],
                "test_legitimate_signal_rate": guarded_test["legitimate_signal_rate"],
                "test_risk_signal_rate": guarded_test["risk_signal_rate"],
                "test_legitimate_release_block_rate": guarded_test["legitimate_release_block_rate"],
                "test_risky_release_rescue_rate": guarded_test["risky_release_rescue_rate"],
                "report_sha256": report["content_sha256"],
                "output": str(output),
            },
            indent=2,
        )
    )


def _partition(frame: pd.DataFrame, merchants: list[str]) -> pd.DataFrame:
    result = frame[frame["merchant_id"].isin(merchants)].reset_index(drop=True)
    if result.empty:
        raise ValueError("empty benchmark partition")
    return result


def _threshold_actions(
    probabilities: np.ndarray, release_threshold: float, escalate_threshold: float
) -> np.ndarray:
    actions: list[HoldDecision] = []
    for row in probabilities:
        if float(row[0]) >= release_threshold:
            actions.append(HoldDecision.RELEASE)
        elif float(row[2]) >= escalate_threshold:
            actions.append(HoldDecision.ESCALATE)
        else:
            actions.append(HoldDecision.EVIDENCE_NEEDED)
    return np.asarray(actions, dtype=object)


def _true_actions(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        [HoldDecision(str(value)) for value in frame["operational_target"]], dtype=object
    )


def _action_metrics(actions: np.ndarray, true_actions: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(DEFAULT_COST_MATRIX.matrix, dtype=float)
    action_idx = np.asarray([LABEL_INDEX[HoldDecision(str(value))] for value in actions], dtype=int)
    true_idx = np.asarray(
        [LABEL_INDEX[HoldDecision(str(value))] for value in true_actions], dtype=int
    )
    non_release = true_idx != LABEL_INDEX[HoldDecision.RELEASE]
    false_release_rate = (
        float(np.mean(action_idx[non_release] == LABEL_INDEX[HoldDecision.RELEASE]))
        if np.any(non_release)
        else 0.0
    )
    return {
        "expected_cost_units": round(float(np.mean(matrix[action_idx, true_idx])), 8),
        "false_release_rate": round(false_release_rate, 8),
        "release_rate": round(float(np.mean(action_idx == LABEL_INDEX[HoldDecision.RELEASE])), 8),
        "evidence_rate": round(
            float(np.mean(action_idx == LABEL_INDEX[HoldDecision.EVIDENCE_NEEDED])), 8
        ),
        "escalation_rate": round(
            float(np.mean(action_idx == LABEL_INDEX[HoldDecision.ESCALATE])), 8
        ),
    }


def _candidate_metrics(
    *,
    alpha: float,
    route: str,
    actions: np.ndarray,
    base_actions: np.ndarray,
    true_actions: np.ndarray,
    legitimate_mask: np.ndarray,
    risky_mask: np.ndarray,
    novelty_signal: np.ndarray,
    calibration: dict[str, object],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "target_false_alarm_rate": alpha,
        "route": route,
        "calibration": calibration,
        **_action_metrics(actions, true_actions),
    }
    base_release = np.asarray(
        [HoldDecision(str(value)) == HoldDecision.RELEASE for value in base_actions], dtype=bool
    )
    legitimate_release = legitimate_mask & base_release
    risky_release = risky_mask & base_release
    metrics.update(
        {
            "legitimate_signal_rate": round(
                float(np.mean(novelty_signal[legitimate_mask])) if np.any(legitimate_mask) else 0.0,
                8,
            ),
            "risk_signal_rate": round(
                float(np.mean(novelty_signal[risky_mask])) if np.any(risky_mask) else 0.0, 8
            ),
            "legitimate_release_block_rate": round(
                float(np.mean(novelty_signal[legitimate_release]))
                if np.any(legitimate_release)
                else 0.0,
                8,
            ),
            "risky_release_rescue_rate": round(
                float(np.mean(novelty_signal[risky_release])) if np.any(risky_release) else 0.0, 8
            ),
            "legitimate_release_count": int(np.count_nonzero(legitimate_release)),
            "risky_release_count": int(np.count_nonzero(risky_release)),
        }
    )
    return metrics


def _family_metrics(
    frame: pd.DataFrame,
    *,
    raw_scores: np.ndarray,
    signal: np.ndarray,
    base_actions: np.ndarray,
    guarded_actions: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    families = frame["scenario_family"].astype(str).to_numpy()
    for family in sorted(set(families)):
        mask = families == family
        family_raw = raw_scores[mask]
        family_signal = signal[mask]
        family_base = base_actions[mask]
        family_guarded = guarded_actions[mask]
        rows.append(
            {
                "scenario_family": family,
                "true_risk_state": str(frame.loc[np.flatnonzero(mask)[0], "true_risk_state"]),
                "n": int(np.count_nonzero(mask)),
                "mean_raw_score": round(float(np.mean(family_raw)), 8),
                "p95_raw_score": round(float(np.quantile(family_raw, 0.95)), 8),
                "signal_rate": round(float(np.mean(family_signal)), 8),
                "base_release_rate": round(
                    float(
                        np.mean([HoldDecision(str(v)) == HoldDecision.RELEASE for v in family_base])
                    ),
                    8,
                ),
                "guard_release_rate": round(
                    float(
                        np.mean(
                            [HoldDecision(str(v)) == HoldDecision.RELEASE for v in family_guarded]
                        )
                    ),
                    8,
                ),
            }
        )
    return rows


def _sha256_json(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
