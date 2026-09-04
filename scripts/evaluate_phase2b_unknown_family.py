from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from razortrust.costs import DEFAULT_COST_MATRIX
from razortrust.domain import HoldDecision
from razortrust.ml.autoencoder_ood import (
    COMPACT_ARCHITECTURE,
    AutoencoderOODScorer,
    calibrate_upper_tail,
)
from razortrust.ml.dataset import build_training_frame, model_matrix
from razortrust.ml.modeling import LABEL_TO_INDEX, train_model_bundle
from razortrust.ml.phase2b_unknown import (
    CONSENSUS_RULES,
    apply_evidence_only_guard,
    consensus_signal,
)
from razortrust.ml.splits import create_split_manifest, write_split_manifest
from razortrust.synthetic import (
    LEGITIMATE_FAMILIES,
    RISK_FAMILIES,
    TrueRiskState,
    generate_dataset,
    write_dataset,
)

ALPHAS = (0.005, 0.01, 0.02, 0.05)
SCOPE = "SYNTHETIC_RESEARCH_ONLY_PHASE2B_UNKNOWN_FAMILY_NO_PROMOTION"
ROUTER_VERSION = "novelty-evidence-router@1-research"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2B end-to-end unknown-family disagreement benchmark"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--merchants-per-family", type=int, default=50)
    parser.add_argument("--transactions-per-merchant", type=int, default=60)
    parser.add_argument("--xgb-estimators", type=int, default=140)
    parser.add_argument("--ae-epochs", type=int, default=70)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    champion_release = ROOT / "artifacts" / "tier0" / "model-release"
    champion_public_key = ROOT / "artifacts" / "tier0" / "model-release-public-key.txt"
    if not champion_release.exists() or not champion_public_key.exists():
        raise FileNotFoundError("current signed champion artifacts are required")

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
    frame = build_training_frame(merchants, transactions, holds).reset_index(drop=True)

    families = tuple(LEGITIMATE_FAMILIES) + tuple(RISK_FAMILIES)
    candidate_case_rows: dict[tuple[float, str], list[dict[str, Any]]] = {
        (alpha, rule): [] for alpha in ALPHAS for rule in CONSENSUS_RULES
    }
    fold_metadata: list[dict[str, Any]] = []

    for family_index, held_out_family in enumerate(families):
        held_out = frame[frame["scenario_family"].astype(str) == held_out_family].reset_index(
            drop=True
        )
        development = frame[frame["scenario_family"].astype(str) != held_out_family].reset_index(
            drop=True
        )
        if held_out.empty or development.empty:
            raise RuntimeError(f"invalid family fold: {held_out_family}")

        fold_hash = hashlib.sha256(
            f"{manifest.content_sha256}|{held_out_family}|phase2b".encode()
        ).hexdigest()
        fold_seed = args.seed + 1000 + family_index
        split = create_split_manifest(development, fold_hash, seed=fold_seed)
        write_split_manifest(output / f"split_{held_out_family}.json", split)

        training_result = train_model_bundle(
            development,
            split,
            seed=fold_seed,
            n_estimators=args.xgb_estimators,
        )
        bundle = training_result.bundle
        train = _partition(development, split.train_merchants)
        calibration = _partition(development, split.calibration_merchants)
        train_legit = train[
            train["true_risk_state"].astype(str) == TrueRiskState.LEGITIMATE
        ].reset_index(drop=True)
        calibration_legit = calibration[
            calibration["true_risk_state"].astype(str) == TrueRiskState.LEGITIMATE
        ].reset_index(drop=True)
        if len(calibration_legit) < 20:
            raise RuntimeError(
                f"insufficient legitimate calibration rows in fold {held_out_family}: {len(calibration_legit)}"
            )

        ae = AutoencoderOODScorer(
            architecture=COMPACT_ARCHITECTURE,
            seed=fold_seed + 200,
            epochs=args.ae_epochs,
            patience=10,
        ).fit(model_matrix(train_legit), train_legit["merchant_id"].astype(str))
        cal_ae = ae.score_details(model_matrix(calibration_legit))
        held_ae = ae.score_details(model_matrix(held_out))

        anomaly_scorer = bundle.anomaly_scorer
        if getattr(anomaly_scorer, "version", "") != "isolation-forest@2":
            raise RuntimeError(
                f"Phase 2B requires isolation-forest@2; got {getattr(anomaly_scorer, 'version', 'unknown')}"
            )
        cal_if = anomaly_scorer.score_details(
            model_matrix(calibration_legit), calibration_legit["cohort"]
        )
        held_if = anomaly_scorer.score_details(model_matrix(held_out), held_out["cohort"])

        probabilities = bundle.predict_proba(model_matrix(held_out), held_out["cohort"])
        base_actions = _threshold_actions(
            probabilities,
            bundle.thresholds.release,
            bundle.thresholds.escalate,
        )
        true_labels = _true_labels(held_out)
        is_legitimate = (
            held_out["true_risk_state"].astype(str).to_numpy() == TrueRiskState.LEGITIMATE
        )

        fold_metadata.append(
            {
                "held_out_family": held_out_family,
                "true_risk_state": "LEGITIMATE" if bool(np.all(is_legitimate)) else "RISKY",
                "n": len(held_out),
                "calibration_method": bundle.calibration_method,
                "release_threshold": bundle.thresholds.release,
                "escalate_threshold": bundle.thresholds.escalate,
                "ae_best_epoch": ae.best_epoch,
                "ae_validation_loss": ae.best_validation_loss,
            }
        )

        for alpha in ALPHAS:
            recon_threshold = calibrate_upper_tail(
                cal_ae.reconstruction_error, target_false_alarm_rate=alpha
            ).threshold
            latent_threshold = calibrate_upper_tail(
                cal_ae.latent_mahalanobis_sq, target_false_alarm_rate=alpha
            ).threshold
            if_threshold = calibrate_upper_tail(
                cal_if.raw_scores, target_false_alarm_rate=alpha
            ).threshold

            recon_signal = held_ae.reconstruction_error >= recon_threshold
            latent_signal = held_ae.latent_mahalanobis_sq >= latent_threshold
            if_signal = held_if.raw_scores >= if_threshold

            for rule in CONSENSUS_RULES:
                novelty_signal = consensus_signal(rule, recon_signal, latent_signal, if_signal)
                guarded_actions = apply_evidence_only_guard(base_actions, novelty_signal)
                key = (alpha, rule)
                for row_index in range(len(held_out)):
                    candidate_case_rows[key].append(
                        {
                            "held_out_family": held_out_family,
                            "true_risk_state": str(held_out.loc[row_index, "true_risk_state"]),
                            "operational_target": str(
                                held_out.loc[row_index, "operational_target"]
                            ),
                            "merchant_id": str(held_out.loc[row_index, "merchant_id"]),
                            "base_action": str(base_actions[row_index]),
                            "guarded_action": str(guarded_actions[row_index]),
                            "novelty_signal": bool(novelty_signal[row_index]),
                            "reconstruction_error": float(held_ae.reconstruction_error[row_index]),
                            "latent_mahalanobis_sq": float(
                                held_ae.latent_mahalanobis_sq[row_index]
                            ),
                            "if_raw_score": float(held_if.raw_scores[row_index]),
                            "true_label_index": int(true_labels[row_index]),
                            "base_action_index": int(
                                LABEL_TO_INDEX[HoldDecision(base_actions[row_index])]
                            ),
                            "guarded_action_index": int(
                                LABEL_TO_INDEX[HoldDecision(guarded_actions[row_index])]
                            ),
                        }
                    )

    candidate_metrics: list[dict[str, Any]] = []
    for (alpha, rule), rows in candidate_case_rows.items():
        metrics = _aggregate_candidate(alpha=alpha, rule=rule, rows=rows)
        candidate_metrics.append(metrics)

    frontier = _pareto_frontier(candidate_metrics)
    frontier_keys = {
        (float(row["target_false_alarm_rate"]), str(row["consensus_rule"])) for row in frontier
    }
    for row in candidate_metrics:
        row["pareto_frontier"] = (
            float(row["target_false_alarm_rate"]),
            str(row["consensus_rule"]),
        ) in frontier_keys

    recommendation = min(
        frontier or candidate_metrics,
        key=lambda row: (
            float(row["guard_expected_cost_units"]),
            float(row["guard_false_release_rate"]),
            float(row["unknown_legitimate_release_diversion_rate"]),
            -float(row["unknown_risk_release_rescue_rate"]),
        ),
    )
    selected_key = (
        float(recommendation["target_false_alarm_rate"]),
        str(recommendation["consensus_rule"]),
    )
    selected_rows = candidate_case_rows[selected_key]
    selected_frame = pd.DataFrame(selected_rows)
    selected_frame.to_csv(output / "unknown_family_case_results.csv", index=False)
    family_metrics = _family_metrics(selected_rows)

    report: dict[str, Any] = {
        "status": "OK",
        "scope": SCOPE,
        "created_at": datetime.now(UTC).isoformat(),
        "promotion_eligible": False,
        "serving_change_authorized": False,
        "production_action_eligible": False,
        "current_serving_champion": "xgb-if-settlement@2",
        "research_fold_model_contract": "xgb-if-settlement@2-structure-retrained-with-family-excluded",
        "if_model_version": "isolation-forest@2",
        "autoencoder_model_version": "autoencoder-ood@2-research",
        "autoencoder_architecture": COMPACT_ARCHITECTURE.name,
        "router_version": ROUTER_VERSION,
        "router_semantics": "novelty can only divert an otherwise RELEASE case to EVIDENCE_NEEDED; novelty alone never escalates",
        "dataset_manifest_sha256": manifest.content_sha256,
        "folds": fold_metadata,
        "candidate_metrics": candidate_metrics,
        "pareto_frontier": frontier,
        "research_recommendation": recommendation,
        "recommended_family_metrics": family_metrics,
        "case_artifact": "unknown_family_case_results.csv",
        "interpretation_contract": {
            "unknown_legitimate_signal": "detection of an unseen legitimate behavior family; not fraud and not automatically an error",
            "unknown_legitimate_release_diversion": "merchant-friction metric: RELEASE candidates diverted to evidence",
            "unknown_risk_release_rescue": "safety metric: risky RELEASE candidates diverted to evidence",
            "selection_warning": "all family-fold results are research/development measurements and cannot authorize promotion",
        },
    }
    report["report_sha256"] = _sha256_json(report)
    _write_json(output / "phase2b_unknown_family_report.json", report)

    summary = {
        "status": "OK",
        "scope": SCOPE,
        "promotion_eligible": False,
        "serving_change_authorized": False,
        "recommended_alpha": recommendation["target_false_alarm_rate"],
        "recommended_rule": recommendation["consensus_rule"],
        "base_expected_cost_units": recommendation["base_expected_cost_units"],
        "guard_expected_cost_units": recommendation["guard_expected_cost_units"],
        "base_false_release_rate": recommendation["base_false_release_rate"],
        "guard_false_release_rate": recommendation["guard_false_release_rate"],
        "unknown_legitimate_signal_rate": recommendation["unknown_legitimate_signal_rate"],
        "unknown_legitimate_release_diversion_rate": recommendation[
            "unknown_legitimate_release_diversion_rate"
        ],
        "unknown_risk_signal_rate": recommendation["unknown_risk_signal_rate"],
        "unknown_risk_release_rescue_rate": recommendation["unknown_risk_release_rescue_rate"],
        "pareto_candidates": len(frontier),
        "report_sha256": report["report_sha256"],
        "output": str(output),
    }
    print(json.dumps(summary, indent=2))


def _aggregate_candidate(*, alpha: float, rule: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    legit = frame["true_risk_state"].astype(str).to_numpy() == TrueRiskState.LEGITIMATE
    risk = ~legit
    signal = frame["novelty_signal"].astype(bool).to_numpy()
    base_actions = frame["base_action"].astype(str).to_numpy()
    guarded_actions = frame["guarded_action"].astype(str).to_numpy()
    true_idx = frame["true_label_index"].to_numpy(dtype=int)
    base_idx = frame["base_action_index"].to_numpy(dtype=int)
    guard_idx = frame["guarded_action_index"].to_numpy(dtype=int)
    matrix = np.asarray(DEFAULT_COST_MATRIX.matrix, dtype=float)

    legit_release = legit & (base_actions == str(HoldDecision.RELEASE))
    risk_release = risk & (base_actions == str(HoldDecision.RELEASE))
    non_release_truth = true_idx != LABEL_TO_INDEX[HoldDecision.RELEASE]

    family_diversions: list[float] = []
    risk_family_rescue: list[float] = []
    for family in sorted(frame["held_out_family"].astype(str).unique()):
        mask = frame["held_out_family"].astype(str).to_numpy() == family
        family_legit = bool(np.all(legit[mask]))
        base_release_mask = mask & (base_actions == str(HoldDecision.RELEASE))
        if family_legit:
            family_diversions.append(
                float(np.mean(signal[base_release_mask])) if np.any(base_release_mask) else 0.0
            )
        else:
            risk_family_rescue.append(
                float(np.mean(signal[base_release_mask])) if np.any(base_release_mask) else 0.0
            )

    return {
        "target_false_alarm_rate": alpha,
        "consensus_rule": rule,
        "base_expected_cost_units": round(float(np.mean(matrix[base_idx, true_idx])), 8),
        "guard_expected_cost_units": round(float(np.mean(matrix[guard_idx, true_idx])), 8),
        "expected_cost_delta_units": round(
            float(np.mean(matrix[guard_idx, true_idx]) - np.mean(matrix[base_idx, true_idx])), 8
        ),
        "base_false_release_rate": round(
            float(np.mean(base_idx[non_release_truth] == LABEL_TO_INDEX[HoldDecision.RELEASE]))
            if np.any(non_release_truth)
            else 0.0,
            8,
        ),
        "guard_false_release_rate": round(
            float(np.mean(guard_idx[non_release_truth] == LABEL_TO_INDEX[HoldDecision.RELEASE]))
            if np.any(non_release_truth)
            else 0.0,
            8,
        ),
        "unknown_legitimate_signal_rate": round(float(np.mean(signal[legit])), 8),
        "unknown_risk_signal_rate": round(float(np.mean(signal[risk])), 8),
        "unknown_legitimate_release_diversion_rate": round(
            float(np.mean(signal[legit_release])) if np.any(legit_release) else 0.0, 8
        ),
        "unknown_risk_release_rescue_rate": round(
            float(np.mean(signal[risk_release])) if np.any(risk_release) else 0.0, 8
        ),
        "unknown_risk_release_count": int(np.count_nonzero(risk_release)),
        "unknown_legitimate_release_count": int(np.count_nonzero(legit_release)),
        "worst_legitimate_family_release_diversion_rate": round(
            max(family_diversions) if family_diversions else 0.0, 8
        ),
        "minimum_risk_family_release_rescue_rate": round(
            min(risk_family_rescue) if risk_family_rescue else 0.0, 8
        ),
        "base_release_rate": round(float(np.mean(base_actions == str(HoldDecision.RELEASE))), 8),
        "guard_release_rate": round(
            float(np.mean(guarded_actions == str(HoldDecision.RELEASE))), 8
        ),
        "guard_evidence_rate": round(
            float(np.mean(guarded_actions == str(HoldDecision.EVIDENCE_NEEDED))), 8
        ),
        "guard_escalation_rate": round(
            float(np.mean(guarded_actions == str(HoldDecision.ESCALATE))), 8
        ),
    }


def _pareto_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier: list[dict[str, Any]] = []
    for candidate in rows:
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            no_worse = (
                float(other["guard_expected_cost_units"])
                <= float(candidate["guard_expected_cost_units"])
                and float(other["guard_false_release_rate"])
                <= float(candidate["guard_false_release_rate"])
                and float(other["unknown_legitimate_release_diversion_rate"])
                <= float(candidate["unknown_legitimate_release_diversion_rate"])
                and float(other["unknown_risk_release_rescue_rate"])
                >= float(candidate["unknown_risk_release_rescue_rate"])
            )
            strictly_better = (
                float(other["guard_expected_cost_units"])
                < float(candidate["guard_expected_cost_units"])
                or float(other["guard_false_release_rate"])
                < float(candidate["guard_false_release_rate"])
                or float(other["unknown_legitimate_release_diversion_rate"])
                < float(candidate["unknown_legitimate_release_diversion_rate"])
                or float(other["unknown_risk_release_rescue_rate"])
                > float(candidate["unknown_risk_release_rescue_rate"])
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(dict(candidate))
    return sorted(
        frontier,
        key=lambda row: (
            float(row["guard_expected_cost_units"]),
            float(row["unknown_legitimate_release_diversion_rate"]),
            -float(row["unknown_risk_release_rescue_rate"]),
        ),
    )


def _family_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    output: list[dict[str, Any]] = []
    matrix = np.asarray(DEFAULT_COST_MATRIX.matrix, dtype=float)
    for family in sorted(frame["held_out_family"].astype(str).unique()):
        group = frame[frame["held_out_family"].astype(str) == family]
        base_idx = group["base_action_index"].to_numpy(dtype=int)
        guard_idx = group["guarded_action_index"].to_numpy(dtype=int)
        true_idx = group["true_label_index"].to_numpy(dtype=int)
        base_release = group["base_action"].astype(str).to_numpy() == str(HoldDecision.RELEASE)
        signal = group["novelty_signal"].astype(bool).to_numpy()
        output.append(
            {
                "held_out_family": family,
                "true_risk_state": str(group.iloc[0]["true_risk_state"]),
                "n": len(group),
                "novelty_signal_rate": round(float(np.mean(signal)), 8),
                "base_release_rate": round(float(np.mean(base_release)), 8),
                "release_diversion_or_rescue_rate": round(
                    float(np.mean(signal[base_release])) if np.any(base_release) else 0.0, 8
                ),
                "base_expected_cost_units": round(float(np.mean(matrix[base_idx, true_idx])), 8),
                "guard_expected_cost_units": round(float(np.mean(matrix[guard_idx, true_idx])), 8),
            }
        )
    return output


def _threshold_actions(
    probabilities: np.ndarray, release_threshold: float, escalate_threshold: float
) -> np.ndarray:
    actions = np.full(len(probabilities), HoldDecision.EVIDENCE_NEEDED, dtype=object)
    release = probabilities[:, 0] >= release_threshold
    escalate = (~release) & (probabilities[:, 2] >= escalate_threshold)
    actions[release] = HoldDecision.RELEASE
    actions[escalate] = HoldDecision.ESCALATE
    return actions


def _true_labels(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        [LABEL_TO_INDEX[HoldDecision(value)] for value in frame["operational_target"]],
        dtype=int,
    )


def _partition(frame: pd.DataFrame, merchants: list[str]) -> pd.DataFrame:
    result = frame[frame["merchant_id"].isin(merchants)].reset_index(drop=True)
    if result.empty:
        raise RuntimeError("empty merchant partition")
    return result


def _sha256_json(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
